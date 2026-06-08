#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
極簡 few-shot 範例(座標定位版):給「多張範例影像 + 座標答案」,讓 VLM 找出新影像裡角的位置。

這是 ../fewshot_minimal_example/ 的姊妹版。骨架一模一樣(system + user/assistant 交替範例
+ 最後一題),只差在**任務**:
    - 上一個範例:判斷角在十字的「哪個方向」→ 輸出 left/center/right(枚舉)
    - 這個範例:  找出角的「座標在哪」      → 輸出 (x, y)(整數)

★這個版本最重要的新觀念:視覺示範式 few-shot(visual-demonstration few-shot)★
    - 範例影像上會用「綠色十字」把角的正確位置畫出來,答案就是綠色十字的 (x, y)。
    - 真正要問的影像「沒有」綠色十字,模型必須自己從影像內容找出角。
    - 換句話說:範例是「已經把答案畫在圖上的示範」,題目是「擦掉答案要你自己做」。

兩種跑法(跟上一個範例一樣):
    python minimal_fewshot_locate.py --dry-run   # 不需要 vLLM、不需要圖片,只看 messages 結構
    python minimal_fewshot_locate.py             # 真的呼叫 vLLM
"""

import argparse
import base64
import json
import os

# 注意:openai 只有「真的呼叫模型」時才需要,所以放到 main() 裡才 import。
# 這樣 --dry-run(只看結構)就算沒裝 openai、沒有 server 也能跑。


# ===========================================================================
# (A) 假設所有影像都是固定尺寸(你的資料就是固定尺寸)。
#     用 pixel 座標時,要讓模型知道影像多大,它才知道座標範圍。
#     如果你的影像不是這個大小,改這兩個數字即可。
# ===========================================================================
IMAGE_W, IMAGE_H = 1024, 1024


# ===========================================================================
# (B) 系統指令:任務說明 + 座標規則 + 綠/紅十字的處理方式。整段對話只出現一次。
# ===========================================================================
SYSTEM_PROMPT = (
    "你是 SEM 影像座標定位器。找出影像中目標結構『角』"
    "(兩條亮邊垂直相交的 L 形頂點)的位置,輸出它的 (x, y) 像素座標。\n"
    f"影像尺寸固定為 {IMAGE_W}×{IMAGE_H};原點 (0,0) 在左上角,x 向右為正、y 向下為正。\n"
    "範例影像會用「綠色」十字標出角的正確位置;真正要回答的影像「不會」有綠色十字,"
    "你要自己從影像內容找出角。\n"
    "影像中若出現「紅色」十字,那是機台 camera 中心 marker,與本任務無關,請完全忽略。\n"
    "只輸出一個 JSON 物件,例如 {\"x\": 487, \"y\": 312},不要有多餘文字。"
)

# 問每一張圖都用同一句話(範例和正式問題都用它,模型才覺得格式一致)。
QUESTION_TEXT = "這張 SEM 影像中,目標結構的角在哪裡?請輸出 (x, y) 座標。"


# ===========================================================================
# (C) few-shot 範例:每一筆 = 一張「畫了綠色十字」的圖 + 綠色十字中心的座標。
#     重點:這裡的圖檔本身畫了綠十字,answer 的 (x, y) 要跟綠十字中心對齊。
# ===========================================================================
EXAMPLES = [
    {"image": "images/example_1_green_cross.png", "answer": {"x": 312, "y": 458}},
    {"image": "images/example_2_green_cross.png", "answer": {"x": 690, "y": 205}},
    {"image": "images/example_3_green_cross.png", "answer": {"x": 540, "y": 540}},
]

# 正式要問的新影像:沒有綠色十字,要模型自己找角。
QUESTION_IMAGE = "images/question_no_cross.png"


# ===========================================================================
# (D) 小工具:把圖片檔讀成 data URI(base64 字串),才能塞進 message。
# ===========================================================================
def load_image_as_data_uri(path):
    if not os.path.exists(path):
        # 教學用:就算沒有真的圖,也能看懂結構 —— 用一段假字串代替真正的 base64。
        return f"data:image/png;base64,(這裡是 <{path}> 的影像內容)"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return "data:image/png;base64," + encoded


# 一個「使用者(user)」turn:一句問題文字 + 一張圖。
def make_user_turn(image_path):
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": QUESTION_TEXT},
            {"type": "image_url", "image_url": {"url": load_image_as_data_uri(image_path)}},
        ],
    }


# 一個「助理(assistant)」turn:模型在範例裡「應該」給的座標答案,寫成 JSON 字串。
def make_assistant_turn(answer):
    return {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}


# ===========================================================================
# (E) ★整個範例的核心★ 把多張範例組成 few-shot 的 messages。
#     這個函式跟 ../fewshot_minimal_example/ 裡的「一模一樣」——
#     few-shot 的骨架不會因為任務不同而改變,變的只有範例內容與答案格式。
#
#         system     ← 任務說明(只 1 段)
#         user       ← 範例1的圖(有綠十字)        ┐
#         assistant  ← 範例1的答案 {"x":312,"y":458} ┘ 一組示範
#         user       ← 範例2的圖(有綠十字)        ┐
#         assistant  ← 範例2的答案                  ┘
#         user       ← 範例3的圖(有綠十字)        ┐
#         assistant  ← 範例3的答案                  ┘
#         user       ← 要問的新圖(沒有綠十字)     ← 真正的問題
# ===========================================================================
def build_fewshot_messages(examples, question_image):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append(make_user_turn(ex["image"]))        # 先給範例的「題目」(畫了綠十字的圖)
        messages.append(make_assistant_turn(ex["answer"]))  # 再給範例的「答案」(綠十字座標)
    messages.append(make_user_turn(question_image))          # 最後才問新題目(沒有綠十字)
    return messages


# (選用)用 guided_json 強制模型只能輸出 {"x": 整數, "y": 整數}。
# 少了它模型也能答,只是偶爾格式會跑掉。
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0},
    },
    "required": ["x", "y"],
}


# ===========================================================================
# (F) 把組好的 messages 印成人看得懂的樣子(影像不印一大串 base64)。
# ===========================================================================
def print_messages(messages):
    for i, m in enumerate(messages):
        role = m["role"]
        if isinstance(m["content"], str):
            # system / assistant 的內容是純文字
            print(f"[{i}] {role:9s} | {m['content']}")
        else:
            # user 的內容是 list(文字 + 圖)
            parts = []
            for c in m["content"]:
                if c["type"] == "text":
                    parts.append(f"文字「{c['text']}」")
                else:
                    parts.append("圖片<一張影像>")
            print(f"[{i}] {role:9s} | " + " + ".join(parts))


# ===========================================================================
# (G) 主程式
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="極簡 few-shot 範例(座標定位版)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只印出 messages 結構,不呼叫 vLLM(不需要 server,也不需要真的圖片)")
    args = parser.parse_args()

    # 1) 組裝 few-shot messages
    messages = build_fewshot_messages(EXAMPLES, QUESTION_IMAGE)

    # 2) 先印出來,看 few-shot 長什麼樣
    print("=" * 70)
    print(f"組好的 messages 共 {len(messages)} 段"
          f"(1 段 system + {len(EXAMPLES)} 組範例×2 + 1 段提問):")
    print("=" * 70)
    print_messages(messages)
    print("=" * 70)

    if args.dry_run:
        print("(dry-run:沒有真的呼叫模型。拿掉 --dry-run 並設好 VLLM_BASE_URL 就會真的問。)")
        return

    # 3) 真的呼叫 vLLM
    from openai import OpenAI  # 只有這一步才需要 openai 套件

    client = OpenAI(
        base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
    )
    model = os.environ.get("VLM_MODEL", "Qwen3.6-27B")

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,                            # 定位類任務用 0,盡量穩定
        extra_body={"guided_json": JSON_SCHEMA},    # 強制輸出合法 JSON(選用但好用)
    )
    answer = completion.choices[0].message.content
    print("模型的回答(新影像中角的座標):")
    print(answer)


if __name__ == "__main__":
    main()

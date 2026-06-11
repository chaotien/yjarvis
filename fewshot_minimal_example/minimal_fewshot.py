#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
極簡 few-shot 範例:給「多張範例影像 + 答案」,讓 VLM 學會判斷一張新影像。

這是把 ../corner_judge_categorical/ 那個完整實驗「砍到只剩骨架」的教學版。
完整實驗有評分、統計、CSV、重試、多條件比較……這裡通通拿掉,
只留 few-shot 最核心的一件事:**怎麼把多張範例組成一串 messages**。

兩種跑法:
    python minimal_fewshot.py --dry-run   # 不需要 vLLM、不需要圖片,只印出 messages 結構
    python minimal_fewshot.py             # 真的呼叫 vLLM(要先設好 VLLM_BASE_URL)

建議第一次先跑 --dry-run,把印出來的結構看懂,再去接真的模型。
"""

import argparse
import base64
import json
import os

# 注意:openai 套件只有「真的呼叫模型」時才需要,所以放到 main() 裡才 import。
# 這樣 --dry-run(只看結構)就算沒裝 openai、沒有 server 也能跑。


# ===========================================================================
# (A) 系統指令:寫給模型看的「任務說明」。整段對話只出現一次,放在最前面。
#     可以想成:面試前先把「題目規則」一次講清楚。
# ===========================================================================
SYSTEM_PROMPT = (
    "你是 SEM 影像判斷器。判斷影像中『角』相對畫面中央十字線的位置。\n"
    # ★ 領域知識:不寫的話,模型只知道「找角」但不知道是哪個角。完整版見 ../corner_judge_categorical/。
    "角的定義:畫面右上有一個長方形 area(內含重複的亮暗 pattern),角是 area「左下方」的 L 形頂點\n"
    "—— 即 area 左邊垂直邊界與下邊水平邊界的交點(視覺上呈「└」,開口朝右上)。\n"
    "找邊界的方法:由右上向左下追蹤,重複 pattern 不再繼續的位置就是 area 邊界。\n"
    "offset_x 只能是 left / center / right;offset_y 只能是 above / center / below。\n"
    "只輸出一個 JSON 物件,不要有多餘文字。"
)

# 問每一張圖都用同一句話。範例和正式問題都用它,模型才會覺得「格式一致」。
QUESTION_TEXT = "這張影像中,角在十字中心的哪個方向?"


# ===========================================================================
# (B) few-shot 範例:每一筆 = 一張圖 + 這張圖的正確答案。
#     這就是「教材」。多放幾筆,模型就多看幾個示範。
#     (把圖片放進同資料夾的 images/ 底下,或改成你自己的路徑。)
# ===========================================================================
EXAMPLES = [
    {"image": "images/example_1.png", "answer": {"offset_x": "left",   "offset_y": "center"}},
    {"image": "images/example_2.png", "answer": {"offset_x": "right",  "offset_y": "above"}},
    {"image": "images/example_3.png", "answer": {"offset_x": "center", "offset_y": "center"}},
]

# 正式要問的新影像(沒有附答案,要模型自己判斷)。
QUESTION_IMAGE = "images/question.png"


# ===========================================================================
# (C) 小工具:把圖片檔讀成 data URI(base64 字串),才能塞進 message。
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


# 一個「助理(assistant)」turn:模型在範例裡「應該」給的答案,寫成 JSON 字串。
def make_assistant_turn(answer):
    return {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}


# ===========================================================================
# (D) ★整個範例的核心★ 把多張範例組成 few-shot 的 messages。
#
#     組出來的結構長這樣:
#         system     ← 任務說明(只 1 段)
#         user       ← 範例1的「題目」(圖)  ┐
#         assistant  ← 範例1的「答案」(JSON) ┘ 一組示範
#         user       ← 範例2的題目             ┐
#         assistant  ← 範例2的答案             ┘
#         user       ← 範例3的題目             ┐
#         assistant  ← 範例3的答案             ┘
#         user       ← 真正要問的新題目(只有圖,沒有答案)
#
#     重點:範例用 user(問)/ assistant(答)成對排列,
#     讓模型「以為」自己已經這樣答過好幾次,接著被丟了一個新問題。
#     這比把範例塞進一段文字裡更接近模型訓練時看到的格式,效果通常更好。
# ===========================================================================
def build_fewshot_messages(examples, question_image):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append(make_user_turn(ex["image"]))        # 先給範例的「題目」
        messages.append(make_assistant_turn(ex["answer"]))  # 再給範例的「答案」
    messages.append(make_user_turn(question_image))          # 最後才問新題目
    return messages


# (選用)用 guided_json 強制模型只能輸出符合這個格式的 JSON。
# 少了它模型也能答,只是偶爾格式會跑掉(多了文字、漏了引號……)。
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "offset_x": {"type": "string", "enum": ["left", "center", "right"]},
        "offset_y": {"type": "string", "enum": ["above", "center", "below"]},
    },
    "required": ["offset_x", "offset_y"],
}


# ===========================================================================
# (E) 把組好的 messages 印成人看得懂的樣子(影像不印一大串 base64)。
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
# (F) 主程式
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="極簡 few-shot 範例")
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
        temperature=0.0,                            # 判斷類任務用 0,盡量穩定
        extra_body={"guided_json": JSON_SCHEMA},    # 強制輸出合法 JSON(選用但好用)
    )
    answer = completion.choices[0].message.content
    print("模型的回答:")
    print(answer)


if __name__ == "__main__":
    main()

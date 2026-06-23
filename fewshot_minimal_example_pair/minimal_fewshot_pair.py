#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
極簡 few-shot 範例(成對比較版):一次給「兩張小圖(def + ref)」,讓 VLM 判斷它們的「差異」。

這是 ../fewshot_minimal_example/ 的姊妹版,把 ../bright_dot_cluster_judge/ 那個完整實驗
「砍到只剩骨架」。骨架還是一樣(system + user/assistant 交替範例 + 最後一題),
只差在**每一個 user turn 同時放兩張圖**,而且任務是比「兩張圖的差異」:

    - 第一個範例:看「一張」圖判斷方向        → 一個 user turn 放一張圖
    - 這個範例:  比「兩張」圖判斷差異形態    → 一個 user turn 放兩張圖(def 在前、ref 在後)

★這個版本最重要的新觀念:一個 user turn 裡放「多張圖」做比較★
    - def_patch(待測)和 ref_patch(參考)是同位置、同尺寸、已對齊的兩張小圖。
    - 模型要判斷的不是 def 自己多亮,而是「def 相對 ref 多出來的那塊亮區」長什麼形態:
        single  —— 一個孤立的小圓點
        cluster —— 多點 / 一大坨 / 整片(只要不是單一孤立小圓點,就算 cluster)
    - 兩張圖的「順序」(def 先、ref 後)是模型判斷「誰是誰」的唯一線索,所以順序要寫死。

兩種跑法(跟前面的範例一樣):
    python minimal_fewshot_pair.py --dry-run   # 不需要 vLLM、不需要圖片,只看 messages 結構
    python minimal_fewshot_pair.py             # 真的呼叫 vLLM

建議第一次先跑 --dry-run,把印出來的結構看懂,再去接真的模型。
"""

import argparse
import base64
import json
import os

# 注意:openai 只有「真的呼叫模型」時才需要,所以放到 main() 裡才 import。
# 這樣 --dry-run(只看結構)就算沒裝 openai、沒有 server 也能跑。


# ===========================================================================
# (A) 系統指令:任務說明 + 「怎麼比兩張圖」+ 類別規則。整段對話只出現一次,放最前面。
#     重點:要教模型「比差異」,不是「看 def 自己多亮」。完整版見 ../bright_dot_cluster_judge/。
# ===========================================================================
SYSTEM_PROMPT = (
    "你是半導體缺陷檢測的視覺判斷器。每次會給你兩張同位置、同尺寸、已對齊的小圖:"
    "第一張 = def_patch(待測),第二張 = ref_patch(參考)。\n"
    # ★ 領域知識:直接告訴模型「比差異、不是比絕對亮度」,推理路徑才會對。
    "比較方法:想像把兩張圖對齊後逐像素相減,只看『def 明顯比 ref 亮』的那塊新增亮區;"
    "兩張都亮或都暗的共同區域要忽略(那不是缺陷訊號)。\n"
    "判斷這塊新增亮區的『形態』,二選一:\n"
    "  single  —— 一個相對獨立、邊界清楚的小圓點(面積小、孤立、旁邊沒有其他同時變亮的點)。\n"
    "  cluster —— 多個鄰近同時變亮的點、或一大坨(blob)、或整片區域一起變亮。\n"
    "準則一句話:只要『不是單一孤立小圓點』,就歸 cluster。\n"
    "只輸出一個 JSON 物件,例如 {\"label\": \"single\"},不要有多餘文字。"
)

# 問每一對圖都用同一句話(範例和正式問題都用它,模型才覺得格式一致)。
# 這句話同時再次寫死「誰是 def、誰是 ref」—— 圖的順序是模型判斷誰是誰的唯一線索。
QUESTION_TEXT = (
    "以下兩張圖:第一張是 def_patch(待測)、第二張是 ref_patch(參考)。"
    "請判斷 def 相對 ref 新增的亮區是 single(單一孤立小圓點)還是 cluster(多點/成團/整片)。"
)


# ===========================================================================
# (B) few-shot 範例:每一筆 = 「一對」圖(def + ref)+ 這對圖的正確答案。
#     注意答案標的是「兩張圖的差異」,不是哪一張本身。
#     這是二元任務,所以兩個類別都放幾組(這裡 2 組 single + 2 組 cluster),
#     模型才不會以為「答案永遠是同一類」。
# ===========================================================================
EXAMPLES = [
    {"def": "images/example_1_def.png", "ref": "images/example_1_ref.png", "answer": {"label": "single"}},
    {"def": "images/example_2_def.png", "ref": "images/example_2_ref.png", "answer": {"label": "cluster"}},
    {"def": "images/example_3_def.png", "ref": "images/example_3_ref.png", "answer": {"label": "single"}},
    {"def": "images/example_4_def.png", "ref": "images/example_4_ref.png", "answer": {"label": "cluster"}},
]

# 正式要問的新一對圖(沒有附答案,要模型自己判斷)。
QUESTION_DEF = "images/question_def.png"
QUESTION_REF = "images/question_ref.png"


# ===========================================================================
# (C) 小工具:把圖片檔讀成 data URI(base64 字串),才能塞進 message。
#     (跟前面兩個範例一模一樣。)
# ===========================================================================
def load_image_as_data_uri(path):
    if not os.path.exists(path):
        # 教學用:就算沒有真的圖,也能看懂結構 —— 用一段假字串代替真正的 base64。
        return f"data:image/png;base64,(這裡是 <{path}> 的影像內容)"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return "data:image/png;base64," + encoded


# ★一個「使用者(user)」turn:一句問題文字 + 「兩張」圖(def 在前、ref 在後)。★
#   這是這個範例跟前面最大的不同 —— content 的 list 裡有「兩個」image_url。
#   def/ref 的先後順序就是模型判斷「誰是待測、誰是參考」的唯一依據,所以固定不能換。
def make_user_turn(def_path, ref_path):
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": QUESTION_TEXT},
            {"type": "image_url", "image_url": {"url": load_image_as_data_uri(def_path)}},  # 第一張 = def
            {"type": "image_url", "image_url": {"url": load_image_as_data_uri(ref_path)}},  # 第二張 = ref
        ],
    }


# 一個「助理(assistant)」turn:模型在範例裡「應該」給的答案,寫成 JSON 字串。
def make_assistant_turn(answer):
    return {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)}


# ===========================================================================
# (D) ★整個範例的核心★ 把多組「成對範例」組成 few-shot 的 messages。
#
#     跟前面範例的差別只有一個:每個 user turn 帶「兩張」圖,不是一張。
#     few-shot 的骨架(system → 範例的 user/assistant 交替 → 最後提問)完全沒變。
#
#         system     ← 任務說明 + 比較方法(只 1 段)
#         user       ← 範例1的「一對圖」[def1, ref1]   ┐
#         assistant  ← 範例1的答案 {"label":"single"}   ┘ 一組示範
#         user       ← 範例2的「一對圖」[def2, ref2]   ┐
#         assistant  ← 範例2的答案 {"label":"cluster"}  ┘
#         …                                            (最多放 len(EXAMPLES) 組)
#         user       ← 要問的新「一對圖」[defq, refq]  ← 真正的問題(只有圖,沒有答案)
#
#     重點:範例一樣用 user(問)/ assistant(答)成對排列,只是「問」的部分從
#     「一張圖」變成「一對圖」。模型會「以為」自己已經比過好幾對,接著被丟一對新的。
# ===========================================================================
def build_fewshot_messages(examples, question_def, question_ref):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in examples:
        messages.append(make_user_turn(ex["def"], ex["ref"]))   # 先給範例的「一對圖」
        messages.append(make_assistant_turn(ex["answer"]))      # 再給範例的「答案」
    messages.append(make_user_turn(question_def, question_ref))  # 最後才問新的一對圖
    return messages


# (選用)用 guided_json 強制模型只能輸出 {"label": "single" 或 "cluster"}。
# 少了它模型也能答,只是偶爾格式會跑掉(多了文字、給了類別外的字……)。
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["single", "cluster"]},
    },
    "required": ["label"],
}


# ===========================================================================
# (E) 把組好的 messages 印成人看得懂的樣子(影像不印一大串 base64)。
#     跟前面範例幾乎一樣,只是 user turn 現在會印出「兩張圖」並標明誰是 def、誰是 ref。
# ===========================================================================
def print_messages(messages):
    for i, m in enumerate(messages):
        role = m["role"]
        if isinstance(m["content"], str):
            # system / assistant 的內容是純文字
            print(f"[{i}] {role:9s} | {m['content']}")
        else:
            # user 的內容是 list(文字 + 兩張圖)
            parts = []
            img_seen = 0
            for c in m["content"]:
                if c["type"] == "text":
                    parts.append(f"文字「{c['text']}」")
                else:
                    img_seen += 1
                    tag = "def_patch" if img_seen == 1 else "ref_patch"
                    parts.append(f"圖片<{tag}>")
            print(f"[{i}] {role:9s} | " + " + ".join(parts))


# ===========================================================================
# (F) 主程式
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="極簡 few-shot 範例(成對比較版)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只印出 messages 結構,不呼叫 vLLM(不需要 server,也不需要真的圖片)")
    args = parser.parse_args()

    # 1) 組裝 few-shot messages
    messages = build_fewshot_messages(EXAMPLES, QUESTION_DEF, QUESTION_REF)

    # 2) 先印出來,看 few-shot 長什麼樣
    print("=" * 70)
    print(f"組好的 messages 共 {len(messages)} 段"
          f"(1 段 system + {len(EXAMPLES)} 組範例×2 + 1 段提問;每個 user turn 帶兩張圖):")
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
    print("模型的回答(這對 patch 的差異是 single 還是 cluster):")
    print(answer)


if __name__ == "__main__":
    main()

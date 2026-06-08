# 極簡 Few-shot 範例(從這裡開始)

這個資料夾是給**第一次接觸 LLM / VLM** 的人,用最少的程式碼看懂一件事:

> **怎麼「給模型看幾個範例」,讓它學會判斷一張新影像?**

旁邊的 `corner_judge_categorical/` 和 `corner_locate_coords/` 是完整實驗(有評分、
統計、多條件比較……)。那些東西第一次看會很雜。這裡把它們**砍到只剩骨架**,
只有一個檔案、一個函式是重點。

---

## 1. 什麼是 few-shot?

跟模型對話時,你可以選擇:

- **Zero-shot(零範例)**:直接問「這張圖的角在哪個方向?」——模型只能憑自己猜。
- **Few-shot(少量範例)**:先給它**幾組「題目 + 正確答案」當示範**,再問新題目。
  就像考試前先看幾題解答,模型會更知道你要什麼格式、什麼標準。

「few」就是「少少幾個」的意思,通常 1~5 組。本範例用 3 組。

---

## 2. 對話是由「三種角色」組成的

LLM 的對話介面(OpenAI / vLLM 都一樣)把訊息分成三種 `role`:

| role | 中文 | 它代表什麼 | 在本範例裡放什麼 |
|---|---|---|---|
| `system` | 系統 | 全域規則、任務說明(只出現一次,放最前面) | 「你是 SEM 判斷器,只能回 left/center/right……」 |
| `user` | 使用者 | 「使用者問的問題」 | 一句問題文字 + 一張圖 |
| `assistant` | 助理 | 「模型回答」 | 範例的正確答案(JSON) |

關鍵:**few-shot 的範例,是用 `user`(問)和 `assistant`(答)成對排出來的**,
而不是把範例塞進一段長文字。這樣模型會「以為」自己之前已經這樣答過好幾次,
接著看到一個新問題——這最接近它訓練時看過的格式,效果通常最好。

---

## 3. 多張範例排起來長這樣

跑一次就看得到(不需要 vLLM、不需要圖片):

```bash
python minimal_fewshot.py --dry-run
```

會印出:

```
[0] system    | 你是 SEM 影像判斷器……(任務說明,只 1 段)
[1] user      | 文字「角在哪個方向?」 + 圖片<一張影像>   ┐ 範例 1
[2] assistant | {"offset_x": "left", "offset_y": "center"} ┘(題目→答案)
[3] user      | 文字「角在哪個方向?」 + 圖片<一張影像>   ┐ 範例 2
[4] assistant | {"offset_x": "right", "offset_y": "above"} ┘
[5] user      | 文字「角在哪個方向?」 + 圖片<一張影像>   ┐ 範例 3
[6] assistant | {"offset_x": "center", "offset_y": "center"} ┘
[7] user      | 文字「角在哪個方向?」 + 圖片<一張影像>   ← 真正要問的新題目
```

看懂這 8 段怎麼來的,就懂了 few-shot 的全部精髓。程式裡負責組出這個結構的,
就是 `build_fewshot_messages()` 這一個函式——其他都是周邊。

---

## 4. 怎麼真的跑(接 vLLM)

```bash
# 1) 安裝套件
pip install openai

# 2) 設定你的 vLLM 連線
export VLLM_BASE_URL="http://你的-vllm-host:8000/v1"
export VLLM_API_KEY="EMPTY"          # vLLM 通常不在乎這個值
export VLM_MODEL="Qwen3.6-27B"       # 換成你的模型名稱

# 3) 把圖片放好
#    images/example_1.png, example_2.png, example_3.png  ← 範例(要有正確答案)
#    images/question.png                                  ← 要問的新圖
#    (路徑和答案都在 minimal_fewshot.py 最上面的 EXAMPLES / QUESTION_IMAGE 改)

# 4) 執行
python minimal_fewshot.py
```

模型會回一段像 `{"offset_x": "right", "offset_y": "center"}` 的 JSON。

> **小提醒**:程式用了 `guided_json`(在 `extra_body` 裡),它會**強迫**模型只能
> 輸出符合格式的 JSON。拿掉它模型也會答,只是偶爾會多吐一些文字害你 parse 失敗。
> 這是 vLLM 的功能,不是所有後端都有。

---

## 5. 把範例改成你自己的任務

只要動 `minimal_fewshot.py` 最上面三個地方:

1. `SYSTEM_PROMPT` —— 改成你的任務說明。
2. `EXAMPLES` —— 換成你的範例圖 + 答案(想加幾組就加幾組)。
3. `JSON_SCHEMA` —— 改成你要的輸出欄位。

其他都不用動。

---

## 6. 看懂了之後,再去看完整版

| 你想知道… | 去看 |
|---|---|
| 怎麼**衡量** few-shot 到底有沒有用(命中率、信賴區間) | `../corner_judge_categorical/` |
| 怎麼讓模型輸出**座標**而不是方向 | `../corner_locate_coords/` |
| 範例該怎麼挑、會不會作弊(資料洩漏) | 兩個資料夾的 design doc 第 4 節 |

完整版多出來的東西(評分、重複測、Wilson 信賴區間、多條件 ladder),
都是為了「**用數據證明** few-shot 值不值得做」。但 few-shot 本身的骨架,
就是這份範例裡的 `build_fewshot_messages()` 那幾行而已。

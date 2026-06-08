# 極簡 Few-shot 範例 — 座標定位版

這是 [`../fewshot_minimal_example/`](../fewshot_minimal_example/) 的**姊妹版**。
建議先看那一個(判斷方向、輸出 left/center/right),再看這一個。

兩者的 few-shot **骨架完全一樣**;差別只有任務:

| | 上一個範例 | 這個範例 |
|---|---|---|
| 問題 | 角在十字的「哪個方向」? | 角的「座標在哪」? |
| 輸出 | `{"offset_x": "left", "offset_y": "center"}` | `{"x": 487, "y": 312}` |
| 範例圖 | 一般 SEM 圖 | **圖上畫了綠色十字標出答案** |
| 題目圖 | 一般 SEM 圖 | **沒有綠色十字**(要模型自己找) |

---

## 1. 這版本最重要的新觀念:用「畫在圖上的答案」當範例

前一個範例的「答案」只存在於 assistant 的文字裡。這個範例多了一個技巧:

> **範例影像上直接用綠色十字把正確答案「畫出來」,assistant 再給出那個十字的座標。**
> 等於同時給模型兩種線索:看得到的(綠十字位置)+ 文字的(x, y 數字)。

然後關鍵的一步——**真正要問的圖把綠十字拿掉**,逼模型把「範例教的東西」
套用到一張乾淨的新圖上。

這就像教小孩:先給幾張「已經把答案圈起來」的練習題,再給一張沒圈答案的考題。
這種做法叫 **visual-demonstration few-shot(視覺示範式 few-shot)**,
在「要模型輸出位置/座標」的任務上特別常用。

> 為什麼值得這樣做?因為這正是我們想驗證的假設:模型到底是「真的學會找角」,
> 還是只會「找綠色像素」?完整實驗(`../corner_locate_coords/`)就是在量這件事。
> 這份極簡版只示範**怎麼組出這個結構**,不做量測。

---

## 2. 對話結構(跟上一個範例同款)

跑一次就看得到(不需要 vLLM、不需要圖片):

```bash
python minimal_fewshot_locate.py --dry-run
```

會印出 8 段 messages:

```
[0] system    | 你是 SEM 座標定位器……範例有綠十字、題目沒有、紅十字忽略……(只 1 段)
[1] user      | 文字「角在哪裡?輸出座標」 + 圖片<範例1,有綠十字>  ┐ 範例 1
[2] assistant | {"x": 312, "y": 458}                              ┘(綠十字的座標)
[3] user      | 文字「角在哪裡?輸出座標」 + 圖片<範例2,有綠十字>  ┐ 範例 2
[4] assistant | {"x": 690, "y": 205}                              ┘
[5] user      | 文字「角在哪裡?輸出座標」 + 圖片<範例3,有綠十字>  ┐ 範例 3
[6] assistant | {"x": 540, "y": 540}                              ┘
[7] user      | 文字「角在哪裡?輸出座標」 + 圖片<題目,沒有綠十字> ← 真正要問的
```

組出這個結構的,一樣是 `build_fewshot_messages()` 這一個函式——
它跟上一個範例裡的版本**幾乎一字不差**。few-shot 的骨架不會因為任務而改變,
變的只有「範例放什麼、答案長什麼樣」。

---

## 3. 怎麼真的跑(接 vLLM)

```bash
pip install openai

export VLLM_BASE_URL="http://你的-vllm-host:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLM_MODEL="Qwen3.6-27B"

# 準備圖片:
#   images/example_1_green_cross.png  ← 範例,圖上有綠十字
#   images/example_2_green_cross.png
#   images/example_3_green_cross.png
#   images/question_no_cross.png       ← 題目,沒有綠十字
# 路徑與座標答案都在 minimal_fewshot_locate.py 最上面的 EXAMPLES / QUESTION_IMAGE 改。

python minimal_fewshot_locate.py
```

模型會回一段像 `{"x": 503, "y": 287}` 的 JSON。

---

## 4. 三個一定要做對的細節

這三點在完整實驗的 design doc 講得很細,極簡版至少要記住:

1. **範例的綠十字中心 = 你寫的 (x, y),要對齊到像素。**
   圖上畫的位置和 answer 的數字不一致,模型看到的是矛盾示範,few-shot 就失效。
   建議用程式畫綠十字(以座標為中心),不要用滑鼠手畫。

2. **題目圖絕對不能有綠十字。**
   洩漏一個綠十字,模型可能直接「找綠色」,你量到的就不是真本事。

3. **紅十字一律忽略。** 紅色 = 機台 camera 中心 marker,跟找角無關。
   程式的 `SYSTEM_PROMPT` 已經寫明要忽略它。

---

## 5. 想再深入

| 你想知道… | 去看 |
|---|---|
| 怎麼**衡量**座標 few-shot 準不準(命中率 hit@τ、誤差 L2、信賴區間) | `../corner_locate_coords/` |
| pixel 座標 vs 0~1000 正規化座標,哪個對模型比較好 | `../corner_locate_coords/`(coord 條件) |
| few-shot 最基本的骨架(判斷方向版) | `../fewshot_minimal_example/` |

這份範例為了好懂,寫死了影像尺寸(`IMAGE_W, IMAGE_H = 1024, 1024`)並用 pixel 座標。
完整實驗支援不同尺寸與正規化座標,並會用數據告訴你該選哪個。

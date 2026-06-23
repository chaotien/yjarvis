# 極簡 Few-shot 範例 — 成對比較版(兩張圖一起看)

這是 [`../fewshot_minimal_example/`](../fewshot_minimal_example/) 的**姊妹版**。
建議先看那一個(看一張圖、判斷方向),再看這一個。

它對應的完整實驗是 [`../bright_dot_cluster_judge/`](../bright_dot_cluster_judge/)
(有評分、平衡準確率、前處理 ladder……第一次看會很雜)。這裡把它**砍到只剩骨架**,
只留一件要學的事。

三個範例的 few-shot **骨架完全一樣**;差別只有「一個 user turn 裡放幾張圖、任務是什麼」:

| | 第一個範例 | 座標版 | 這個範例 |
|---|---|---|---|
| 每個 user turn 的圖 | 1 張 | 1 張 | **2 張(def + ref)** |
| 問題 | 角在哪個方向? | 角的座標? | 兩張圖的**差異**是哪種形態? |
| 輸出 | `{"offset_x": …}` | `{"x": …, "y": …}` | `{"label": "single" 或 "cluster"}` |

---

## 1. 這版本最重要的新觀念:一個 user turn 放「兩張圖」做比較

前面兩個範例每個 user turn 只放一張圖。這個範例的關鍵新技巧是:

> **一個 user turn 的 `content` 裡放兩個 `image_url`**,讓模型在同一輪裡同時看到
> def_patch(待測)和 ref_patch(參考),然後判斷它們的**差異**。

任務不是「def 自己有多亮」,而是「**def 相對 ref 多出來的那塊亮區**長什麼形態」:

- **single** —— 一個孤立的小圓點。
- **cluster** —— 多點 / 一大坨 / 整片(只要不是單一孤立小圓點,就算 cluster)。

> 為什麼要兩張一起看?因為這是個「比較」任務:單看一張圖,模型分不出哪塊「亮」是新增的
> 缺陷、哪塊只是本來就亮的背景。把參考圖一起給,模型才有辦法「相減」再判斷。
> 完整實驗(`../bright_dot_cluster_judge/`)就是在量這種成對 few-shot 到底有沒有用。

**最關鍵的細節:兩張圖的順序(def 先、ref 後)是模型判斷「誰是誰」的唯一線索。**
所以 prompt 文字會寫死「第一張是 def、第二張是 ref」,而且範例和題目都用同一個順序。
順序一旦顛倒,模型就會反方向比較。

---

## 2. 對話結構(跟前面範例同款,只是 user turn 變兩張圖)

跑一次就看得到(不需要 vLLM、不需要圖片):

```bash
python minimal_fewshot_pair.py --dry-run
```

會印出 10 段 messages(1 段 system + 4 組範例×2 + 1 段提問):

```
[0] system    | 你是半導體缺陷判斷器……比的是 def 相對 ref 的差異……(只 1 段)
[1] user      | 文字「第一張def、第二張ref…」 + 圖片<def_patch> + 圖片<ref_patch>  ┐ 範例 1
[2] assistant | {"label": "single"}                                              ┘
[3] user      | 文字「…」 + 圖片<def_patch> + 圖片<ref_patch>                      ┐ 範例 2
[4] assistant | {"label": "cluster"}                                             ┘
[5] user      | 文字「…」 + 圖片<def_patch> + 圖片<ref_patch>                      ┐ 範例 3
[6] assistant | {"label": "single"}                                              ┘
[7] user      | 文字「…」 + 圖片<def_patch> + 圖片<ref_patch>                      ┐ 範例 4
[8] assistant | {"label": "cluster"}                                             ┘
[9] user      | 文字「…」 + 圖片<def_patch> + 圖片<ref_patch>  ← 真正要問的新一對
```

注意每個 `user` 都是「**一段文字 + 兩張圖**」。組出這個結構的,一樣是
`build_fewshot_messages()` 這一個函式——它跟前面範例**幾乎一字不差**,唯一的差別是
`make_user_turn()` 收兩個路徑、塞兩個 `image_url`。few-shot 的骨架不會因為任務而改變。

> 為什麼用 4 組範例而不是 3?因為這是**二元**任務,範例最好讓兩個類別都出現
>(這裡 2 組 single + 2 組 cluster),模型才不會以為「答案永遠是同一類」。

---

## 3. 怎麼真的跑(接 vLLM)

```bash
pip install openai

export VLLM_BASE_URL="http://你的-vllm-host:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLM_MODEL="Qwen3.6-27B"

# 準備圖片(每組都要 def 和 ref 一對):
#   images/example_1_def.png  images/example_1_ref.png   ← 範例 1(答案 single)
#   images/example_2_def.png  images/example_2_ref.png   ← 範例 2(答案 cluster)
#   …
#   images/question_def.png   images/question_ref.png    ← 要問的新一對
# 路徑與答案都在 minimal_fewshot_pair.py 最上面的 EXAMPLES / QUESTION_* 改。

python minimal_fewshot_pair.py
```

模型會回一段像 `{"label": "cluster"}` 的 JSON。

> **小提醒**:程式用了 `guided_json`(在 `extra_body` 裡)強迫模型只能輸出
> `{"label": "single"}` 或 `{"label": "cluster"}`。拿掉它模型也會答,只是偶爾會多吐文字
> 害你 parse 失敗。這是 vLLM 的功能,不是所有後端都有。

---

## 4. 幾個一定要做對的細節

1. **def 在前、ref 在後,從頭到尾固定。**
   圖的順序是模型判斷「誰是待測、誰是參考」的唯一依據。範例和題目順序不一致,few-shot 就失效。

2. **比的是「差異」,不是絕對亮度。**
   兩張都亮的共同區域不是缺陷;只看「def 相對 ref 多出來」那塊。`SYSTEM_PROMPT` 已寫死這條,
   你挑範例、標答案時也要照同一把尺。

3. **`single` 要同時「孤立」又「小」。**
   大到像一坨的單點算 cluster;兩個點不管多小都算 cluster。邊界 case 的門檻要先定好、一致套用,
   不然二元任務會悄悄爛掉。

4. **真實的 patch 很小,通常要先放大。**
   實際 patch 可能才約 64×64、目標亮點只有 ~5px,直接丟給模型,它內部 resize 可能就把點抹掉。
   這份極簡版為了好懂**沒有做放大**;真的要用,放大倍率與內插方式(nearest vs lanczos——
   後者會把鄰近點糊成一坨而誤判成 cluster)本身就是一條要量的實驗軸,見
   [`../bright_dot_cluster_judge/preprocess_experiment_zh.md`](../bright_dot_cluster_judge/preprocess_experiment_zh.md)。

---

## 5. 想再深入

| 你想知道… | 去看 |
|---|---|
| 怎麼**衡量**成對 few-shot 準不準(平衡準確率、兩個方向的漏判、信賴區間) | `../bright_dot_cluster_judge/` |
| 小 patch 該放大幾倍、用哪種內插 | `../bright_dot_cluster_judge/preprocess_experiment_zh.md` |
| few-shot 最基本的骨架(單張圖、判斷方向) | `../fewshot_minimal_example/` |
| 把答案「畫在範例圖上」的視覺示範式 few-shot | `../fewshot_minimal_example_locate/` |

這份範例為了好懂,答案只留一個 `label`。完整實驗的答案還多了 `reasoning`(先描述再下結論)、
`morphology`(single_dot / multi_dots / large_blob / broad_area,把「為什麼」攤開)和
`brighter_region_found`,並用這些欄位檢查模型有沒有自打嘴巴。但成對 few-shot 的骨架,
就是這份範例裡 `build_fewshot_messages()` 那幾行而已。

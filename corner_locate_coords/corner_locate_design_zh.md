# 座標回歸實驗解說(繁體中文)— SEM 角座標定位

> 姊妹實驗:`../corner_judge_categorical/`(類別判斷)+ 本實驗(座標輸出)。
> 共享的 role 矩陣、guided_json/response_format 機制、reasoning-first 邏輯、Wilson CI、
> 條件 ladder 紀律,在前一份中文 doc(`../corner_judge_categorical/fewshot_experiment_zh.md`)
> 都講過,**這份只寫不一樣的部分**:任務語意改變後,prompt、schema、指標、labeling
> 都跟著變。

---

## 0. 一句話定位

類別實驗在問「角有沒有對上十字」,本實驗在問「**角在哪裡**(像素座標)」。前者輸出
枚舉值(left/right/above/below),後者輸出整數 (x, y)。任務質變,所以是兩個獨立實驗,
但用同一套客戶端紀律(直接打 vLLM、條件 ladder、Wilson CI)來量。

---

## 1. 核心機制 — 視覺示範式 few-shot(這個實驗的精髓)

### 1.1 範例 vs 測試影像的不對稱

| | exemplar 影像 | eval 影像 |
|---|---|---|
| 綠色十字 | **有**,畫在真實角的位置 | **沒有**(production 也不會有) |
| 紅色十字 | 可能有(機台 camera marker) | 可能有 |
| assistant 答案 | 給出綠色十字中心的 (x, y) | (待 VLM 預測) |

換句話說:**範例用「圖上畫個綠十字 + 答案 JSON 給出綠十字座標」雙重信號**,告訴模型
「我們認定的『角』長這樣、座標應該長這樣輸出」。eval 影像把綠十字拿掉,模型要把這個
示範**generalize 到沒有 marker 的影像上**。這是「visual demonstration few-shot」的標準用法,
跟單純把答案塞進 assistant 的 few-shot 差很多 — 它讓模型看到「正確答案在影像上的位置」,
而不只是文字答案。

### 1.2 為什麼這個設計值得實驗

兩個極端可能:

- **示範會 generalize**:模型從綠十字看到「這就是角」,在沒有綠十字的影像上也找得到
  那個 L 形頂點並輸出座標。→ few-shot 是真正的 lever,值得做進 production。
- **示範不會 generalize**:模型只學到「找綠色像素」,綠十字一拿掉就完全失準。→ 這條
  路走不通,localization 該回 CV(template matching / NCC)。

實驗就是要把這條岔路量出來。

### 1.3 對應的 message 排列

`build_messages` 產出的順序(`k=3` 範例,`coord="pixel"`):

```
system:    任務規則 + 座標格式說明(包括 W×H)
user:      [USER_TURN_TEXT, exemplar_image_1(有綠十字)]
assistant: {"reasoning":"...", "corner_found":true, "x":487, "y":312}
user:      [USER_TURN_TEXT, exemplar_image_2(有綠十字)]
assistant: {"reasoning":"...", "corner_found":true, "x":315, "y":420}
user:      [USER_TURN_TEXT, exemplar_image_3(有綠十字)]
assistant: {"reasoning":"...", "corner_found":true, "x":620, "y":188}
user:      [USER_TURN_TEXT, test_image(沒有綠十字)]   ← 真正要回答的題目
```

- **system 寫死了「範例會有綠十字、測試影像不會」的契約**。這樣 user 段就可以保持一致
  (同一段 `USER_TURN_TEXT`),不會在最後一輪暗示模型「現在規則變了」。
- system 也寫死「**忽略紅色十字**」,因為紅色 = 機台 camera marker,在任何影像上都
  可能出現,跟任務無關。

---

## 2. 座標格式 — `pixel` vs `norm1000`

VLM 在輸出精確像素座標時表現很挑模型。腳本支援兩種格式作為條件比較:

### 2.1 `pixel`

- 直接輸出像素整數,如 `{"x": 487, "y": 312}`。
- **system prompt 會明示影像尺寸**(`影像尺寸為 W×H`),讓模型有絕對座標的參考。
- schema 限制 `x, y ≥ 0`。
- 適合所有影像同尺寸的情況(本實驗就是)。

### 2.2 `norm1000`(Qwen-VL 慣例)

- 不論影像尺寸,輸出 `[0, 1000]` 整數,如 `{"x": 476, "y": 305}`。
- 程式端把 1000 對應到實際影像寬高再換算回像素。
- Qwen 等 vision 模型內部 bbox 輸出本來就用這套表示;對它們可能比 pixel 直觀。
- schema 限制 `0 ≤ x, y ≤ 1000`。

### 2.3 為什麼只在 `fewshot5` 階段比較這兩種?

避免條件矩陣爆炸。座標格式是模型本身的偏好,通常不會跟 few-shot 互動;在最強的 setup
(k=5 + guided + reasoning)比較一次就夠了。如果發現兩者差距大,再回頭把 pixel 版本
換成 norm1000 重跑 zeroshot/few-shot 系列。

---

## 3. 指標 — 為什麼長這樣

下游一樣是「對位控制器」,失敗模式換了形式但精神不變:控制器拿到一個錯的座標,會把
載台推到錯的方向。所以指標分兩類。

### 3.1 安全關鍵(closed-loop 會發散)

- **`wrongQuadrant%`**:預測點落在影像中心的「**對角象限**」上(x 與 y 同時跨到 GT
  的相反邊)。在閉環中等於兩軸都驅動錯誤,**最危險**。計算時在影像中心 ±5% 範圍內留
  dead-band,避免雜訊把貼近中心的 case 誤判成 quadrant flip。
- 跟類別實驗的 `wrongDir%` 是同種失敗模式的座標版。

### 3.2 精度關鍵(controller 收斂得有多快)

- **`hit@τ%`**:`L2(pred, GT) ≤ τ × min(W, H)` 的比例。預設 τ ∈ {1%, 2%, 5%, 10%}。
  - `hit@5%` 是「實用門檻」的代表 — 一般控制器在這個誤差內可以慢慢收斂。
  - `hit@2%` / `hit@1%` 是「能不能一次就到位」的精度上限。
- **`medianL2%`**:典型誤差,**對離群值不敏感**,看模型「大多數時候」差多少。
- **`meanL2%`**:對離群值敏感;如果它比 median 高很多,代表少數預測爆炸 — 這通常就是
  `wrongQuadrant%` 的那批 case。
- **`hit_bbox%`**:若 manifest 提供 `bbox`,計算「預測點是否落在範圍內」的命中率。
  這是 task-defined 的 acceptance 條件,跟 `hit@τ%` 互補(τ 是相對誤差,bbox 是絕對範圍)。

### 3.3 為什麼不用單一「accuracy」?

座標回歸的「對」是一個區間判斷而不是 boolean。用 `hit@τ%` 而非單一 accuracy,可以一次
看到模型在多嚴的標準下表現如何;製造者可以根據 production tolerance 選對應的 τ 來看。

### 3.4 處理 malformed 預測

跟類別實驗一致:JSON 解析失敗或不符合 schema → `pred=None`。在 corner-present 子集上
記為:
- 所有 `hit@τ%` = 0(沒命中)
- `wrongQuadrant` = 1(視為控制器風險)
- `L2_*` = None(沒有點估計可比)

原因:malformed 在 production 等於「沒有座標可送控制器」,controller 只能 stall/retry,
**那就是一個風險事件**。

---

## 4. Labeling discipline — 三條決定實驗成敗的細節

跟類別實驗的「資料準備不變式」精神相同,但因為任務換成座標,**規矩變了**:

### (1) Exemplar 的綠色十字中心 = manifest 的 (corner_x, corner_y)

兩者必須**像素級對齊**,不能差一兩格。範例同時提供「圖像線索(綠色十字位置)」與
「文字答案 (x, y)」,如果這兩者不一致,模型看到的是矛盾的示範 — few-shot 直接失效,
還可能教壞模型。

實作建議:label 時用程式生成綠十字 — 在原圖上以 `(corner_x, corner_y)` 為中心畫十字,
然後一次 dump 出 image + manifest entry。**不要**用 GUI 工具手畫,容易飄。

### (2) Eval 影像「絕對不能」有綠十字

任何洩漏都會讓 eval 結果失真(模型可能在訓練時學會「找綠色」,綠色一出現就秒答)。
如果 production data 本身有時也帶綠色標記(例如儀器產出的 overlay),要先處理掉再進
eval。

驗證方法:跑一張 eval image 做 PIL 色彩直方圖,確認沒有顯著的綠色(R<100 G>200 B<100)
像素群。

### (3) 紅色十字無關,prompt 寫死「請完全忽略」

紅十字 = 機台 camera 中心 marker,production 影像有時會有,跟「找 SEM 結構的角」這件事
無關。

- 範例與 eval 都可能有紅十字。
- prompt 在 `_TASK_RULES` 寫明「紅色十字...與本任務無關,**請完全忽略**」。
- 第一次跑完之後,**做一次紅色 ablation**:把 eval 拆成「有紅十字」「沒紅十字」兩組,
  比較 `wrongQ%` 與 `hit@5%`。若兩組差很多,代表模型偷偷在用紅十字當錨點,prompt
  需要加強或要在前處理把紅色拿掉。

---

## 5. 程式碼導讀(只挑跟前一個實驗不同的部分)

### 5.1 動態 system prompt — `system_prompt()`

跟類別實驗的「兩個固定常數」不同,這個實驗的 system prompt **動態組合**:

```python
def system_prompt(coord_mode, reasoning, W, H):
    fmt = _FORMAT_PIXEL.format(W=W, H=H, ...) if coord_mode == "pixel" else _FORMAT_NORM
    out = _OUTPUT_REASONING if reasoning else _OUTPUT_NO_REASONING
    return base_rules + fmt + out
```

原因:`pixel` 模式需要把實際影像尺寸寫進 prompt,讓模型有絕對座標的 anchor;`norm1000`
模式則完全與尺寸無關。所以 system 不能寫死。

**注意 prefix cache 影響**:由於同 condition 內所有 call 的 system 都相同(假設影像同
尺寸),vLLM 的 prefix cache 仍然能命中。如果影像尺寸混合,需要先把 manifest 按尺寸分
桶,或乾脆改用 `norm1000`。

### 5.2 `_exemplar_answer()` — 把 GT 座標轉成 condition 對應的格式

```python
if coord_mode == "pixel":
    x_val, y_val = int(ex["corner_x"]), int(ex["corner_y"])
else:
    x_val = round(1000 * ex["corner_x"] / (W - 1))
    y_val = round(1000 * ex["corner_y"] / (H - 1))
```

範例的 manifest 一定存「像素座標」(人比較好標、也容易視覺對照),但模型看到的答案
要跟條件對應 — `norm1000` 條件下,範例答案會被轉成 0..1000 的整數。

### 5.3 `_quadrant()` 的 dead-band

```python
band_x = 0.05 * W
sx = 0 if abs(x - cx) <= band_x else (1 if x > cx else -1)
```

GT 或 pred 落在影像中心 ±5% 範圍內時,quadrant 值取 0,不參與「對角象限」判斷。原因:
如果角剛好在影像中央附近,任何小擾動都可能在 quadrant 之間跳動,把可避免的雜訊計入
`wrongQuadrant%` 會誤導讀者。

### 5.4 `--tolerances` 的 CLI 設計

預設 `0.01 0.02 0.05 0.10`(分別對 min(W,H) 的 1%/2%/5%/10%)。production 把 tolerance
壓緊時:

```bash
python run_corner_locate_experiment.py --data-dir ./d --tolerances 0.005 0.01 0.02
```

τ 是相對值,影像換尺寸不需要改 τ — 這是用 min(W,H) 正規化的目的。

---

## 6. 怎麼判讀 summary.md — 一張表的閱讀順序

```
condition                      fmt%  hit@1%  hit@2%  hit@5%  hit@10%  hit5±sd  bbox%  L2µ%  L2~%  wrongQ%  lat_ms
```

讀的順序:

1. **`fmt%`**:guided 條件下應該 ~100%。若不到,先解決(可能 schema 跟模型不相容)。
2. **`wrongQ%`**:絕對要 < 5%(經驗值)。任何一條 wrongQ% 高的條件直接淘汰。
3. **`hit@5%`**:headline 命中率。`zeroshot_*` vs `fewshot{3,5}_*` 的相對差距決定 H3。
4. **`hit@2%` / `hit@1%`**:精度上限。若 `hit@5%` 很好但 `hit@1%` 接近零,模型其實只
   能粗定位,production 的 tolerance 要相應放寬,或考慮升解析度。
5. **`L2~%`**(median):「大多數時候誤差多少」。
6. **`L2µ%`**(mean):若比 median 高很多,代表有離群暴衝 case,跟 `wrongQ%` 對照看。
7. **`hit5±sd`**:重複間穩定度。sd 大 → 單次判斷有風險,production 要 majority-vote。
8. **`bbox%`**:task-defined acceptance,跟 `hit@τ%` 互相驗證。
9. **`lat_ms`**:成本。如果 fewshot5 贏但 latency 翻倍,考慮 vLLM prefix cache。

---

## 7. 跟前一個實驗的對照與整合

| 角色 | 類別實驗 | 本實驗 | 整合到 production 後的責任 |
|---|---|---|---|
| `sem_corner_judge`(類別) | ✅ 已驗證(暫定) | — | 「目前位置算不算對齊」的最終驗收 |
| `sem_corner_locate`(座標) | — | ✅ 待驗證 | 「下一步該往哪移」的座標供給 |

兩個是**互補**,不是替代。閉環流程大致是:

```
LOOP:
  座標 = sem_corner_locate(影像)        # ← 本實驗驗證的能力
  控制器移動載台(根據座標 - 影像中心,反向)
  類別 = sem_corner_judge(新影像)       # ← 類別實驗驗證的能力
  if 類別.aligned: break
```

所以兩個實驗都要過才能整套上線。若**本實驗(B)分支**(座標完全不可信),則 locate
回退到 CV(template matching / NCC),類別 judge 仍可用 VLM。若**兩個都(D)分支**,
整套對位閉環就回 CV 為主,VLM 暫時擱置等下一代模型。

---

## 8. 一頁總結(印出來貼牆上)

1. **任務**:輸出 SEM 影像中「L 形角頂點」的像素 (x, y)。
2. **示範**:exemplar 上有綠色十字(GT 位置),eval 影像沒有。模型要把這個示範 generalize。
3. **指標**:`wrongQ%`(安全)→ `hit@5%`(實用)→ `hit@2%`/`hit@1%`(精度)→ `L2~%`
   (典型誤差),`align%` 概念不適用本實驗。
4. **格式**:`pixel`(影像同尺寸時)或 `norm1000`(Qwen 慣例)在 k=5 階段比較一次。
5. **labeling**:exemplar 綠十字像素級對齊 manifest 座標;eval **零綠色**;紅十字一律
   prompt 忽略。
6. **重複**:`temperature=0` 仍要 `--repeats 3`,看 `hit5±sd`。
7. **決策**:(A) few-shot 顯著贏 → 上 VLM coords;(B) 全部不準 → 回 CV;(C) guided 反而
   傷 → 鬆 schema;(D) 格式偏好顯著 → 用贏的那個格式。

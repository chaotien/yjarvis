# 前處理實驗(繁中)— 放大超小 patch,讓 VLM 看得到 ~5px 的目標

> 對照英文版 `preprocess_experiment.md`,搭配 `fewshot_experiment_zh.md`。
> 那組實驗變的是 **prompt**(k / guided / reasoning);這組把 prompt 固定(k=6、guided、
> reasoning),變的是 **影像前處理** —— 因為 patch 很小(約 64×64),`single` 目標可能才 ~5px。
> 原圖直接送,VLM 內部 resize 可能就把這個 5px 點抹掉,所以「client 端怎麼放大」變成最關鍵的
> 槓桿。用同一支 `run_fewshot_experiment.py`,以 `--ladder preprocess` 選取。

---

## 1. 為什麼前處理在這裡是「主軸」而不是細節

64×64 patch ≈ 4096 像素。Vision LLM(如 Qwen-VL 系列)把影像切成 patch token,有**最小/最大
像素預算**,會自動 resize 輸入來符合。64×64 在下限附近甚至以下,伺服器可能用它自己(未知)的
內插放大、或只用極少視覺 token 表示 —— 不論哪種,5px 的點都可能糊進背景或直接消失。把影像在
**client 端**先放大到模型 token 預算內的舒適尺寸(例如 512×512),你就**完全掌控模型看到什麼**,
也給了目標足夠的 token 被「看見」。

但放大不是沒有取捨,而且內插方式會跟**這個任務本身**互相影響:

- **nearest(最近鄰)**:保留硬邊與鄰近點的**離散性** —— 三個分開的點放大後還是三個分開的方塊。
  有利於把 `multi_dots` 跟 `single_dot`、把 `single` 跟 `large_blob` 區分開。
- **bilinear / bicubic / lanczos**:會平滑。平滑可能把**兩個鄰近點糊成一坨**(真 cluster 被讀成
  一顆大點),也可能把真正的 `single` 點邊緣抹糊成一坨而被讀成 `cluster`。不論哪種,平滑都可能
  **改變模型感知到的形態** —— 這正是本實驗要測的核心假設。

所以問題不是「要不要放大」,而是「**哪個放大倍率 + 內插,能在可接受的 token 成本下,最好地保留
single vs cluster 的形態?**」

---

## 2. 假設

- **P1(原圖是地板)**:`prep_raw_x1`(64×64 原圖)`balAcc%` 最差 —— 目標太小,模型解析不穩。
- **P2(放大有用,但邊際遞減)**:`x4 → x8` 提升 `balAcc%`;某個倍率之後只增 token/latency
  而不增訊號(`x12`)。
- **P3(內插影響形態)**:`nearest` ≥ `lanczos`(看 `balAcc%`),而且差異會出現在漏判方向上:
  lanczos 應該會推高 `missClu%`(cluster 被糊成一顆點)或 `missSgl%`(single 被抹成一坨)。
- **P4(contrast 救微弱差異)**:若 def/ref 的 GLV 差很小,**聯合**百分位拉伸
  (`prep_x8_nearest_contrast`)能提高亮區的可見度 —— 但**必須聯合做**(見 §4),
  各自獨立拉伸會把差異本身**抹掉**。
- **要接受的虛無假設**:若連最好的前處理都讓 `balAcc%` 接近亂猜,代表目標在這個原始解析度下
  已超出 VLM 能力 → 答案是對差異影像做 **CV**(def−ref → 閾值 → 連通元件 → 用 blob 數/面積分類)。

---

## 3. preprocess ladder

定義在 `PREP_CONDITIONS`(全部固定在最佳 prompt:k=6、guided、reasoning):

| condition | scale | 尺寸(64→) | interp | contrast | 隔離出什麼 |
|---|---|---|---|---|---|
| `prep_raw_x1` | 1 | 64 | nearest | 否 | 地板基準(P1) |
| `prep_x4_nearest` | 4 | 256 | nearest | 否 | 中度放大(P2) |
| `prep_x8_nearest` | 8 | 512 | nearest | 否 | 較大放大(P2) |
| `prep_x8_lanczos` | 8 | 512 | lanczos | 否 | 內插效果(P3)—— 對照 `prep_x8_nearest` |
| `prep_x8_nearest_contrast` | 8 | 512 | nearest | 是 | 聯合 contrast 效果(P4) |
| `prep_x12_nearest` | 12 | 768 | nearest | 否 | 飽和/成本上限(P2) |

**重點看(一次一個變數):**
- `prep_raw_x1 → x4 → x8 → x12`:放大曲線,找轉折點(knee)。
- `prep_x8_nearest` vs `prep_x8_lanczos`:固定 scale 比內插(P3)。
- `prep_x8_nearest` vs `prep_x8_nearest_contrast`:固定 scale+interp 比 contrast(P4)。

`lat_ms` 在這組是**結果的一部分**,不是註腳:64→512 是 64 倍像素,倍率越大、視覺 token 越多、
越慢。勝出者是**能保住 `balAcc%`/`missSgl%` 增益的「最小」前處理**。

---

## 4. 前處理不變式(弄錯就量錯)

1. **範例與 eval 用同一套前處理。** 腳本會把該 condition 的 `prep` 同時套在 few-shot 範例對與
   eval 對上。若範例是原圖、eval 卻放大了,示範就跟題目對不起來。(已自動處理;不要自己只在硬碟上
   預先 resize 部分圖片而繞過它。)
2. **contrast 一定要聯合,不能逐張。** `--contrast` / `prep…_contrast` 是用**兩張合起來**算
   一條百分位 LUT,再同時套到兩張(`_joint_contrast`)。這很關鍵:逐張獨立拉伸會把 def、ref
   各自正規化,**抹掉**任務賴以判斷的亮度差。你若自己加 contrast,也要保持聯合。
3. **是放大,不是重新裁切。** 放大不可改變視野或重新置中 —— def、ref 放大後仍須 co-register
   (純整數倍 resize 會保持;裁切/補邊則不會)。
4. **只用 PNG。** 輸出 PNG(無損)。JPEG 壓縮在 5px 亮點周圍產生的假影本身就像結構,會污染形態訊號。

---

## 5. 怎麼跑(兩階段)

```bash
pip install openai pillow
export VLLM_BASE_URL="http://your-vllm-host:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLM_MODEL="Qwen3.6-27B"

# 階段 1 —— 決定最佳前處理(prompt 固定 k=6):
python run_fewshot_experiment.py --data-dir ./patch_eval --ladder preprocess --repeats 3

# 階段 2 —— 在勝出的前處理下跑 PROMPT ladder(例如 x8 nearest):
python run_fewshot_experiment.py --data-dir ./patch_eval --ladder prompt \
    --upscale 8 --interp nearest --repeats 3

# 臨時:用全域 override 把任意前處理強加到任意 ladder
python run_fewshot_experiment.py --data-dir ./patch_eval --ladder prompt \
    --upscale 8 --interp nearest --contrast
```

`--upscale` / `--interp` / `--contrast` 會**覆寫**所有選到的 condition 的 `prep` —— 這就是
階段 2 把整條 prompt ladder 釘在同一套前處理的方法。不加這些旗標時,每個 `prep_*` condition
用自己內建的設定。

輸出在 `summary.md` 與 `raw_results.csv` 都帶一個 `prep` 欄(例如 `s8-nearest-c0`),
讓前處理永遠跟指標並列可見。

> 先用合成資料驗證管線 —— `make_synthetic_demo.py` 會畫 64×64 的對,含 ~5px 單點與
> 多點/blob/整片 的 cluster,跟真實任務同個 regime:
> `python make_synthetic_demo.py --out-dir ./synthetic_demo && \
> python run_fewshot_experiment.py --data-dir ./synthetic_demo --ladder preprocess --repeats 1`

---

## 6. 結果怎麼讀

1. **找放大轉折點**:看 `balAcc%` 在 `raw → x4 → x8 → x12` 的曲線。過了轉折點,`lat_ms` 白付。
2. **比內插**(在轉折倍率上,`nearest` vs `lanczos`)。若 lanczos 推高 `missClu%` 或 `missSgl%`,
   代表平滑在破壞形態 → 用 nearest。
3. **比 contrast**(`…_contrast` vs 沒有)。只有在原始差異很淡時才有用;沒幫助就拿掉,少一個變數。
4. **優先指標是 `missSgl%`**(此用途 single 漏判較嚴重):在 `balAcc%` 相當的前處理裡,選
   `missSgl%` 最低的 —— 但要排除那種靠「永遠答 single」贏 `missSgl%` 的(盯著 `missClu%`)。
5. **定案**:取能保住 `balAcc%` 與 `missSgl%` 增益的「最小/最便宜」前處理,再以該前處理跑階段 2
   的 prompt ladder 決定 `k`。兩者都寫進 ADR:*「prep = x8 nearest;k = 3;missSgl X%→Y%(95% CI …)」*。

若曲線怎麼都過不了亂猜線,就停在這裡 —— 這是 **CV 分支**(差異影像 + 連通元件分析),
在這個原始解析度下不是 VLM 的題目。

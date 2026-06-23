# Few-shot 實驗解說(繁體中文)— def/ref patch 的 single vs cluster 判斷

> 對照英文版 `fewshot_experiment.md` 與執行腳本 `run_fewshot_experiment.py`。
> 這是 `../corner_judge_categorical/` 的姊妹實驗。**底層機制(role 矩陣、guided_json vs
> response_format、reasoning-first、Wilson CI、temperature=0 仍要 repeats)在那邊的
> `fewshot_experiment_zh.md` 已經講得很完整,這裡不重複** —— 本文只講「這個任務和它們不一樣
> 的地方」:一次輸入兩張 patch、二元分類、以及對應的指標設計。第一次接觸 few-shot 的人,
> 建議先讀 `../fewshot_minimal_example/`,再回來看這份。

---

## 0. 一段話定錨

我們有一組新資料:每個 example 是**兩張小圖** —— `def_patch`(待測)和 `ref_patch`(參考)。
兩張的灰階(GLV)大部分重疊,通常只有局部差異。要判斷的是:**def_patch 相對 ref_patch
多出來的那塊亮區,是「單一孤立的小圓點(single)」,還是「多點/一大坨/整片(cluster)」**。
這次同樣**先用最便宜的方式(直接打 vLLM)量 few-shot 有沒有用**,再決定要不要做進產品。

> 一句話:任務從「看一張圖」變成「比兩張圖的差異」,但「先量、後建」的精神不變。

---

## 1. 這個任務跟姊妹實驗差在哪

| | 角判斷(姊妹資料夾) | 本實驗 |
|---|---|---|
| 每題輸入 | 一張 SEM 影像 | **兩張 patch**:def_patch + ref_patch |
| 輸出 | offset 列舉 + `aligned` | `label ∈ {single, cluster}`(+ 描述用的 `morphology`) |
| 模型要做的事 | 找角相對 marker | **比對** def 與 ref,分類「差異」的形態 |
| 核心指標 | `wrongDir%` / `falseAln%` | `balAcc%` + 兩個方向的漏判(`missClu%`、`missSgl%`) |
| few-shot 範例 | 範例影像 + 答案 JSON | **範例「一對」(def+ref)+ 答案 JSON** |

新的關鍵手段是 **「一個 user turn 帶兩張圖」**:每一輪都放兩張圖(def 在前、ref 在後),
prompt 文字寫死誰是誰。few-shot 範例示範的是**「比較」**,不只是「長相」。

---

## 2. 任務定義 = 規格(labeling 必須對齊)

判斷的是 **def 減 ref 的「差異」**,不是 def 自己的絕對亮度:

- **single**:def 相對 ref 多出的亮區是「**一個相對獨立、邊界清楚的小圓點**」—— 面積小、
  孤立、周圍沒有其他同時變亮的點。
- **cluster**:多出的亮區是「**多個鄰近同時變亮的點**」、或「**一大坨(blob)**」、或
  「**整片區域一起變亮**」。準則一句話:**只要不是單一孤立小圓點,就是 cluster。**

`morphology`(描述欄位,用來建立推理路徑、並回推 `label`):

| morphology | 意思 | → label |
|---|---|---|
| `single_dot` | 單一孤立小圓點 | `single` |
| `multi_dots` | 多個鄰近同時變亮的點 | `cluster` |
| `large_blob` | 一大坨、邊界糊成一團 | `cluster` |
| `broad_area` | 整片一起變亮 | `cluster` |
| `unknown` | 看不出來/沒有明顯新增亮區 | (仍須在 single/cluster 二擇一給最接近者) |

`brighter_region_found`(bool):def 是否存在「明顯比 ref 亮」的局部區域;兩張幾乎一致則 false。

### 2.1 Labeling 不變式(prompt 寫死,違反就會量錯)

1. **判斷差異,不是絕對亮度。** 在 def 與 ref **都亮**的區域不是缺陷訊號,要忽略。只標
   「def 相對 ref 多出來」那塊的形態。`_DOMAIN_KNOWLEDGE` 已把這條寫死,你的標籤也要一致。
2. **def 在前、ref 在後。** 兩張圖以這個固定順序進每個 turn,prompt 文字也這麼說。資料準備
   若把順序顛倒,模型就會反方向比較。範例與 eval 都要一致。
3. **這一對 patch 同位置、同尺寸、已對齊(本資料已滿足)。** 「想像把 ref 從 def 減掉」的前提
   就是 co-register;新資料也要維持。沒對齊的話,模型再強也算不出有意義的差異。
4. **`single` 是「孤立」且「小」。** 大到看起來像一坨的單點算 `large_blob`(cluster);
   兩個點不管多小都算 `multi_dots`(cluster)。**大小/間距的門檻要先定好寫進標註指南、
   一致套用** —— 邊界 case 是二元任務悄悄爛掉的地方。

> **patch 很小要放大。** patch 只有約 64×64,`single` 目標可能才 ~5px —— 原圖直接丟,VLM 內部
> resize 可能就把它抹掉。**client 端放大是另一條獨立的實驗軸**,而且內插方式會跟 single/cluster
> 判斷互相影響(smoothing 會把鄰近點糊成一坨 → 誤判)。那組實驗看
> **`preprocess_experiment_zh.md`**(與 `--ladder preprocess`)。

---

## 3. 條件 ladder

定義在 `CONDITIONS`,每一階只多加一件事,相鄰兩階的差就隔離出那個決定。你預計 ~6 組範例,
所以 few-shot 階用 **k=3** 和 **k=6**。

| condition | k | guided | reasoning | 隔離出什麼 |
|---|---|---|---|---|
| `zeroshot_freetext` | 0 | off | yes | baseline;從散文 parse JSON |
| `zeroshot_guided` | 0 | on | no | guided_json 可靠性 + 裸 schema 的感知代價 |
| `zeroshot_guided_reasoning` | 0 | on | yes | 約束下 reasoning-first 欄位的效果 |
| `fewshot3_guided_reasoning` | 3 | on | yes | 3 組成對範例的效果 |
| `fewshot6_guided_reasoning` | 6 | on | yes | 6 組成對範例(飽和點) |
| `fewshot6_guided_noreasoning` | 6 | on | no | reasoning × few-shot 的交互作用 |

**重點看**:`zeroshot_guided_reasoning → fewshot3 → fewshot6` 是 few-shot 的主問題與飽和點;
`fewshot6_guided_reasoning` vs `..._noreasoning` 問「有了 6 組成對範例後,reasoning 欄位還
值不值得那些 token」。

---

## 4. 指標 — 為「不平衡的二元決策」設計

只看 `label` 準確率會說謊:若 8 成是 cluster,每次都答 cluster 就有 80%。所以頭號指標是
**balanced accuracy**,並把兩個方向的錯誤都攤開:

- **`balAcc%`(頭號):** single recall 與 cluster recall 的平均;對類別不平衡穩健。
- **`missSgl%`(優先 —— 此用途 single 漏判較嚴重):** 真 single 被判成 cluster 的比例。先壓這個。
- **`missClu%`(次要):** 真 cluster 被判成 single 的比例。
- `sRec% / cRec%`:兩類各自的 recall(`balAcc%` 平均的就是這兩個)。
- `acc%`:原始準確率,附 **Wilson 95% CI** —— 當 sanity check,不是目標。
- `consist%`:`label` 與 `morphology` 是否自洽(`single_dot→single`,其餘→`cluster`)。
  偏低代表模型自打嘴巴 —— 是 prompt/schema 的壞味道。
- `fmt%`:schema 合規(guided 下應 ~100%)。`lat_ms`:成本(兩張圖 × k)。

> **此資料的優先指標是 `missSgl%`。** single 漏判(真 single 被判成 cluster)較嚴重,所以腳本
> 把 `missSgl%` 印在 `missClu%` 前面,要先優化它 —— 但小心別用「永遠答 single」的退化解把
> `missClu%` 炸高。兩個都印,`balAcc%` 守住平衡。

---

## 5. 成對影像的 few-shot 怎麼組(本實驗唯一真正的新東西)

`build_messages` + `_pair_content` 在 k 個範例、reasoning 開的情況下產生:

```
system                                   # 任務 + 領域知識 + 類別規則(固定)
user      [文字, def圖1, ref圖1]           # 範例 1:一對 patch
assistant {"reasoning":…, "morphology":"single_dot", "label":"single"}
user      [文字, def圖2, ref圖2]           # 範例 2
assistant {…}
…                                        # 最多 k 組
user      [文字, def圖q, ref圖q]           # 真正要分類的那一對
```

非顯而易見的重點:

1. **一個 user turn 裡放兩個 `image_url`,def 先 ref 後**,順序由 `USER_TURN_TEXT` 講明。
   圖的順序是模型判斷「誰是誰」的**唯一**訊號 —— 固定住。
2. **每一輪都用同一句 `USER_TURN_TEXT`**(範例與正式題),讓模型把「比一對 patch」當成穩定
   重複的 pattern。
3. **`SYSTEM_REASONING` / `SYSTEM_NO_REASONING` 是常數**(不是每次動態組),長前綴(system +
   範例輪)才會 byte 穩定 → 命中 vLLM prefix/KV 快取。6 組成對範例 = 12 張圖,**prefix cache
   是每次呼叫最大的省成本槓桿**。
4. **no-reasoning 條件會把範例答案的 `reasoning` 拿掉**(`ans.pop`),示範的答案形狀才會跟
   schema 約束一致。

> 想更深理解「為什麼 few-shot 要用 user/assistant 交替、為什麼不塞進 system 或單一 user
> turn」,看姊妹資料夾 `../corner_judge_categorical/fewshot_experiment_zh.md` §1 —— 那段
> 原理在這裡照樣成立,只是把單張圖換成一對圖。

---

## 6. 資料夾與執行

```
<data-dir>/
    eval_manifest.json        # [{ "def_patch": "...", "ref_patch": "...", "answer": {...} }, ...]  (HELD OUT)
    exemplar_manifest.json    # 同格式,與 eval 不重疊(那 ~6 組範例)
    <patch PNG 圖>
```

每對的 `answer`:

```json
{ "reasoning": "", "brighter_region_found": true,
  "morphology": "single_dot|multi_dots|large_blob|broad_area|unknown",
  "label": "single|cluster" }
```

執行:

```bash
pip install openai pillow
export VLLM_BASE_URL="http://your-vllm-host:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLM_MODEL="Qwen3.6-27B"

# (選用)還沒有真標籤前,先生合成資料驗證整條 pipeline(64×64、~5px 單點):
python make_synthetic_demo.py --out-dir ./synthetic_demo
python run_fewshot_experiment.py --data-dir ./synthetic_demo --repeats 1

# 階段 1:先決定放大/內插(prompt 固定 k=6,詳見 preprocess_experiment_zh.md):
python run_fewshot_experiment.py --data-dir ./patch_eval --ladder preprocess --repeats 3

# 階段 2:在勝出的前處理下跑 prompt ladder 決定 k:
python run_fewshot_experiment.py --data-dir ./patch_eval --ladder prompt \
    --upscale 8 --interp nearest --repeats 3

# 只跑 few-shot 主問題(同樣可加 --upscale/--interp):
python run_fewshot_experiment.py --data-dir ./patch_eval --ladder prompt --upscale 8 --interp nearest \
    --conditions zeroshot_guided_reasoning fewshot3_guided_reasoning fewshot6_guided_reasoning
```

輸出在 `--out-dir`(預設 `./experiment_out`):`raw_results.csv`(每次呼叫的稽核:gt/pred
label、各 flag、latency、截斷的原始回應)與 `summary.md`(每條件一列的表)。

---

## 7. 結果解讀 — 決策樹

- **(A) few-shot 明顯贏**:`fewshot*` 的 `balAcc%` 上升,或你最在意的漏判方向下降,且超過
  CI 重疊。→ 值得做;上線用**最小的 k**(比 3 vs 6,若相當就用 3)。
- **(B) guided+reasoning 已經夠強,few-shot 加不了多少**:CI 重疊。→ 跳過範例機制,留
  guided_json 的可靠性就好。這是好結果,接受它。
- **(C) guided_json 反而傷準度**:`fmt%`→100% 但 `balAcc%` 掉。→ schema 太緊;確保 reasoning
  先生成、放鬆 schema,重跑。
- **(D) 每條件都卡在感知**:`balAcc%` 全部接近 50%(亂猜)。→ few-shot 救不了。改用 **CV**:
  差異影像(def−ref)→ 閾值化 → 連通元件 → 用 blob 數量/面積/緊緻度分類。等下一代模型再回來。
- **(E) temp=0 變異大**:`balAcc%` 的 `±sd` 不小。→ vLLM 非決定性;production 考慮多數決或
  穩定度 gating。

最小可行版本就是 **(A)/(B)/(D)**:few-shot 有沒有贏 zero-shot,而且哪個好到能用?其餘都是
在這個問題上加嚴謹度而已。

---

## 8. 跟 yJarvis 後端的關係(同姊妹實驗)

yJarvis 維持無狀態 relay:把 `messages` + `guided_json` 原封轉給 vLLM、原封回傳。所有任務邏輯
—— prompt、schema、成對影像的組裝、範例挑選、`k` —— 全留在 client 端、跟 git 走在一起。
這正是為什麼一個 `/api/call-chat/` 端點能同時服務 `sem_corner_judge`、這個 patch judge、
以及未來每個 judge 而不用改後端。本實驗的角色:**在動後端之前,用數據決定這個任務的 few-shot
機制值不值得做** —— 先量、後建。

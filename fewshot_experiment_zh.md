# Few-shot 實驗解說(繁體中文)— SEM 角對位判斷

> 對照英文版設計文件 `fewshot_experiment.md` 與執行腳本 `run_fewshot_experiment.py`。
> 本文重點不在「重複設計」,而在**講清楚實驗背後的底層機制**:Chat API 的 role 矩陣、
> few-shot 為什麼要用 user/assistant 交替、guided_json 與 response_format 在 vLLM 中
> 怎麼運作、reasoning-first 為何放第一個欄位、以及指標為什麼要那樣定義。讀過這份再去
> 看英文版的 H1–H4 假設與決策樹會更穩。

---

## 0. 為什麼要先做這個實驗?(一段話定錨)

我們想把 SEM 對位的「角是否對上中央十字線」交給 VLM 來判斷,讓對位流程從現在的 **L2
(無人值守 CV 偵測)** 升到 **L3(視覺閉環)**。但 few-shot、guided_json、reasoning
欄位每一項都會**增加 token 成本與後端複雜度**,在我們動工去後端 yJarvis 開新的
`/api/call-chat/` 端點之前,**先用最便宜的方式(直接打 vLLM)把「這條路到底有沒有用」
量出來**。實驗結果才會決定要不要建 `B` 端點、要不要做 few-shot 機制。

> 一句話:**先量、後建**。實驗是建後端的前置條件,不是事後驗證。

---

## 1. 底層機制 ① — Chat API 的 role 矩陣

OpenAI/vLLM 的 chat completions 介面把對話切成有 `role` 的 message 序列。每個 role
的「語意」與「在訓練時被怎麼用」決定了它的權重。理解 role 矩陣,你才會懂為什麼
few-shot 要這樣排,以及為什麼亂塞會出包。

### 1.1 三種 role 的角色定位

| role | 在訓練時學到的語意 | 我們用它做什麼 |
|---|---|---|
| `system` | 「全域指令、人格、規則」— 模型對這段的服從性最強,通常會在整個對話保持效力 | 寫死任務描述、輸出規格、語意定義(`_OFFSET_RULES`) |
| `user` | 「使用者要問的具體問題」— 每一輪要回答的對象 | 提示文字 + 要看的影像 |
| `assistant` | 「模型先前的回答」— 用來建立對話脈絡,讓模型看到「之前自己/同類模型是怎麼答的」 | 在 few-shot 中放入**範例答案**(`exemplar_manifest.json` 裡的 `answer`) |

> 關鍵直覺:few-shot 不是把「題目+答案」塞進 user 那一段裡,而是**用 user 問、再用
> assistant 答**,讓模型以為這是它自己已經在這個情境下回答了好幾次,接著被丟了一個
> 新的 user 問題。這比塞 prompt 文字更接近模型在訓練資料中遇到的格式。

### 1.2 本實驗的訊息排列(以 k=3 為例)

`build_messages` 在腳本中產生的順序如下(對應 `cond["reasoning"]=True`):

```
[
  {role: system,    content: SYSTEM_REASONING},          # ← 規則,固定
  {role: user,      content: [文字提示, 範例影像1]},        # ← 範例 1 的「題目」
  {role: assistant, content: <範例1的 JSON 答案>},         # ← 範例 1 的「答案」
  {role: user,      content: [文字提示, 範例影像2]},        # ← 範例 2
  {role: assistant, content: <範例2的 JSON 答案>},
  {role: user,      content: [文字提示, 範例影像3]},        # ← 範例 3
  {role: assistant, content: <範例3的 JSON 答案>},
  {role: user,      content: [文字提示, 待判斷影像]},        # ← 真正要回答的題目
]
```

幾個非顯而易見的設計重點:

1. **每個 user turn 都重複同一段 `USER_TURN_TEXT`**:這讓模型把「題目格式」當成穩定
   pattern,可預期下一個 user turn 也是同樣結構的判斷請求。如果範例 user turn 跟
   live user turn 文字不一致,等於告訴模型「這次規則不同」。
2. **assistant 內容是 `json.dumps(...)` 後的字串**,刻意不是 markdown code block。模型
   學到的是「直接吐 JSON」,沒有額外包裝。
3. **`SYSTEM_REASONING` 與 `SYSTEM_NO_REASONING` 是兩個獨立常數**,而不是動態組裝。
   原因:`system` 在 vLLM prefix cache 上是常數前綴,寫死才能命中快取(見 §4.3)。
4. **`no_reasoning` 條件下,範例答案會把 `reasoning` 欄位移掉**(`ans.pop("reasoning", None)`)。
   若不移除,模型會學到「assistant 該吐含 reasoning 的物件」,但 schema 又不允許,
   就會打架 → schema 拒絕或胡言亂語。

### 1.3 為什麼不是把範例放在 `system` 或塞進單一 user turn?

兩個常見的反例:

- **塞進 `system`**:把「範例 1 圖+答案;範例 2 圖+答案」全寫進 system prompt。問題:
  vision model 通常**禁止/不支援 system role 帶圖**;就算支援,模型也會把它當「規則」
  而不是「範例」,語意錯位,效果通常比較差。
- **塞進單一 `user` turn**:把範例與最後一題的圖全擠進同一個 user message。問題:
  模型沒辦法在「回想之前自己怎麼答」這個位置看到 assistant 的範例 JSON,只能看到一坨
  影像和文字,**喪失 role 對齊的訓練先驗**;通常表現會明顯變差,而且很容易把第一張圖
  跟最後一張答案搞混。

> 結論:**role 排列就是 few-shot 的「語法」**,排錯就等於用錯 prompt,跟 schema 鬆緊
> 比起來,這是更前置、更根本的決定。

---

## 2. 底層機制 ② — `guided_json` 與 `response_format` 的差別

兩種都是**結構化輸出(structured output)**,但路徑不同。腳本透過 `--mechanism` 切換,
背後的差別:

### 2.1 `guided_json`(vLLM 預設,腳本中的 mechanism A)

```python
return {"extra_body": {"guided_json": schema}}
```

- 走 vLLM 專屬的 `extra_body` 通道,**vLLM 在解碼時用 grammar/regex 約束**每個 token,
  保證輸出 100% 符合 JSON schema。
- 廣相容,不限模型版本;Qwen3.6 / Llama / Mistral 在 vLLM 上都吃。
- 缺點:OpenAI SDK 不認得這個欄位,得包在 `extra_body` 裡。換成別家後端就要重寫。

### 2.2 `response_format`(OpenAI 原生風格,腳本中的 mechanism B)

```python
return {"response_format": {"type": "json_schema",
                            "json_schema": {"name": "result", "schema": schema, "strict": True}}}
```

- 走 OpenAI 標準介面,**vLLM 新版本已實作**。
- 換成 OpenAI/Anthropic/Azure 不用改 client 程式碼。
- 缺點:舊版本 vLLM 不支援;`strict: true` 對 schema 的限制比 `guided_json` 更嚴
  (例如 `additionalProperties: False` 是強制要求)。

### 2.3 為什麼這個切換很重要

實驗最後決定要送到 yJarvis 後端時,**選哪個機制決定後端 contract**。如果現有 vLLM
版本兩個都吃,優先選 `response_format`(未來換 provider 成本低);如果只吃
`guided_json`,就老實用,並把這個事實寫進 ADR。**腳本同時支援兩種,讓你用一行
參數就驗證**:

```bash
python run_fewshot_experiment.py --data-dir ./sem_eval --mechanism response_format
```

如果跑得通且結果不差於 `guided_json`,就有信心走 mechanism B。

---

## 3. 底層機制 ③ — 為什麼 `reasoning` 欄位放第一個?

`SCHEMA_REASONING` 的 properties 順序:

```python
{"reasoning": {"type": "string"}, **_PROPS_CORE}   # reasoning 在最前面
```

而 system prompt 強調:

> "reasoning 欄位放最前面(先描述看到的角結構、十字線位置、角相對中心的方位,
> 再給結論)"

### 3.1 因果順序的重要性

LLM 是**自回歸**(autoregressive)模型 — 後面的 token 是基於前面已生成的 token 條件
產生。Schema 的 properties 順序 + grammar 約束,等於**強制模型先把推理「寫出來」、
再決定最後的 `offset_x` / `offset_y` / `aligned`**。這帶來兩件事:

1. **思考被外顯化**:模型在決定 `offset_x` 之前,已經被迫先寫出對角位置的描述。視覺
   感知的不確定性會在 reasoning 段裡顯露(「角結構不清晰」、「十字線部分被遮擋」),
   讓 downstream 判斷更穩定。
2. **避免「先決定再合理化」**:沒有 reasoning 欄位時,模型直接吐 `offset_x: "left"`,
   就只是一個 softmax 抽樣的結果;有 reasoning 在前面,模型必須先**書寫**它對影像的
   觀察,這份觀察會作為後續欄位的條件機率輸入。

> 這是「chain-of-thought 被結構化封裝」的具體做法 — 把 CoT 當成 schema 的一個必須先
> 出現的欄位,而不是放任模型自由 ramble。

### 3.2 為什麼還要做 `fewshot5_guided_noreasoning` 這條對照?

要量「reasoning 欄位在已經有 5 個範例的情況下,是否還有邊際貢獻」。可能結果:

- 5 個範例本身已經教會模型怎麼答,reasoning 變成多餘的 token 浪費 → 拿掉省錢。
- 5 個範例仍然會在邊界 case 上偏掉,reasoning 還在賺取它的 token → 留著。

只有量才知道。

---

## 4. 底層機制 ④ — 為什麼以「閉環安全」設計指標

英文版列了八九個指標,但腳本的列印與 `summary.md` 都把 `wrongDir%` 和 `falseAln%`
擺在最關鍵。原因要從 downstream 怎麼用結果說起。

### 4.1 真正消費 VLM 輸出的是「對位控制器」

實驗的判斷結果不是給人看,是**直接送進閉環控制器**(調整 SEM 載台的 XY 微調)。所以:

- **`wrongDir%`(方向錯誤率)**:VLM 說「角在左邊」其實在右邊。控制器會**往錯誤方向
  推載台**,離對位點愈來愈遠 → **發散**。這是「會把對位過程搞砸」的失敗模式。
- **`falseAln%`(假對齊率)**:VLM 說「已對齊(`aligned=true`)」其實還沒對。控制器
  **過早停止調整**,留下未收斂的對位 → 後續晶圓量測誤差。這是「過早收手」的失敗模式。

對控制器來說,**錯方向 ≫ 不知道 > 知道但量值錯**。所以實驗排序就是
`wrongDir% → falseAln% → magMAE → ox/oy% → align%`,跟人類習慣看 overall accuracy
完全相反。

### 4.2 `wrongDir` 只在「corner 真的存在」的子集上算

```python
if found_subset:
    wrong_dir = int(_opp(("left", "right"), gt["offset_x"], pred["offset_x"])
                    or _opp(("above", "below"), gt["offset_y"], pred["offset_y"]))
```

理由:ground truth 是「找不到角」時,根本沒有方向可以算錯;這種 case 屬於另一條失敗
模式(`corner_found_ok`)。把它混進去會稀釋風險指標。

`_opp` 的判斷:用集合 `{gt, pred} == {a, b}` 來檢查「兩個值剛好是相反的兩端」,
排除掉 `center`/`unknown` → `left` 這種「差一格」的情況(這些算 ox/oy 不對,但不是
「方向相反」)。

### 4.3 為什麼 `align%` 反而最不重要

兩個原因:

1. **類別不平衡**:如果 eval set 八成是「未對齊」,模型只要每次答 `aligned=false`,
   `align%` 就有 80%,卻完全沒判斷能力。
2. **複合指標**:`aligned` 是 `offset_x == center && offset_y == center` 的合成,
   單獨看會把 axis 細節隱藏。

所以腳本只把 `align%` 當 sanity check,並附上 **Wilson 95% CI** 提醒讀者
「點估計不要太認真」。

### 4.4 Wilson CI 為什麼選它,不是常見的 normal-approximation?

```python
def wilson(k: int, n: int, z: float = 1.96): ...
```

當 `n` 小(實驗常見 30–100)或 `p` 接近 0/1(像是 `wrongDir%` 我們希望接近 0)時,
傳統的 `p ± 1.96·√(p(1-p)/n)` 會給出**不合理的區間**(可能跑出 [-0.05, 0.10] 這種)。
Wilson 區間是調整過邊界行為的版本,在小樣本和極端 p 上都比較準。

> 翻譯成白話:**在我們的樣本量下,Wilson CI 比較不會說謊。**

---

## 5. 底層機制 ⑤ — `temperature=0` 為什麼還要做 `--repeats 3`?

理論上 `temperature=0` 應該是 deterministic(每次取最大機率的 token),實際上**在 vLLM
這類批次推論引擎上不是**:

- **批次組合不同**:同一個請求,跟其他請求一起 batch 時的 attention/kernel 路徑不一定
  相同 → 浮點誤差累積 → 第一個有歧義的 token 抽不同了 → 整段輸出岔開。
- **continuous batching**:新請求進來後 batch 大小變了,做 kernel 選擇可能不同。

腳本因此跑 `--repeats 3`,在 `summary.md` 印 `align%±sd`:

- **sd ≈ 0**:模型對這條 input 真的很穩,可以信任單次判斷。
- **sd 明顯**:單次判斷有風險,production 應該**多數決(majority vote)** 或加
  stability gating。

> 這不是「模型不可靠」,是 serving 層的物理現實,你必須量它。

---

## 6. 腳本中與底層機制對應的程式碼導讀

### 6.1 影像如何進 message

```python
def _data_uri(img_path: str) -> str:
    with Image.open(img_path) as raw:
        img = raw.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
```

OpenAI 風格的 multimodal message,影像用 `data URI`(base64-PNG)塞進
`{"type": "image_url", "image_url": {"url": "..."}}`。vLLM 解析 URL,如果開頭是
`data:image/...`,就**從 base64 直接還原 bytes**,不會去發 HTTP。

> **PNG 是 lossless,而 SEM 灰階對亮邊的判讀對 JPEG 壓縮假影很敏感**,所以即使影像比較
> 大,也用 PNG。

### 6.2 範例 URI 快取(本次新增的最佳化)

```python
_URI_CACHE: Dict[str, str] = {}

def _cached_data_uri(img_path: str) -> str:
    uri = _URI_CACHE.get(img_path)
    if uri is None:
        uri = _data_uri(img_path)
        _URI_CACHE[img_path] = uri
    return uri
```

範例影像在實驗中是**常數**(N_eval × N_repeats × N_conditions 次都用同一批 5 張),
若每次都 PIL 讀檔 + RGB 轉 + PNG 編碼 + base64,客戶端 CPU 會被白白吃掉。快取掉就好;
live item 的 URI 則照舊每次算(因為它每張不同)。

### 6.3 Transient 錯誤的有限重試(本次新增)

```python
_RETRYABLE_ERROR_NAMES = {
    "APITimeoutError", "APIConnectionError", "InternalServerError",
    "RateLimitError", "ConnectionError", "Timeout", "ReadTimeout",
}
```

只重試**運輸層**問題(timeout、暫時性 5xx、connection drop)。
**不重試** schema 拒絕、auth 錯誤、bad request — 那些是 signal,要立刻冒上來。
backoff 用 1s/4s,刻意短;實驗只是 R&D 想跑完,不是 production SLA。

### 6.4 malformed 預測在 `score_one` 的處理(本次補註解)

```python
if pred is None:
    return {..., "wrong_dir": (1 if found_subset else None),
                  "false_aligned": (0 if not_aligned_subset else None), ...}
```

- **`wrong_dir` 算成 1**(在 corner-present 子集上):malformed 回應在 production 等於
  「沒有結論可用」,控制器只能 fallback / retry,這對對位來說就是一個風險事件,
  跟「答錯方向」一樣會讓循環不順 → 視為 wrong_dir。
- **`false_aligned` 算成 0**(在 not-aligned 子集上):malformed 回應根本沒說
  `aligned=true`,所以**不會觸發**「過早停止」這個失敗模式 → 不計入 falseAln。

兩個都是「跟控制器實際會看到的行為對齊」的選擇,所以看起來不對稱,但語意上一致。

### 6.5 `--mechanism` 對應的 kwargs

```python
def _structured_kwargs(schema, mechanism):
    if mechanism == "response_format":
        return {"response_format": {"type": "json_schema",
                                    "json_schema": {"name": "result", "schema": schema, "strict": True}}}
    return {"extra_body": {"guided_json": schema}}
```

兩條路徑互斥,腳本只送其中一個。當你的 vLLM 同時吃兩種時,跑一次每種、比對結果穩定
性後再選 production 的版本。

---

## 7. 實際操作 — 從零跑一次

```bash
# 0. 環境
pip install openai pillow
export VLLM_BASE_URL="http://your-vllm-host:8000/v1"
export VLLM_API_KEY="EMPTY"
export VLM_MODEL="Qwen3.6-27B"

# 1. 準備資料夾(eval_manifest.json + exemplar_manifest.json + 影像)
ls ./sem_eval/
# eval_manifest.json  exemplar_manifest.json  img001.png  img002.png  ...

# 2. 先用 --limit 跑一個小子集驗證 pipeline 通
python run_fewshot_experiment.py --data-dir ./sem_eval --limit 5 --repeats 1

# 3. 跑單一條件 sanity check
python run_fewshot_experiment.py --data-dir ./sem_eval --repeats 1 \
    --conditions zeroshot_guided_reasoning

# 4. 正式跑完整 ladder + 3 次 repeats
python run_fewshot_experiment.py --data-dir ./sem_eval --repeats 3

# 5. 看結果
cat experiment_out/summary.md
head experiment_out/raw_results.csv
```

### 7.1 怎麼讀 `summary.md`?

按照「閉環安全」順序,從左往右看:

1. **`fmt%`** 應該 ~100%(guided 條件下)或 ≥ 90%(freetext 條件)。如果 guided 條件下
   不到 100%,通常是 schema 跟模型版本兼容問題,要先解決。
2. **`wrongDir%`**:絕對值要 < 5%(經驗值,需依產品需求設定);條件間的相對下降才是
   重點。如果 `fewshot5` 沒有比 `zeroshot_guided_reasoning` 顯著下降(看 CI 是否分離),
   表示 few-shot 對方向判斷沒幫助。
3. **`falseAln%`**:同上,絕對值要小、相對要降。
4. **`ox% / oy%`**:可用感知指標;通常會跟 `wrongDir%` 反向變化(直覺對的話)。
5. **`magMAE`**:收斂速度指標,愈低愈好,但 `wrongDir%` 沒先壓下來,這個沒意義。
6. **`align%` 看 CI 寬度**:如果 CI 是 [40, 80] 這種,等於沒結論 → 加資料。
7. **`lat_ms`**:成本軸。如果 few-shot 贏但 lat 翻倍,要去開啟 vLLM 的 prefix cache。

---

## 8. 跟 yJarvis 後端的關係 — 為什麼「先量、後建」這個順序很重要

英文版 §10 的圖把 yMinion / yJarvis / vLLM 三層的職責劃出來。從中文角度再強調一次:

- **yJarvis 後端要做的事極簡**:接 HTTP,把 `messages` + `guided_json` 原封不動轉發給
  vLLM,把 response 原封不動回傳。這稱為**「無狀態 relay」**。
- **業務邏輯(prompt、schema、few-shot 範例選擇、`k` 是多少)全部留在 client 端**
  (yMinion 模組 `sem_corner_judge` 裡,跟 git 版本控管在一起)。

這樣設計的回報:**未來新增任何 vision judge(`yellow_score_box_judge`、其他 CV 任務的
驗證 judge),都不需要改 yJarvis**。一個端點服務全部。

而這個實驗的角色:**證明這個設計值得做**。

- 若實驗結果是「分支 A」(few-shot 顯著贏),就建 `B` 端點 + client 機制,並用實驗
  量出的 `k` 上線。
- 若是「分支 B」(`guided_json + reasoning` 就夠),仍然建 `B` 端點(為了 schema 可靠性),
  但**跳過 few-shot 機制**,省下 token 與維護成本。
- 若是「分支 D」(模型 perception 不夠),**完全不建** `B` 端點,留在 CV;等下一代
  視覺模型再回來跑這個實驗。

> 這也是為什麼 §11 強調**永久回歸 eval**:模型/vLLM 一升級,就重跑一次,看分支會不會
> 從 D → A。這個實驗框架本身就是組織學習的記憶體。

---

## 9. 與英文版的對照表

| 概念 | 中文章節 | 英文章節 |
|---|---|---|
| Role 矩陣 / few-shot 排列 | §1 | (隱含於 §3、§8 程式碼) |
| `guided_json` vs `response_format` | §2 | §2、§7 `--mechanism` |
| Reasoning-first 結構化 CoT | §3 | §1(H2)、§3 條件 ladder |
| 閉環安全指標 | §4 | §5 |
| 確定性 / repeats | §5 | §6 |
| 程式碼導讀 | §6 | §8 |
| 結果解讀與決策 | §7.1 | §9 決策樹 |
| 與 yJarvis 後端的關係 | §8 | §10、§11 |

---

## 10. 一頁總結(印出來貼牆上)

1. **Role 矩陣**:`system` 是規則、`user` 是題目、`assistant` 是答案。few-shot 用
   user/assistant 交替,模仿模型訓練時的格式。
2. **`guided_json`** 給可靠的 JSON,**`response_format`** 是 OpenAI 標準寫法,
   實驗兩條都跑過再決定 production 用哪條。
3. **Reasoning-first** 利用自回歸特性,**先逼模型寫推理、再決定欄位**,等於把 CoT
   結構化封裝。
4. **指標排序**:`wrongDir% → falseAln% → magMAE → ox/oy% → align%`,因為下游是控制器
   不是看板。
5. **temperature=0 仍要 repeats**:vLLM batch 不保證 bit-identical,變異要量。
6. **Few-shot 排程要看 Wilson CI**:小樣本,別用點估計做決策。
7. **先量、後建**:實驗驅動後端決定,不是反過來。

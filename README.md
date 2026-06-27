# Question Evolution Pipeline

本目录包含一套用于 **PoliceQA 数据难度进化** 的脚本。核心目标：对当前候选模型（如 Qwen）得分过高的题目进行"题目进化"，使其能更好地区分强模型与弱模型，从而提升合成训练数据的质量。

> 背景：在 PoliceQA 上，rubric 增强遇到瓶颈——候选答案（Qwen）并不显著差于参考答案（GPT），差异多在风格偏好。因此我们把进化对象从 **rubric** 转向 **question**：让题目本身变得更难、更需要深度推理。

---

## 1. 整体流程

当前流水线准确流程是：

  1. 原始题目 + 原始 reference + 原始 rubric + 原始 score_prompt
     ↓
  2. Qwen 回答原题，得到 candidate_answer
     ↓
  3. Judge 按 rubric/score_prompt 给 Qwen 的 candidate_answer 打分
     当前默认 Judge 也是 Qwen
     ↓
  4. 挑选 Qwen 得分率 >= 0.8 的高分题
     ↓
  5. GPT-5.4 改写这些高分题，让题目更难
     ↓
  6. GPT-5.4 针对改写后的新题生成 reference
     ↓
  7. GPT-5.4 针对新题 + 新 reference 生成新 rubric 和新 score_prompt
     ↓
  8. Qwen 回答改写后的新题，得到新的 candidate_answer
     ↓
  9. Judge 再按新 rubric/score_prompt 给 Qwen 新答案打分

  所以每轮产物关系是：

  GPT 生成：
  - evolved_prompt
  - reference answer
  - rubric
  - score_prompt

  Qwen 生成：
  - 原题 candidate_answer
  - 新题 candidate_answer

  Judge 生成：
  - scoring_result

  当前代码里 Judge 默认也是 hjl_Qwen3.6-27B，所以可以理解
  成：

  Qwen 自己答题
  Qwen 按 rubric 给自己的答案打分
  GPT 负责把高分题升级并重建参考答案和评分标准
  Qwen 再答新题并评分

整个流水线包含 4 个脚本，执行顺序如下：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 0: 原始数据 (已包含 prompt + reference + rubric)                        │
│  data/police_qa_testset_v2.8.jsonl                                          │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 1: scoring.py  — 用候选模型(Qwen)答题并评分 (Qwen)                       │
│  输入:  *.jsonl (含 prompt, rubric, score_prompt, meta_info.references)       │
│  输出:  *_scored.jsonl (新增 scoring_result)                                  │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 2: question_evolution.py  — 对高分题目进行进化（gpt-5.4）                │
│  输入:  *_scored.jsonl (含 scoring_result)                                    │
│  输出:  *_evolved.jsonl (prompt 被改写, 旧评分产物移入 meta_info.stale_*)      │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 3: collect_answers.py  — 为进化后的题目重新采集参考答案(gpt-5.4)          │
│  输入:  *_evolved.jsonl                                                       │
│  输出:  *_with_answers.jsonl (meta_info.references 被覆盖为新 reference)       │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 4: gen_rubric.py  — 针对新题和新 reference 生成 rubric (gpt-5.4)         │
│  输入:  *_with_answers.jsonl                                                  │
│  输出:  *_rubric.jsonl (新增 rubric, score_prompt)                            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  阶段 5: scoring.py  — 再次用候选模型(Qwen)答题并评分，验证进化效果              │
│  输入:  *_rubric.jsonl                                                        │
│  输出:  *_rubric_scored.jsonl (新增 scoring_result)                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.1 脚本职责速查

| 脚本 | 职责 | 典型输入 | 典型输出 |
| --- | --- | --- | --- |
| `scoring.py` | 调用候选模型生成答案，并用评分模型按 rubric 打分 | `*_rubric.jsonl` | `*_scored.jsonl` |
| `question_evolution.py` | 筛选高分题目，调用强模型把题目改得更难 | `*_scored.jsonl` | `*_evolved.jsonl` |
| `collect_answers.py` | 调用强模型为题目生成参考答案 | `*.jsonl` | `*_with_answers.jsonl` |
| `gen_rubric.py` | 根据题目和参考答案生成 rubric 与 score_prompt | `*_with_answers.jsonl` | `*_rubric.jsonl` |

---

## 2. 数据格式详解

数据采用 **JSONL** 格式，每行一个样本（一个 JSON 对象）。下文按流水线各阶段说明字段的"增删改"。

### 2.1 阶段 0：原始数据

以 `data/police_qa_testset_v2.8.jsonl` 为例，每行至少包含：

```json
{
  "index": 11112,
  "prompt": "在使用重合比较法（将两台摄像机画面重叠或将照片重叠比对）来研判嫌疑人身份时，对两张照片的拍摄角度有什么具体要求？",
  "meta_info": {
    "references": ["参考答案文本，来自 GPT-5.4"],
    "answer_from_book": "教材/资料中的简版答案",
    "source_file": "视频侦查技术-公大社-2015.jsonl",
    "labels": { "topic": ["情报研判"], "difficulty": "专业", ... },
    ...
  },
  "rubric": [
    { "title": "核心要点", "description": "...", "weight": 4 },
    { "title": "边界条件", "description": "...", "weight": 3 },
    { "title": "常见错误", "description": "...", "weight": -2 }
  ],
  "rubric_thought_process": "rubric 设计思路...",
  "score_prompt": "你是严格的模型评测打分员...<<<待评答案>>..."
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `index` | int | 样本编号 |
| `prompt` | string | **题目文本**，是模型需要回答的问题 |
| `meta_info` | object | 元数据容器 |
| `meta_info.references` | list[string] | 参考答案列表，通常取 `[0]` 作为标准答案；由 GPT-5.4 生成 |
| `meta_info.answer_from_book` | string | 教材/资料原始答案，可作参考 |
| `meta_info.labels` | object | 题目标签：主题、难度、题型、场景等 |
| `rubric` | list[object] | 评分标准；`weight > 0` 为加分项，`weight < 0` 为扣分项 |
| `rubric[].title` | string | 评分维度标题 |
| `rubric[].description` | string | 评分细则 |
| `rubric[].weight` | int | 该项满分/扣分值 |
| `rubric_thought_process` | string | 生成 rubric 时的设计思路 |
| `score_prompt` | string | 给评分模型的完整提示词，其中 `<<<待评答案>>>` 为占位符 |

---

### 2.2 阶段 1：`scoring.py` 输出

`scoring.py` 会在每条样本上新增一个 `scoring_result` 字段，记录候选模型（如 Qwen）的回答及评分结果。

```json
{
  "index": 11112,
  "prompt": "...",
  "meta_info": { "references": [...], ... },
  "rubric": [...],
  "score_prompt": "...",
  "scoring_result": {
    "answer_mode": "llm",
    "answer_model": "hjl_Qwen3.6-27B",
    "candidate_answer": "候选模型的完整回答文本...",
    "item_scores": [
      { "title": "核心要点", "weight": 4, "awarded": 4, "brief_reason": "覆盖了关键信息" },
      { "title": "边界条件", "weight": 3, "awarded": 2, "brief_reason": "提到了部分边界" },
      { "title": "常见错误", "weight": -2, "awarded": 0, "brief_reason": "未触发" }
    ],
    "overall_comment": "整体评价...",
    "total_awarded": 6,
    "total_possible": 7,
    "judge_model": "hjl_Qwen3.6-27B",
    "judge_raw_response": "评分模型的原始输出..."
  }
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `scoring_result` | object | 评分结果容器 |
| `scoring_result.answer_mode` | string | `reference`（用参考答案自评）或 `llm`（调用候选模型） |
| `scoring_result.answer_model` | string | 生成 candidate_answer 的模型名 |
| `scoring_result.candidate_answer` | string | 候选模型对 `prompt` 的回答 |
| `scoring_result.item_scores` | list[object] | 逐条 rubric 得分；`title` 必须与 `rubric` 严格一致 |
| `scoring_result.total_awarded` | int | 实际总得分（含负分项扣分） |
| `scoring_result.total_possible` | int | 正分项满分之和 |
| `scoring_result.judge_model` | string | 执行评分的模型 |
| `scoring_result.judge_raw_response` | string | 评分模型原始返回，用于 debug |

#### 得分率计算

```text
score_rate = scoring_result.total_awarded / scoring_result.total_possible
```

`question_evolution.py` 默认对 `score_rate >= 0.8` 的题目进行进化。

---

### 2.3 阶段 2：`question_evolution.py` 输出

这是本套程序的核心。它会：

1. **全量输出**所有样本；
2. 对 `score_rate >= --min-score-rate` 的样本：
   - 把原 `prompt` 移到 `meta_info.prompt_old`；
   - 把 `prompt` 替换为模型生成的 `evolved_prompt`；
   - 把原 `rubric` / `score_prompt` / `scoring_result` 移到 `meta_info.stale_*`；
   - 新增 `meta_info.question_evolution_metadata`；
   - 顶层新增 `question_evolved: true`。
3. 对未触发进化的样本：
   - 原样透传；
   - 顶层新增 `question_evolved: false`。

> 为什么要把 rubric/score_prompt/scoring_result 移走？因为 `prompt` 变了，旧的 rubric 和评分结果已经失效，必须重新生成。

#### 进化后的样本示例

```json
{
  "index": 11112,
  "prompt": "升级后的新题目文本...",
  "question_evolved": true,
  "meta_info": {
    "references": ["原参考答案，此时已不适用新题"],
    "prompt_old": "原题文本...",
    "stale_rubric": [
      { "title": "核心要点", "description": "...", "weight": 4 }
    ],
    "stale_score_prompt": "旧的评分提示词...",
    "stale_scoring_result": {
      "total_awarded": 6,
      "total_possible": 7,
      ...
    },
    "question_evolution_metadata": {
      "question_evolved": true,
      "trigger_score_rate": 0.857,
      "question_evolution_model": "gpt-5.4",
      "evolution_strategy": "增加反事实条件与最小充分证据要求...",
      "notes_for_reference": "基本适用",
      "question_evolution_raw_response": "模型原始返回..."
    },
    ...
  }
}
```

#### 字段说明

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `prompt` | string | **进化后的新题目**；后续所有步骤都基于这个 prompt |
| `question_evolved` | bool | 本题是否被进化 |
| `meta_info.prompt_old` | string | 原题文本，便于对比 |
| `meta_info.stale_rubric` | list[object] | 旧 rubric（对新题已失效，仅供参考） |
| `meta_info.stale_score_prompt` | string | 旧 score_prompt（对新题已失效） |
| `meta_info.stale_scoring_result` | object | 旧评分结果（对新题已失效） |
| `meta_info.question_evolution_metadata` | object | 进化元数据 |
| `meta_info.question_evolution_metadata.trigger_score_rate` | float | 触发进化的得分率 |
| `meta_info.question_evolution_metadata.evolution_strategy` | string | 采用的进化策略说明 |
| `meta_info.question_evolution_metadata.notes_for_reference` | string | 原参考答案是否仍适用 |

---

### 2.4 阶段 3：`collect_answers.py` 输出

`collect_answers.py` 会调用强模型（默认 GPT-5.4）为每个 `prompt` 生成参考答案，并覆盖 `meta_info.references`。

输出结构（仅保留关键字段）：

```json
{
  "index": 11112,
  "prompt": "升级后的新题目文本...",
  "meta_info": {
    "references": ["新的参考答案，由 GPT-5.4 针对 evolved_prompt 生成"],
    "prompt_old": "原题文本...",
    "stale_rubric": [...],
    "stale_score_prompt": "...",
    "stale_scoring_result": {...},
    "question_evolution_metadata": {...},
    ...
  }
}
```

注意：`collect_answers.py` 的输出只保留 `index`、`prompt`、`meta_info` 三个顶层字段。原来的顶层 `rubric`、`score_prompt`、`scoring_result`、`question_evolved` 会被丢弃（但 `meta_info` 里的历史信息会保留）。

---

### 2.5 阶段 4：`gen_rubric.py` 输出

`gen_rubric.py` 读取 `meta_info.references[0]` 作为参考答案，为新的 `prompt` 生成 rubric。

输出会在阶段 3 的基础上新增：

```json
{
  "index": 11112,
  "prompt": "升级后的新题目文本...",
  "meta_info": { "references": [...], "prompt_old": "...", ... },
  "rubric": [
    { "title": "...", "description": "...", "weight": 5 }
  ],
  "rubric_thought_process": "...",
  "score_prompt": "...<<<待评答案>>..."
}
```

---

### 2.6 阶段 5：第二次 `scoring.py` 输出

与阶段 1 相同，再次用候选模型（Qwen）回答新题，并用评分模型打分。

```json
{
  "index": 11112,
  "prompt": "升级后的新题目文本...",
  "meta_info": { "references": [...], "prompt_old": "...", ... },
  "rubric": [...],
  "score_prompt": "...",
  "scoring_result": {
    "answer_mode": "llm",
    "answer_model": "hjl_Qwen3.6-27B",
    "candidate_answer": "候选模型对新题的回答...",
    "item_scores": [...],
    "total_awarded": 4,
    "total_possible": 10,
    ...
  }
}
```

通过对比 `meta_info.stale_scoring_result` 和新的 `scoring_result`，即可判断题目进化是否有效：理想情况下，新题的候选模型得分率应显著下降。

---

## 3. 字段流转总表

| 字段 | 阶段 0 原始数据 | 阶段 1 scoring | 阶段 2 question_evolution | 阶段 3 collect_answers | 阶段 4 gen_rubric | 阶段 5 scoring |
| --- | --- | --- | --- | --- | --- | --- |
| `index` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `prompt` | 原题 | 原题 | **可能改写为 evolved_prompt** | evolved_prompt | evolved_prompt | evolved_prompt |
| `meta_info.references` | 原参考答案 | 不变 | 不变（但已失效） | **覆盖为新参考答案** | 不变 | 不变 |
| `meta_info.prompt_old` | 无 | 无 | **新增** | 保留 | 保留 | 保留 |
| `meta_info.stale_rubric` | 无 | 无 | **新增（原 rubric 移入）** | 保留 | 保留 | 保留 |
| `meta_info.stale_score_prompt` | 无 | 无 | **新增** | 保留 | 保留 | 保留 |
| `meta_info.stale_scoring_result` | 无 | 无 | **新增** | 保留 | 保留 | 保留 |
| `meta_info.question_evolution_metadata` | 无 | 无 | **新增** | 保留 | 保留 | 保留 |
| `rubric` | ✓ | ✓ | **被移走** | 无 | **重新生成** | ✓ |
| `score_prompt` | ✓ | ✓ | **被移走** | 无 | **重新生成** | ✓ |
| `scoring_result` | 无 | **新增** | **被移走** | 无 | 无 | **重新生成** |
| `question_evolved` | 无 | 无 | **新增** | 无（丢失） | 无 | 无 |

> 注：`collect_answers.py` 会丢弃顶层非 `meta_info` 字段。如需在阶段 3 后继续知道哪些题被进化过，请读取 `meta_info.question_evolution_metadata.question_evolved`。

---

## 4. 快速开始

### 4.1 单轮运行

当前仓库没有单独维护 `run.sh`。如需单轮验证，请按下方 **4.3 分步运行** 依次执行阶段 1 到阶段 5；如需自动循环多轮，请直接使用 `run_loop.sh`。

### 4.2 多轮循环运行（推荐）

如果你想让 question evolution 自动循环多轮，直到 Qwen 平均得分率低于阈值或达到最大轮数，使用：

```bash
bash run_loop.sh
```

默认配置：

- 最大轮数：`MAX_ROUNDS=20`
- 提前停止阈值：`EARLY_STOP_RATE=0.5`（当某轮 Qwen 平均得分率 < 50% 时停止）
- 每轮触发进化的阈值：`MIN_SCORE_RATE=0.8`

每轮结果保存在 `exp/round_N/` 子文件夹中：

```text
exp/
├── round_0/
│   ├── input.jsonl       # 初始输入（从 data/police_qa_testset_v2.8.jsonl 复制）
│   └── scored.jsonl      # 初始 baseline 评分结果
├── round_1/
│   ├── input.jsonl       # 复制自 round_0/scored.jsonl
│   ├── evolved.jsonl     # question_evolution 输出
│   ├── with_answers.jsonl
│   ├── rubric.jsonl
│   └── scored.jsonl      # 本轮复测结果
├── round_2/
│   └── ...
├── summary.txt           # 各轮平均得分率汇总
└── final/
    └── final_scored.jsonl
```

`run_loop.sh` 会自动生成 `exp/summary.txt`，方便你观察得分率下降趋势：

```text
Round | Avg Score Rate | Status
------|----------------|--------
    0 |         0.7865 | baseline
    1 |         0.6123 | continue
    2 |         0.4532 | early_stop
```

脚本也支持**断点续跑**：如果某轮的 `scored.jsonl` 已存在且非空，则跳过该轮执行，直接读取已有结果进行评估。因此即使中途中断，也可以直接重新运行 `bash run_loop.sh` 继续。

#### 修改循环参数

直接编辑 `run_loop.sh` 顶部的配置区即可：

```bash
MAX_ROUNDS=20
EARLY_STOP_RATE=0.5
MIN_SCORE_RATE=0.8
```

### 4.3 分步运行

```bash
# 阶段 1：用 Qwen 对现有题目答题并评分
python scoring.py \
  --input data/police_qa_testset_v2.8.jsonl \
  --output data/questions_rubric_scored.jsonl \
  --answer-mode llm \
  --answer-base-url http://127.0.0.1:18011/v1 \
  --answer-api-key "" \
  --answer-model hjl_Qwen3.6-27B \
  --concurrency 30

# 阶段 2：对得分率 >= 0.8 的题目进化
python question_evolution.py \
  --input data/questions_rubric_scored.jsonl \
  --output data/questions_evolved.jsonl \
  --min-score-rate 0.8 \
  --concurrency 20

# 阶段 3：为进化后的题目重新采集参考答案
python collect_answers.py \
  --input data/questions_evolved.jsonl \
  --output data/questions_evolved_with_answers.jsonl \
  --concurrency 100 \
  --samples 1 \
  --model gpt-5.4

# 阶段 4：重新生成 rubric
python gen_rubric.py \
  --input data/questions_evolved_with_answers.jsonl \
  --output data/questions_evolved_rubric.jsonl \
  --concurrency 30 \
  --model gpt-5.4

# 阶段 5：再次用 Qwen 答题并评分，验证进化效果
python scoring.py \
  --input data/questions_evolved_rubric.jsonl \
  --output data/questions_evolved_rubric_scored.jsonl \
  --answer-mode llm \
  --answer-base-url http://127.0.0.1:18011/v1 \
  --answer-api-key "" \
  --answer-model hjl_Qwen3.6-27B \
  --concurrency 30
```

---

## 5. 常见问题

### Q1：如何只看哪些题目被进化了？

阶段 2 的输出中，过滤 `question_evolved == true`：

```python
import json

with open("data/questions_evolved.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item.get("question_evolved"):
            print(item["meta_info"]["prompt_old"])
            print("→", item["prompt"])
            print()
```

### Q2：阶段 3 之后如何知道哪些题是进化过的？

阶段 3 会丢弃顶层 `question_evolved`，但保留了 `meta_info.question_evolution_metadata`：

```python
if item["meta_info"].get("question_evolution_metadata", {}).get("question_evolved"):
    print("本题是进化题")
```

### Q3：为什么进化后要把 rubric 移走而不是直接更新？

题目进化后，问题的核心、约束、推理要求都可能改变，旧的 rubric 可能不再匹配新题。因此本程序选择把旧 rubric 标记为 `stale_`，让 `gen_rubric.py` 针对新 prompt 和新 reference 重新生成 rubric，避免评分标准与新题脱节。

### Q4：`--min-score-rate` 设多少合适？

默认值 `0.8` 是一个经验值：得分率 80% 以上说明候选模型（Qwen）基本答对了这道题，题目对当前候选模型区分度不足。你可以根据实际得分分布调整：

- 想进化更多题：降到 `0.7`
- 只想进化满分题：提高到 `0.9` 或 `1.0`

### Q5：可以只对进化题重新采集参考答案吗？

可以。先用 `jq` 或 Python 过滤出 `question_evolved == true` 的样本，再喂给 `collect_answers.py`：

```bash
python -c "
import sys, json
for line in sys.stdin:
    j = json.loads(line)
    if j.get('question_evolved'):
        print(json.dumps(j, ensure_ascii=False))
" < data/questions_evolved.jsonl > data/questions_evolved_only.jsonl

python collect_answers.py --input data/questions_evolved_only.jsonl --output data/questions_evolved_only_with_answers.jsonl ...
```

---

## 6. 各脚本参数速查

### scoring.py

```bash
python scoring.py \
  --input INPUT.jsonl \
  --output OUTPUT.jsonl \
  --answer-mode {reference,llm} \
  --answer-base-url URL \
  --answer-api-key KEY \
  --answer-model MODEL \
  --judge-model MODEL \
  --concurrency N \
  --retries N
```

### question_evolution.py

```bash
python question_evolution.py \
  --input INPUT_scored.jsonl \
  --output OUTPUT_evolved.jsonl \
  --min-score-rate 0.8 \
  --model gpt-5.4 \
  --base-url https://hanbbq.labpilot.top/v1 \
  --api-key <API_KEY> \
  --concurrency 20 \
  --retries 3 \
  --prompt-version {v1,v2}
```

### collect_answers.py

```bash
python collect_answers.py \
  --input INPUT.jsonl \
  --output OUTPUT_with_answers.jsonl \
  --samples 1 \
  --concurrency 100 \
  --model gpt-5.4 \
  --retries 3
```

### gen_rubric.py

```bash
python gen_rubric.py \
  --input INPUT_with_answers.jsonl \
  --output OUTPUT_rubric.jsonl \
  --concurrency 30 \
  --model gpt-5.4 \
  --prompt-version v4
```

---

## 7. 调试建议

1. **先看少量样本**：不要直接跑整个数据集。先用 `head -n 5` 切一个小文件验证流程。
2. **检查进化质量**：重点看 `meta_info.prompt_old` → `prompt` 的变化是否合理，是否引入了外部未提供的知识。
3. **对比前后得分**：通过 `meta_info.stale_scoring_result.total_awarded/total_possible` 与新 `scoring_result` 对比，判断进化是否有效。
4. **查看失败文件**：每个脚本失败的数据会写入 `*.failed` 文件，里面包含错误信息。

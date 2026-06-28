# Question Evolution Pipeline

本目录包含一套用于 **PoliceQA 数据难度进化** 的脚本。核心目标：对当前候选模型（如 Qwen）得分过高的题目进行"题目进化"，使其能更好地区分强模型与弱模型，从而提升合成训练数据的质量。

> 背景：在 PoliceQA 上，rubric 增强遇到瓶颈——候选答案（Qwen）并不显著差于参考答案（GPT），差异多在风格偏好。因此我们把进化对象从 **rubric** 转向 **question**：让题目本身变得更难、更需要深度推理。

---

## 1. 整体流程

当前推荐主流程由 `run_loop.sh` 编排，入口是已完成准入的
`admitted_seed_samples.jsonl`。脚本仍保留旧输入回退，但正式实验应显式使用准入样本。

```text
Stage 0: admitted_seed_samples.jsonl
Stage 1: scoring.py -> round_0/scored.jsonl
Stage 2: profile_samples.py -> select_evolution_candidates.py
Stage 3: operator_router.py -> question_evolution.py
Stage 4: validate_evolved_question.py -> candidate_selection.py
Stage 5: collect_answers.py -> gen_rubric.py -> scoring.py
         -> analyze_evolution_effect.py -> update_sample_state.py
```

从 Round 1 开始，每轮 11 个步骤与 `run_loop.sh` 保持一致：

| 步骤 | 脚本 | 产物 |
| --- | --- | --- |
| 0 | 复制上一轮 scored/state 输入 | `round_N/input.jsonl` |
| 1 | `profile_samples.py` | `profiled.jsonl` |
| 2 | `select_evolution_candidates.py` | `profiled_candidates.jsonl` |
| 3 | `operator_router.py` | `routed.jsonl` |
| 4 | `question_evolution.py` | `candidates.jsonl` |
| 5 | `validate_evolved_question.py` | `validated_candidates.jsonl` |
| 6 | `candidate_selection.py` | `evolved.jsonl` |
| 7 | `collect_answers.py` | `with_answers.jsonl` |
| 8 | `gen_rubric.py` | `rubric.jsonl` |
| 9 | `scoring.py` | `scored.jsonl` |
| 10 | `analyze_evolution_effect.py` | `effect_analysis.jsonl`, `effect_matrix.jsonl` |
| 11 | `update_sample_state.py` | `state_updated.jsonl`, memory bank |

`question_evolution.py` 的 legacy 单脚本路径仍可用于兼容旧数据或局部调试，但不再是推荐主流程。推荐路径必须经过画像、分流、路由、复杂度/可回答性校验、候选选择、效果统计和状态更新。

### 1.1 脚本职责速查

| 脚本 | 职责 | 典型输入 | 典型输出 |
| --- | --- | --- | --- |
| `scoring.py` | 调用候选模型生成答案，并用评分模型按 rubric 打分 | `*.jsonl` | `*_scored.jsonl` |
| `profile_samples.py` | 生成样本画像和虚高诊断 | `*_scored.jsonl` | `profiled.jsonl` |
| `select_evolution_candidates.py` | 输出 `evolution_action`，区分进化、低分重构、透传和停止 | `profiled.jsonl` | `profiled_candidates.jsonl` |
| `operator_router.py` | 根据画像、状态和 memory 选择 operator | `profiled_candidates.jsonl` | `routed.jsonl` |
| `question_evolution.py` | 按 operator 生成 1-4 个候选题，支持 validate-retry | `routed.jsonl` | `candidates.jsonl` |
| `validate_evolved_question.py` | 校验复杂度、可回答性、重复题型和格式风险 | `candidates.jsonl` | `validated_candidates.jsonl` |
| `candidate_selection.py` | 从局部树状探索候选中选择主链题目 | `validated_candidates.jsonl` | `evolved.jsonl` |
| `collect_answers.py` | 调用强模型为题目生成参考答案 | `*.jsonl` | `*_with_answers.jsonl` |
| `gen_rubric.py` | 根据题目和参考答案生成 rubric 与 score_prompt | `*_with_answers.jsonl` | `*_rubric.jsonl` |
| `analyze_evolution_effect.py` | 统计轻量边界命中和 operator 效果矩阵 | `*_scored.jsonl` | `effect_analysis.jsonl` |
| `update_sample_state.py` | 更新跨轮状态并写入三类 memory bank | `effect_analysis.jsonl` | `state_updated.jsonl` |

---

## 2. 数据格式详解

数据采用 **JSONL** 格式，每行一个样本（一个 JSON 对象）。下文按流水线各阶段说明字段的"增删改"。

### 2.1 阶段 0：原始数据

推荐入口是 `admitted_seed_samples.jsonl`，每行至少包含：

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
| `score_prompt` | string | 给评分模型的完整提示词，其中 `<<<待评答案>>` 为占位符 |

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

新版主流程优先读取 `evolution_action` 决定是否进化；legacy 单脚本路径才继续使用 `score_rate >= 0.8` 作为触发条件。

---

### 2.3 主流程中间字段：画像、路由、进化、校验和候选选择

当前主流程不是直接把高分题送入统一 prompt，而是依次补充：

1. `profile_samples.py`：新增 `sample_profile` 与 `overscore_diagnosis`。
2. `select_evolution_candidates.py`：新增 `evolution_action`。
3. `operator_router.py`：新增 `operator_route`。
4. `question_evolution.py`：按 operator 生成候选题，新增 `candidate_group_id`、`candidate_id`、`candidate_operator`、`candidate_generation` 和 `meta_info.question_evolution_metadata`。
5. `validate_evolved_question.py`：新增 `validation_result`，可包含 LLM/mock 校验字段 `main_axis_clear`、`answerable`、`external_knowledge_required`、`repeated_pattern_with_previous_round`、`format_difficulty_dominant`。
6. `candidate_selection.py`：在选中记录上新增 `candidate_selection`。

`question_evolution.py` 在需要进化时会把原 `prompt` 移到 `meta_info.prompt_old`，并把旧 `rubric` / `score_prompt` / `scoring_result` 移到 `meta_info.stale_*`；透传样本会保留 `question_evolved=false`。

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

### 2.4 标准闭环：`collect_answers.py` 输出

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

### 2.5 标准闭环：`gen_rubric.py` 输出

`gen_rubric.py` 读取 `meta_info.references[0]` 作为参考答案，为新的 `prompt` 生成 rubric。

输出会在 `collect_answers.py` 的基础上新增：

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

### 2.6 标准闭环：第二次 `scoring.py` 输出

与 baseline scoring 相同，再次用候选模型（Qwen）回答新题，并用评分模型打分。

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

随后 `analyze_evolution_effect.py` 会新增 `effect_analysis`，`update_sample_state.py` 会新增下一轮使用的 `evolution_state`，并把有效、失败和无效生成经验写入 `memory/`。

---

## 3. 字段流转总表

| 字段 | Stage 0 输入 | scoring | profile/select/router | evolution/validation/selection | standard closure | effect/state |
| --- | --- | --- | --- | --- | --- | --- |
| `index` / `sample_id` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `prompt` | 原题 | 原题 | 原题 | 可能改写为候选/选中题 | 新题 | 新题 |
| `sample_profile` / `overscore_diagnosis` | 无 | 无 | 新增 | 保留 | 保留 | 保留 |
| `evolution_action` / `operator_route` | 无 | 无 | 新增 | 消费并保留 | 保留 | 保留 |
| `meta_info.question_evolution_metadata` | 无 | 无 | 无 | 新增 | 保留 | 保留 |
| `validation_result` / `candidate_selection` | 无 | 无 | 无 | 新增 | 保留 | 保留 |
| `rubric` / `score_prompt` | ✓ | ✓ | ✓ | 进化题移入 stale | 重新生成 | ✓ |
| `scoring_result` | 无 | 新增 | 保留 | 进化题移入 stale | 重新生成 | ✓ |
| `effect_analysis` / `evolution_state` | 无 | 无 | 可继承上一轮 | 可继承上一轮 | 无 | 新增 |

> 注：标准闭环脚本可能只保留部分顶层字段；需要跨阶段稳定消费的进化信息应读取 `meta_info.question_evolution_metadata`、`validation_result`、`candidate_selection`、`effect_analysis` 和 `evolution_state`。

---

## 4. 快速开始

### 4.1 单轮运行

先创建本地环境并安装依赖：

```bash
python -m venv .venv
pip install -r requirements.txt
```

真实运行前至少配置以下环境变量之一：

```bash
# profile/question evolution/answer/rubric 可共用 OpenAI-compatible 配置
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
export OPENAI_API_KEY="..."

# 如需拆分配置，可分别设置
export PROFILE_API_KEYS="..."
export EVOLVE_API_KEYS="..."
export ANSWER_API_KEYS="..."
export RUBRIC_API_KEYS="..."

# 候选模型与 judge
export QWEN_BASE_URL="http://127.0.0.1:18011/v1"
export QWEN_API_KEY=""
export QWEN_MODEL="hjl_Qwen3.6-27B"
```

如果你更习惯原来的明文 Python 配置方式，可以直接在本地 `config.py` 中填写：

```python
BASE_URL = "https://hanbbq.labpilot.top/v1"
GPT_MODEL = "gpt-5.4"
HIAPI_KEYS_BIG = ["REPLACE_WITH_YOUR_LOCAL_KEY"]

QWEN_BASE_URL = "http://127.0.0.1:18011/v1"
QWEN_API_KEY = ""
QWEN_MODEL = "hjl_Qwen3.6-27B"
```

`config.py` 已被 `.gitignore` 忽略，脚本会按 `CLI 非空参数 > 环境变量 > config.py > 默认值` 的顺序读取配置。Qwen 本地服务不需要 key 时保持空字符串即可，`scoring.py` 会在 OpenAI SDK 需要参数时内部使用占位值。不要把真实 strong-model API key 写回受版本控制的源码文件；已经暴露过的 key 应在服务端人工轮换。

真实 Bash/API 验收前可先运行预检：

```bash
python check_runtime_environment.py
```

### 4.2 多轮循环运行（推荐）

如果你想让 question evolution 自动循环多轮，直到 Qwen 平均得分率低于阈值或达到最大轮数，使用：

```bash
bash run_loop.sh
```

默认配置：

- 最大轮数：`MAX_ROUNDS=5`
- 提前停止阈值：`EARLY_STOP_RATE=0.5`（当某轮 Qwen 平均得分率 < 50% 时停止）
- 每轮触发进化的阈值：`MIN_SCORE_RATE=0.8`
- 每条样本最多候选：`NUM_CANDIDATES=2`
- 单轮候选总预算：`MAX_CANDIDATE_BUDGET=0`，表示自动使用待进化样本数 × 2

每轮结果保存在 `experiments/YYYY-MM-DD/exp*/round_N/` 子文件夹中：

```text
experiments/YYYY-MM-DD/exp/
├── round_0/
│   ├── input.jsonl       # 初始输入（从 admitted_seed_samples.jsonl 复制）
│   └── scored.jsonl      # 初始 baseline 评分结果
├── round_1/
│   ├── input.jsonl
│   ├── profiled.jsonl
│   ├── profiled_candidates.jsonl
│   ├── routed.jsonl
│   ├── candidates.jsonl
│   ├── validated_candidates.jsonl
│   ├── evolved.jsonl
│   ├── with_answers.jsonl
│   ├── rubric.jsonl
│   ├── scored.jsonl
│   ├── effect_analysis.jsonl
│   ├── effect_matrix.jsonl
│   └── state_updated.jsonl
├── round_2/
│   └── ...
├── memory/
│   ├── operator_memory_bank.jsonl
│   ├── failure_memory_bank.jsonl
│   └── invalid_generation_cases.jsonl
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
MAX_ROUNDS=5
EARLY_STOP_RATE=0.5
MIN_SCORE_RATE=0.8
NUM_CANDIDATES=2
MAX_CANDIDATE_BUDGET=0
VALIDATION_RETRIES=1
```

### 4.3 分步运行

```bash
# Round 0：baseline 评分
python scoring.py \
  --input admitted_seed_samples.jsonl \
  --output round_0_scored.jsonl \
  --answer-mode llm \
  --answer-base-url "$QWEN_BASE_URL" \
  --answer-api-key "$QWEN_API_KEY" \
  --answer-model "$QWEN_MODEL" \
  --judge-base-url "$QWEN_BASE_URL" \
  --judge-api-key "$QWEN_API_KEY" \
  --judge-model "$QWEN_MODEL"

# Step 1：画像
python profile_samples.py \
  --input round_0_scored.jsonl \
  --output round_1_profiled.jsonl \
  --model "$PROFILE_MODEL" \
  --base-url "$PROFILE_BASE_URL"

# Step 2：候选分流
python select_evolution_candidates.py \
  --input round_1_profiled.jsonl \
  --output round_1_profiled_candidates.jsonl \
  --high-score-threshold 0.8

# Step 3：算子路由
python operator_router.py \
  --input round_1_profiled_candidates.jsonl \
  --output round_1_routed.jsonl \
  --memory-dir memory

# Step 4：多候选进化，含 validate-retry
python question_evolution.py \
  --input round_1_routed.jsonl \
  --output round_1_candidates.jsonl \
  --min-score-rate 0.8 \
  --model "$EVOLVE_MODEL" \
  --base-url "$EVOLVE_BASE_URL" \
  --num-candidates 2 \
  --max-candidate-budget 0 \
  --validation-retries 1

# Step 5：复杂度/可回答性校验
python validate_evolved_question.py \
  --input round_1_candidates.jsonl \
  --output round_1_validated_candidates.jsonl \
  --validate-schema

# Step 6：候选选择
python candidate_selection.py \
  --input round_1_validated_candidates.jsonl \
  --output round_1_evolved.jsonl \
  --invalid-output round_1_invalid_generation_cases.jsonl

# Step 7：采集参考答案
python collect_answers.py \
  --input round_1_evolved.jsonl \
  --output round_1_with_answers.jsonl \
  --samples 1 \
  --model "$GPT_MODEL" \
  --base-url "$ANSWER_BASE_URL"

# Step 8：重新生成 rubric
python gen_rubric.py \
  --input round_1_with_answers.jsonl \
  --output round_1_rubric.jsonl \
  --model "$GPT_MODEL" \
  --base-url "$RUBRIC_BASE_URL"

# Step 9：再次评分
python scoring.py \
  --input round_1_rubric.jsonl \
  --output round_1_scored.jsonl \
  --answer-mode llm \
  --answer-base-url "$QWEN_BASE_URL" \
  --answer-api-key "$QWEN_API_KEY" \
  --answer-model "$QWEN_MODEL" \
  --judge-base-url "$QWEN_BASE_URL" \
  --judge-api-key "$QWEN_API_KEY" \
  --judge-model "$QWEN_MODEL"

# Step 10：效果统计
python analyze_evolution_effect.py \
  --before round_0_scored.jsonl \
  --input round_1_scored.jsonl \
  --output round_1_effect_analysis.jsonl \
  --matrix-output round_1_effect_matrix.jsonl

# Step 11：状态更新和 memory bank 写入
python update_sample_state.py \
  --input round_1_effect_analysis.jsonl \
  --output round_1_state_updated.jsonl \
  --memory-dir memory
```

---

## 5. 常见问题

### Q1：如何只看哪些题目被进化了？

`candidate_selection.py` 之后的 `evolved.jsonl` 或标准闭环后的记录中，过滤 `question_evolved == true`：

```python
import json

with open("round_1_evolved.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item.get("question_evolved"):
            print(item["meta_info"]["prompt_old"])
            print("→", item["prompt"])
            print()
```

### Q2：标准闭环之后如何知道哪些题是进化过的？

部分标准闭环脚本可能丢弃顶层 `question_evolved`，但会保留 `meta_info.question_evolution_metadata`：

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
  --judge-base-url URL \
  --judge-api-key KEY \
  --judge-model MODEL \
  --concurrency N \
  --retries N
```

### profile / select / router

```bash
python profile_samples.py --input scored.jsonl --output profiled.jsonl --model "$PROFILE_MODEL" --base-url "$PROFILE_BASE_URL"
python select_evolution_candidates.py --input profiled.jsonl --output profiled_candidates.jsonl --high-score-threshold 0.8
python operator_router.py --input profiled_candidates.jsonl --output routed.jsonl --memory-dir memory
```

### question_evolution.py

```bash
python question_evolution.py \
  --input routed.jsonl \
  --output candidates.jsonl \
  --min-score-rate 0.8 \
  --model gpt-5.4 \
  --base-url "$EVOLVE_BASE_URL" \
  --concurrency 20 \
  --retries 3 \
  --prompt-version {v1,v2} \
  --num-candidates 2 \
  --max-candidate-budget 0 \
  --validation-retries 1
```

### validate / select

```bash
python validate_evolved_question.py --input candidates.jsonl --output validated_candidates.jsonl --validate-schema
python candidate_selection.py --input validated_candidates.jsonl --output evolved.jsonl --invalid-output invalid_generation_cases.jsonl
```

### collect_answers.py

```bash
python collect_answers.py \
  --input INPUT.jsonl \
  --output OUTPUT_with_answers.jsonl \
  --samples 1 \
  --concurrency 100 \
  --model gpt-5.4 \
  --base-url "$ANSWER_BASE_URL" \
  --retries 3
```

### gen_rubric.py

```bash
python gen_rubric.py \
  --input INPUT_with_answers.jsonl \
  --output OUTPUT_rubric.jsonl \
  --concurrency 30 \
  --model gpt-5.4 \
  --base-url "$RUBRIC_BASE_URL" \
  --prompt-version v4
```

### effect / state

```bash
python analyze_evolution_effect.py --before previous_scored.jsonl --input scored.jsonl --output effect_analysis.jsonl --matrix-output effect_matrix.jsonl
python update_sample_state.py --input effect_analysis.jsonl --output state_updated.jsonl --memory-dir memory
```

---

## 7. 调试建议

1. **先看少量样本**：不要直接跑整个数据集。先用 `head -n 5` 切一个小文件验证流程。
2. **检查进化质量**：重点看 `meta_info.prompt_old` → `prompt` 的变化是否合理，是否引入了外部未提供的知识。
3. **对比前后得分**：通过 `meta_info.stale_scoring_result.total_awarded/total_possible` 与新 `scoring_result` 对比，判断进化是否有效。
4. **查看失败文件**：每个脚本失败的数据会写入 `*.failed` 文件，里面包含错误信息。

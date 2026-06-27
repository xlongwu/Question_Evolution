# Question Evolution Pipeline 项目优化方案

版本：v1.0  
面向项目：`xlongwu/Question_Evolution`  
优化目标：从“统一 Prompt 改难题”升级为“样本诊断、算子路由、有状态进化、复杂度校验、效果归因”的 Qwen 能力边界发现系统。

---

## 1. 背景与问题定义

当前项目已经具备完整的基础流水线：初始评分、题目进化、参考答案采集、Rubric 生成、重新答题评分、循环迭代。项目最初的核心逻辑是：对上一轮中得分率较高的样本执行 question evolution，让题目变得更难，再观察 Qwen 在新题上的得分变化。

经过多轮实验后可以确认，单纯依靠一个统一的 question evolution prompt 持续迭代，已经不能稳定找到 Qwen 的真实能力边界。原因不是 prompt 不够长、不够细，而是当前流程缺少三个关键机制：

1. 进化前缺少样本诊断：高分样本不一定是高价值进化样本，低分样本也不一定没有边界价值。
2. 进化中缺少算子路由：不同样本的虚高原因不同，不能用同一种问法持续处理。
3. 进化后缺少效果归因：不能只看平均分下降，而要判断是否命中了预期能力缺口。

因此，下一阶段优化不应继续追求“更强的万能 Prompt”，而应将项目改造成一个机制化的能力边界发现系统。

---

## 2. 已有实验结论总结

### 2.1 原始 Prompt 的问题

原始 question evolution prompt 的核心问题是：容易把题目推向“超长、多模块、强格式、强约束”的方向。虽然原始 prompt 中写了“选择 1-3 种策略，不要全部堆砌”，但实际实验中模型经常把增加约束、显式推理、反事实、综合复杂度、固定格式同时叠加。

实验现象包括：

1. 真正进化样本的平均题长从约 300 字膨胀到接近 2000 字；
2. 题目越来越长，但 Qwen 仍然长期保持 95% 以上得分率；
3. 多数题没有形成稳定压分，只是变成多步骤结构化分析题；
4. 部分样本在多轮后出现无效改写，例如进化后问题与原题完全相同。

结论：原始 prompt 更擅长“把题做大”，不擅长“把问题问尖”。

### 2.2 V3 Prompt 的改进与不足

V3 prompt 引入了单主轴、复杂度预算、禁止长篇格式、输出 `evaluation_focus` 和 `complexity_budget` 等约束。实验结果显示，V3 有效控制了题长膨胀，题目开始围绕“事实、线索、结论”之间的边界进行收束。

V3 的有效点：

1. 题目长度明显下降，不再单调膨胀；
2. 进化方向更聚焦；
3. 在部分样本上压出了新的边界，例如 `8638` 中 Qwen 将“工作流更专业、更方便”这类不属于本题判据的信息错误保留为线索。

V3 的不足：

1. 前几轮进化样本仍然几乎全满分，压分来得偏慢；
2. 过度依赖“边界判断”题型；
3. 对已经能稳定处理基础边界判断的样本，缺少进一步转向机制；
4. 很多题最终变成“哪些可写、哪些只能作线索、哪些不宜写入”的分类变体。

结论：V3 解决了复杂度膨胀，但容易把样本收束成 Qwen 已经较稳定的基础边界判断题。

### 2.3 exp4 / V5 的改进与不足

exp4 / V5 的主要变化是将题目从“边界分层”进一步推进为“最小关键事实 / 最小跳步 / 最小充分条件”型问题。

有效点：

1. 压分出现得更早，`round_1`、`round_2` 就能压中部分样本；
2. `801` 从满分快速降到较低分，说明“独立必要条件识别”是有效能力轴；
3. `13364` 早期明显掉分，说明 Qwen 能判断“时间差不能直接推出具体行为”，但不一定能说准最小关键缺口；
4. `14865` 暴露出“双门槛结论”问题，即模型抓住显眼动作层，但漏掉真正决定定性的性质层。

不足点：

1. 过度依赖“最小关键事实 / 最小前提 / 最小跳步”；
2. 多轮后容易变成同一问法换壳；
3. 某些在 V3 中有效的细层级分界题，在 exp4 中被换成 Qwen 熟悉的隐藏前提识别题后反而失效；
4. 后续轮次推进能力不足，早压中的题容易提前退出，剩余题又容易被 Qwen 吃满。

结论：最小关键事实是有效算子，但不能连续复用，否则会形成新的模板塌缩。

---

## 3. Qwen 当前能力边界判断

### 3.1 Qwen 已经较稳定的能力

基于已有实验，Qwen 在以下能力上表现相对稳定，不应继续作为主力压测方向：

1. 长题、多步骤题、结构化展开；
2. 基础边界判断，例如“线索不能直接写成结论”；
3. 一般复杂场景下的思路组织；
4. 按步骤、按层次回答的任务；
5. 简单的“可写结论 / 只能作线索”二分题。

这些题型继续加长或加结构，通常只会让题目更复杂，但不会显著暴露 Qwen 的新短板。

### 3.2 更值得继续压测的能力边界

后续应重点围绕以下能力边界设计问题。

#### 3.2.1 多个缺口都像有用时，选不准最关键缺口

代表样本：`801`

Qwen 能说“证据不足”，但当题目中存在多个都合理的缺口时，不总能判断哪一个才是真正卡住结论的独立必要条件。

推荐压测方式：候选缺口二选一、独立必要条件比较、被吸收条件识别。

#### 3.2.2 知道不能下结论，但说不准最少还缺哪类事实

代表样本：`13364`

Qwen 能判断“时间差不能直接推出盲区内发生具体行为”，但在追问“最少还缺哪一类事实”时，容易停留在“还要更多证据”这种宽泛表达。

推荐压测方式：最小关键事实、子判断定位、具体行为事实 vs 异常现象区分。

#### 3.2.3 双门槛结论中，抓住显眼动作层，漏掉定性层

代表样本：`14865`

在“强行脱拽式猥亵”这类题中，结论至少包含两个门槛：动作是否发生，以及该动作是否达到特定定性所需的性质层。Qwen 容易抓住衣物位移、拉扯、挣扎等显眼动作，但漏掉性指向接触、行为性质等真正决定定性的事实。

推荐压测方式：双门槛结论拆分、目标子判断定位、显眼动作层 vs 定性性质层比较。

#### 3.2.4 有用但不属于本题依据的信息，容易被保留为线索

代表样本：`8638`

Qwen 有时能判断某个理由不够硬，但不会进一步判断它是否根本不属于本题判据。例如“PSD 更专业、更适合工作流”并不是“高压缩 JPG 是否损伤鉴定比对细节”的直接依据，但模型可能仍将其保留为线索。

推荐压测方式：近似项分层、判据内/判据外区分、相关但不可用信息排除。

#### 3.2.5 反常线索出现后，研判主线切换不稳

代表样本：`6582`

题目要求根据“骑车进盲区、空手走出”这一反常线索，追踪空手人员的后续去向，从而反推藏车点、交接点或转移路径。但 Qwen 容易回到“继续找车”的常规模板。

推荐压测方式：反常线索主线切换、错误主线识别、盯人反推车 vs 继续找车比较。

---

## 4. 总体优化目标

项目下一阶段应从：

```text
高分样本 → 统一 Prompt → 进化题 → 重新评分 → 看平均分
```

升级为：

```text
样本诊断 → 能力轴识别 → 算子路由 → 有状态进化 → 复杂度校验 → 重新评分 → 预期失分验证 → 边界命中统计
```

具体目标包括：

1. 不再把所有高分样本都送入进化，而是先判断是否属于高分虚高；
2. 不再只保留高分样本，低分但真实暴露边界的样本也要进入边界重构流程；
3. 不再使用一个 prompt 覆盖所有题型，而是将进化方式拆解为多个 operator；
4. 不再只看平均分下降，而是评估是否命中预期能力缺口；
5. 不再允许题目靠长度、格式、任务数量压分；
6. 不再让同一样本连续多轮换壳问同一类问题；
7. 将成功和失败都沉淀为可复用经验。

---

## 5. 推荐项目结构

建议在当前仓库基础上新增以下文件与目录：

```text
Question_Evolution/
  question_evolution.py
  collect_answers.py
  gen_rubric.py
  scoring.py
  run_loop.sh

  profile_samples.py
  select_evolution_candidates.py
  operator_router.py
  validate_evolved_question.py
  analyze_evolution_effect.py
  update_sample_state.py

  prompts/
    profile_prompt.py
    router_prompt.py
    operators/
      O1_gap_choice.py
      O2_subclaim_localization.py
      O3_step_jump.py
      O4_near_level_ranking.py
      O5_extra_premise_detection.py
      O6_single_variable_counterfactual.py
      O7_fact_binding_constraint.py
      O8_double_threshold_claim.py
      O9_abnormal_clue_mainline_switch.py

  schemas/
    sample_profile.schema.json
    overscore_diagnosis.schema.json
    evolution_state.schema.json
    evolution_result.schema.json
    validation_result.schema.json
    effect_analysis.schema.json

  memory/
    operator_memory_bank.jsonl
    failure_memory_bank.jsonl
```

各模块职责：

| 模块 | 作用 |
|---|---|
| `profile_samples.py` | 为样本生成画像，判断题型、能力轴、风险与虚高原因 |
| `select_evolution_candidates.py` | 将样本分为高分虚高、低分真实边界、rubric 问题、透传等类型 |
| `operator_router.py` | 根据样本画像、虚高原因、历史状态选择进化算子 |
| `question_evolution.py` | 根据 operator 调用对应 prompt 生成 evolved question |
| `validate_evolved_question.py` | 对题长、任务数、反事实数、题型重复、可回答性做硬校验 |
| `collect_answers.py` | 采集进化题参考答案，未进化样本继续透传 |
| `gen_rubric.py` | 生成与当前能力轴对齐的 rubric |
| `scoring.py` | 对 Qwen 答案重新评分 |
| `analyze_evolution_effect.py` | 判断是否命中能力边界，而不只是看分数下降 |
| `update_sample_state.py` | 更新样本状态，决定继续、换算子、停止或局部树状探索 |

---

## 6. 数据结构设计

### 6.1 `sample_profile`

用于描述题目本身是什么类型、测什么能力、存在什么风险。

```json
{
  "sample_profile": {
    "core_capability": "证据链补强",
    "claim_level": "可疑线索",
    "problem_shape": "候选项区分",
    "reasoning_granularity": "两步链条",
    "answer_mode_expected": "比较型",
    "easy_judgment_risk": "low",
    "external_knowledge_risk": "low",
    "complexity_expansion_risk": "medium",
    "rubric_drift_risk": "medium"
  }
}
```

推荐枚举：

`core_capability`：概念识别、证据链补强、时空关联、边界判断、排他性认定、反事实推理、程序规范、行为模式识别。  
`claim_level`：事实识别、可疑线索、高度怀疑、可写结论、程序合法性判断。  
`problem_shape`：单概念、双概念比较、多条件组合、多阶段流程、候选项区分。  
`answer_mode_expected`：罗列型、比较型、排除型、排序型、选择型。

### 6.2 `overscore_diagnosis`

用于判断候选答案是否属于高分虚高。

```json
{
  "overscore_diagnosis": {
    "is_worth_evolving": true,
    "candidate_overscore_cause": "漏最小关键事实",
    "target_failure_mode": "选错最关键缺口",
    "why_high_score_is_suspicious": "候选答案能泛泛说证据不足，但没有指出最卡结论的独立缺口",
    "recommended_operator": "O1_gap_choice"
  }
}
```

推荐 `candidate_overscore_cause`：

1. 泛化罗列；
2. 层级越推；
3. 题外补设；
4. 漏最小关键事实；
5. 抓显眼点漏关键层；
6. 受干扰信息带偏；
7. 反常线索主线切换失败；
8. 答案写太满超题。

### 6.3 `evolution_state`

用于记录上一轮情况，防止同一样本连续多轮换壳重复。

```json
{
  "evolution_state": {
    "round": 3,
    "previous_operator": "O1_gap_choice",
    "previous_score_rate": 0.54,
    "previous_effect_label": "effective_boundary",
    "previous_failure_mode": "没有区分独立必要条件与被吸收条件",
    "consecutive_full_score_count": 0,
    "consecutive_same_operator_count": 1,
    "avoid_methods": [
      "继续问最少还缺什么",
      "继续问最小前提",
      "继续问最小跳步"
    ],
    "recommended_next_methods": [
      "O2_subclaim_localization",
      "O4_near_level_ranking"
    ],
    "stop_status": "continue"
  }
}
```

### 6.4 `question_evolution_metadata`

建议扩展当前已有字段。

```json
{
  "question_evolution_metadata": {
    "question_evolved": true,
    "trigger_score_rate": 0.92,
    "question_evolution_model": "gpt-5.4",
    "operator_used": "O1_gap_choice",
    "ability_axis": "最小关键事实识别",
    "target_subclaim": "考生是否实际接收并使用提示",
    "boundary_hypothesis": "Qwen 可能能说证据不足，但说不准哪个缺口才是独立必要条件",
    "expected_qwen_failure": "把有帮助的旁证误判为最小关键事实",
    "scoring_anchor": [
      "是否指出真正最卡结论的独立缺口",
      "是否说明另一个补充事实为什么不足或已被吸收"
    ],
    "evolution_strategy": "...",
    "notes_for_reference": "...",
    "raw_response": "..."
  }
}
```

### 6.5 `validation_result`

用于记录复杂度和可回答性校验结果。

```json
{
  "validation_result": {
    "passed": true,
    "main_axis_count": 1,
    "new_facts_count": 2,
    "output_tasks_count": 1,
    "candidate_options_count": 2,
    "counterfactual_count": 0,
    "estimated_prompt_chars": 420,
    "external_knowledge_risk": "low",
    "format_difficulty_risk": "low",
    "repeat_pattern_risk": "low",
    "why_passed": "主轴清晰，候选项数量可控，未引入题外知识"
  }
}
```

### 6.6 `effect_analysis`

用于判断本轮进化是否真正有效。

```json
{
  "effect_analysis": {
    "score_rate_before": 1.0,
    "score_rate_after": 0.54,
    "delta_score_rate": -0.46,
    "expected_failure": "选错最关键缺口",
    "actual_failure": "没有区分独立必要条件与被吸收条件",
    "failure_matched_expected": true,
    "answer_error_concentration": "high",
    "is_boundary_hit": true,
    "effect_label": "effective_boundary",
    "invalid_reason": null,
    "boundary_type": "trainable",
    "optimization_recommendation": "data",
    "generalization_expectation": "high"
  }
}
```

---

## 7. 进化算子体系

### O1. 候选缺口二选一

适用场景：多个补充事实都看似有用，但只有一个是真正最小关键事实。

目标：压测 Qwen 是否能区分“真正独立必要条件”和“有帮助但不决定结论的旁证”。

典型题型：

```text
若只能补充 A/B 中一项，哪一项才是让结论成立的最小关键事实？另一项为什么不足？
```

代表样本：`801`、`13364`。

### O2. 子判断定位

适用场景：目标结论可以拆成多个子判断。

目标：让模型先指出当前结论中哪一层还不能成立，再说明最少缺什么。

典型题型：

```text
当前结论包含两个子判断：动作层与性质层。现有画面已经支持哪一层？哪一层仍不能成立？
```

代表样本：`14865`、`18485`。

### O3. 单步跳跃识别

适用场景：候选答案从线索直接跳到结论。

目标：让模型指出“从哪一步跳到了哪一步”。

典型题型：

```text
这段分析从“可疑动作”跳到了“确定结论”，中间缺少哪一步判断？
```

### O4. 近似项分层

适用场景：几个说法都不能直接写结论，但层级不同。

目标：区分“可写结论 / 只能作线索 / 根本不属于本题依据”。

典型题型：

```text
以下三种理由中，哪一种可以直接支撑结论，哪一种最多只能作为线索，哪一种根本不能作为本题依据？
```

代表样本：`8638`。

### O5. 题干外补设识别

适用场景：候选答案偷偷引入题干外事实。

目标：检查模型是否能识别哪些判断依赖了题干没有提供的信息。

注意：该算子不能滥用。对某些样本，隐藏前提识别可能是 Qwen 已经熟练掌握的题型。

### O6. 单变量反事实

适用场景：只改变一个事实变量，结论顺序或优先级应变化。

目标：压测 Qwen 是否能根据新增信息重排判断优先级。

典型题型：

```text
如果只把条件 A 改为 B，原结论中哪一层判断会变化，哪一层不变？
```

### O7. 具体化约束

适用场景：候选答案泛化罗列、行业套话多。

目标：强制答案绑定题干事实，避免通用模板覆盖。

### O8. 双门槛结论拆分

适用场景：结论由两个门槛组成，模型抓住显眼门槛但漏掉决定性门槛。

目标：压测模型是否能分清“动作发生”与“性质成立”。

典型题型：

```text
该结论至少包含“动作发生”和“行为性质成立”两层。现有画面支持哪一层？哪一层仍缺关键事实？
```

代表样本：`14865`。

### O9. 反常线索主线切换

适用场景：题干出现反常线索后，模型没有切换研判主线，而是退回常规模板。

目标：测试模型能否根据异常事实调整研判主线。

典型题型：

```text
当嫌疑人骑车进入盲区后空手走出，研判主线应从“找车”切换到什么？为什么继续只找车会错过关键线索？
```

代表样本：`6582`。

---

## 8. 算子路由规则

`operator_router.py` 初版可采用规则路由，不必一开始做复杂模型。

### 8.1 路由输入

```json
{
  "sample_profile": {},
  "overscore_diagnosis": {},
  "evolution_state": {},
  "scoring_result": {},
  "previous_round_metadata": {}
}
```

### 8.2 路由输出

```json
{
  "operator_route": {
    "primary_operator": "O1_gap_choice",
    "backup_operators": ["O2_subclaim_localization", "O4_near_level_ranking"],
    "avoid_operators": ["O1_gap_choice_if_same_as_last_round"],
    "routing_reason": "候选答案能说证据不足，但没有抓准真正独立缺口",
    "is_high_value_sample": true,
    "should_use_local_tree_search": false
  }
}
```

### 8.3 初始路由规则

```text
candidate_overscore_cause = 漏最小关键事实
→ O1，次选 O2

candidate_overscore_cause = 层级越推
→ O3，次选 O4

candidate_overscore_cause = 题外补设
→ O5

candidate_overscore_cause = 泛化罗列
→ O7

candidate_overscore_cause = 抓显眼点漏关键层
→ O8

candidate_overscore_cause = 受干扰信息带偏
→ O6 或 O9

target_failure_mode = 反常线索主线切换失败
→ O9

上一轮 operator = O1 且本轮满分
→ 禁止继续 O1，优先 O2/O4/O8

上一轮已边界命中
→ 停止，不继续进化
```

---

## 9. 新版流水线设计

当前主流程：

```text
Round 0:
  scoring.py

Round N:
  question_evolution.py
  collect_answers.py
  gen_rubric.py
  scoring.py
```

建议升级为：

```text
Round 0:
  scoring.py
  profile_samples.py
  select_evolution_candidates.py

Round N:
  profile_samples.py
  select_evolution_candidates.py
  operator_router.py
  question_evolution.py
  validate_evolved_question.py
  collect_answers.py
  gen_rubric.py
  scoring.py
  analyze_evolution_effect.py
  update_sample_state.py
```

### 9.1 `profile_samples.py`

负责生成 `sample_profile` 与 `overscore_diagnosis`。

### 9.2 `select_evolution_candidates.py`

将样本划分为：

```text
evolve_high_score_overscore
reconstruct_low_score_boundary
pass_through_or_review_rubric
stop_evolution
```

### 9.3 `operator_router.py`

根据画像、虚高原因和历史状态选择 operator。

### 9.4 `question_evolution.py`

根据 `operator_used` 调用对应 operator prompt，而不是使用一个统一 prompt。

### 9.5 `validate_evolved_question.py`

校验题目是否通过复杂度预算、可回答性和去重复约束。

### 9.6 `analyze_evolution_effect.py`

判断是否属于有效边界样本。

### 9.7 `update_sample_state.py`

更新样本状态，控制下一轮是否继续、换算子、停止或进入局部树状探索。

---

## 10. 复杂度校验设计

### 10.1 规则校验

硬性规则：

1. 题长建议不超过 900～1200 字；
2. 输出任务最多 2 个，优先 1 个；
3. 候选项最多 3 个；
4. 反事实最多 1 组；
5. 不得要求大表格、多层标签体系、固定句数、复杂编号；
6. 同一样本不得连续两轮使用同一 operator；
7. 上一轮已问“最小关键事实”，本轮不得只换成“最小前提 / 最小跳步”。

### 10.2 LLM 校验

```json
{
  "main_axis_clear": true,
  "answerable": true,
  "external_knowledge_required": false,
  "repeated_pattern_with_previous_round": false,
  "format_difficulty_dominant": false,
  "rubric_can_score_stably": true,
  "reject_reason": null
}
```

校验失败处理：

1. 轻微问题：自动重试 1 次；
2. 严重问题：标记为 `invalid_complexity`；
3. 连续失败：停止该样本进化。

---

## 11. Rubric 优化方案

当前 rubric 生成应从“大而全评分”改为“能力轴锚定评分”。

建议在 `gen_rubric.py` 中读取：

```json
{
  "ability_axis": "最小关键事实识别",
  "expected_qwen_failure": "把有帮助的旁证误判为最小关键事实",
  "scoring_anchor": [
    "是否指出真正独立必要条件",
    "是否说明另一个候选事实为什么不是最小关键事实"
  ]
}
```

Rubric prompt 增加规则：

```text
本题的目标不是全面覆盖所有可写内容，而是重点判断 candidate 是否命中以下能力轴。
Rubric 至少 60% 正向分值应服务于 scoring_anchor。
不得把格式完整性、长篇展开、额外术语作为主要得分来源。
```

这样可以避免题面已经变尖，但 rubric 又重新变成大而全评分标准。

---

## 12. 效果评估指标

不要继续用平均分作为唯一指标。

### 12.1 核心指标

```json
{
  "score_rate_before": 1.0,
  "score_rate_after": 0.54,
  "delta_score_rate": -0.46,
  "boundary_hit": true,
  "failure_matched_expected": true,
  "complexity_passed": true,
  "rubric_drift_risk": "low",
  "answer_error_concentration": "high"
}
```

### 12.2 边界命中标准

`boundary_hit = true` 需要同时满足：

1. `score_rate_after` 明显下降；
2. 实际失分点与 `expected_qwen_failure` 一致；
3. 题目通过复杂度校验；
4. 题目仍然可回答；
5. rubric 没有靠格式、长度、碎细项压分；
6. 失分集中在一个主错误。

### 12.3 Operator 效果矩阵

每轮输出：

| operator_used | 样本数 | 平均降分 | 边界命中数 | 满分样本数 | 无效复杂化数 | 不可回答数 | 重复模板数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| O1_gap_choice | 8 | -0.31 | 4 | 2 | 0 | 0 | 1 |
| O4_near_level | 5 | -0.22 | 2 | 2 | 0 | 0 | 0 |
| O8_double_threshold | 3 | -0.28 | 2 | 1 | 0 | 0 | 0 |

这张表用于回答：

1. 哪类样本适合什么算子；
2. 哪个算子最容易压出 Qwen 边界；
3. 哪个算子容易模板塌缩；
4. 哪些题应该停止继续进化。

---

## 13. 停止条件与样本终态

### 13.1 停止条件

同一样本满足以下任一条件时应停止或改变策略：

1. 已经稳定压出单一、可解释的失分；
2. 连续两轮满分，且换算子后仍无法压分；
3. 连续两轮使用相近题型，出现模板重复；
4. 题目复杂度上升但没有带来更清晰失分；
5. 出现不可回答、多解或题外知识风险；
6. rubric 已难以跟住题面主轴。

### 13.2 样本终态

```text
effective_boundary_sample
invalid_complexity_sample
unanswerable_or_trap_sample
stable_high_score_stop
rubric_issue_review
continue_with_new_operator
local_tree_search_needed
```

---

## 14. 低分样本处理机制

当前项目不能只看高分样本。已有实验显示，部分低分样本已经暴露了更真实的能力边界，例如 `6582`。

建议新增分支：

```text
low_score_boundary_reconstruction
```

适用条件：

1. baseline 得分低；
2. 低分不是因为 rubric 细项、关键词或负向项；
3. 候选答案存在明显主线抓偏；
4. 题目可以重写成更干净的边界测试题。

处理流程：

```text
低分样本
→ 判断低分类型
→ 如果是真实主线抓偏
→ 重写成更干净的边界题
→ 重新评分
→ 验证错误是否稳定复现
```

对 `6582`，推荐使用 O9 反常线索主线切换算子。

---

## 15. 近期小规模验证实验设计

下一轮不建议直接全量运行，应先做 8 条样本的小规模机制验证。

### 15.1 样本池

强有效样本：

1. `801`：最小关键事实 / 独立必要条件；
2. `13364`：时间差不能推出具体行为，缺最小关键事实；
3. `14865`：双门槛结论，抓显眼动作漏性质层。

V3 有效但 exp4 退步样本：

4. `8638`：近似项分层，不要误转成隐藏前提识别。

稳定满分样本：

5. `7337`：用于测试 stop rule 和换算子机制；
6. `18485`：用于测试连续高分样本是否停止。

低分真实边界样本：

7. `6582`：反常线索主线切换；
8. `5486`：大框架正确但实战颗粒度不足。

### 15.2 实验方式

对每条样本生成两个候选题：

```text
候选 A：router 推荐算子
候选 B：历史上未使用过的备选算子
```

先由 validator 选择最优候选进入 scoring。

### 15.3 验收标准

1. 平均题长控制在 300～700 字；
2. 每题主轴数为 1；
3. consecutive same operator rate < 20%；
4. boundary_hit_rate ≥ 30%；
5. invalid_complexity_sample = 0；
6. `801`、`13364`、`14865` 至少保留 2 条高价值边界样本；
7. `7337`、`18485` 若继续满分，应正确触发 stop 或换算子，不再无限循环。

---

## 16. 实施优先级

### 第一阶段：结构化元数据改造

目标：让每轮实验可分析。

任务：

1. 修改 `question_evolution.py`，保存完整 `question_evolution_metadata`；
2. 增加 `operator_used`、`ability_axis`、`expected_qwen_failure`、`scoring_anchor`；
3. 新增 `schemas/`；
4. 新增 `analyze_evolution_effect.py` 的基础版本。

### 第二阶段：样本画像与候选筛选

任务：

1. 新增 `profile_samples.py`；
2. 新增 `select_evolution_candidates.py`；
3. 将样本划分为高分虚高、低分真实边界、rubric 问题和透传样本。

### 第三阶段：算子路由

任务：

1. 新增 `operator_router.py`；
2. 新增 `prompts/operators/`；
3. 第一版优先启用 O1、O2、O4、O8、O9。

### 第四阶段：复杂度校验与停止机制

任务：

1. 新增 `validate_evolved_question.py`；
2. 新增 `update_sample_state.py`；
3. 实现重复题型检测、连续满分停止、有效边界停止。

### 第五阶段：经验库

任务：

1. 新增 `memory/operator_memory_bank.jsonl`；
2. 新增 `memory/failure_memory_bank.jsonl`；
3. 每次有效或失败进化都写入经验单元。

---

## 17. 最小可行改造清单

如果只做一版最小可行改造，建议先完成以下 6 项：

1. 在 `question_evolution.py` 中保存 `operator_used`、`ability_axis`、`expected_qwen_failure`、`scoring_anchor`；
2. 新增 `profile_samples.py`，为样本补充 `sample_profile` 和 `overscore_diagnosis`；
3. 新增 `operator_router.py`，先用规则路由实现 O1、O2、O4、O8、O9；
4. 新增 `validate_evolved_question.py`，做题长、任务数、候选项、反事实、重复题型校验；
5. 新增 `analyze_evolution_effect.py`，输出 `boundary_hit` 与 operator 效果矩阵；
6. 修改 `run_loop.sh`，在每轮中插入 profile、router、validate、effect analysis 步骤。

---

## 18. 最终目标

优化完成后，项目应能稳定回答以下问题：

1. Qwen 到底在哪些 PoliceQA / 犯罪分析题型上容易失分；
2. 失分是来自最小关键事实、双门槛结论、近似项分层、反常线索主线切换，还是题外依据控制；
3. 哪些边界属于可训练边界，值得构造训练数据；
4. 哪些边界更可能是规模敏感边界，需要更大模型；
5. 哪些边界更适合通过系统拆解、模型路由或中间判别步骤解决；
6. 哪类样本适合哪类进化算子；
7. 哪些进化方向会导致无效复杂化或模板塌缩，应避免继续使用。

最终项目不应只是生成越来越多的进化题，而应沉淀出：

1. 高价值原题画像；
2. 有效进化算子模板；
3. 稳定边界样本；
4. 无效复杂化反例；
5. Qwen 能力边界类型矩阵；
6. 可训练边界与系统级边界的优化建议。

一句话总结：

> 项目下一阶段的核心不是继续寻找一个万能 question evolution prompt，而是建立一个有状态、有路由、有校验、有归因的能力边界发现 pipeline。

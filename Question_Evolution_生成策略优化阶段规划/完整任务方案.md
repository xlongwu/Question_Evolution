# Question Evolution 生成策略优化阶段实施方案

版本：v1.0  
面向项目：`xlongwu/Question_Evolution`  
优化目标：当前阶段只优化 question evolution 的生成策略，从“统一 Prompt 改难题”升级为“样本诊断、算子路由、有状态进化、复杂度校验、轻量效果统计”的问题生成流程；评分规则与 rubric 生成逻辑暂时保持不变。

---

## 1. 背景与问题定义

当前项目已经具备完整的基础流水线：初始评分、题目进化、参考答案采集、Rubric 生成、重新答题评分、循环迭代。项目最初的核心逻辑是：对上一轮中得分率较高的样本执行 question evolution，让题目变得更难，再观察 Qwen 在新题上的得分变化。

经过多轮实验后可以确认，单纯依靠一个统一的 question evolution prompt 持续迭代，已经不能稳定找到 Qwen 的真实能力边界。原因不是 prompt 不够长、不够细，而是当前流程缺少三个关键机制：

1. 进化前缺少样本诊断：高分样本不一定是高价值进化样本，低分样本也不一定没有边界价值。
2. 进化中缺少算子路由：不同样本的虚高原因不同，不能用同一种问法持续处理。
3. 进化后缺少轻量效果统计：不能只看平均分下降，还要记录题目预期压测点、使用算子、题长、是否满分与分数变化。

因此，下一阶段优化不应继续追求“更强的万能 Prompt”，也不应在当前阶段扩展到 rubric / judge 可信度建设，而应先把 question evolution 本身改造成可诊断、可路由、可停止、可复盘的生成策略。

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

## 4. 当前阶段目标

项目下一阶段应从：

```text
高分样本 → 统一 Prompt → 进化题 → 重新评分 → 看平均分
```

升级为：

```text
样本诊断 → 能力轴识别 → 算子路由 → 有状态进化 → 复杂度校验 → 重新评分 → 轻量效果统计
```

具体目标包括：

1. 不再把所有高分样本都送入进化，而是先判断是否属于高分虚高；
2. 不再只保留高分样本，低分但真实暴露边界的样本也要进入边界重构流程；
3. 不再使用一个 prompt 覆盖所有题型，而是将进化方式拆解为多个 operator；
4. 不再只看平均分下降，而是记录题目预期压测点、使用算子与分数变化；
5. 不再允许题目靠长度、格式、任务数量压分；
6. 不再让同一样本连续多轮换壳问同一类问题；
7. 将成功和失败都沉淀为可复用经验。

本阶段完整实现以下 question evolution 生成策略相关内容：

1. 样本画像；
2. 候选样本分流；
3. 算子路由；
4. 多 operator prompt；
5. 局部树状探索；
6. 有状态进化；
7. 复杂度校验；
8. 停止条件；
9. 轻量边界命中统计；
10. operator 效果矩阵；
11. operator memory bank；
12. failure memory bank；
13. invalid generation case bank。

原始样本准入作为单独前置过程处理。实施本方案时默认准入已经完成，输入为 `admitted_seed_samples.jsonl`。

这一阶段的成功标准是：

1. 题目不再膨胀；
2. 同一样本不再题型塌缩；
3. 不同样本能走不同进化算子；
4. 新题能更早、更稳定地压到 Qwen 的具体弱点；
5. `collect_answers.py`、`gen_rubric.py`、`scoring.py` 照常运行但不是优化对象；
6. `gen_rubric.py`、rubric prompt、score prompt、rubric item、权重与扣分项设计暂时保持不变。

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
  candidate_selection.py
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
    candidate_selection.schema.json
    effect_analysis.schema.json

  memory/
    operator_memory_bank.jsonl
    failure_memory_bank.jsonl
    invalid_generation_cases.jsonl
```

各模块职责：

| 模块 | 作用 |
|---|---|
| `profile_samples.py` | 为样本生成画像，判断题型、能力轴、风险与虚高原因，不推荐 operator |
| `select_evolution_candidates.py` | 将样本分为高分虚高、低分真实边界、评分噪声或透传等类型 |
| `operator_router.py` | 唯一负责根据样本画像、虚高原因、历史状态和去重复规则选择进化算子 |
| `question_evolution.py` | 根据 operator 调用对应 prompt 生成 evolved question |
| `validate_evolved_question.py` | 对题长、任务数、反事实数、题型重复、可回答性做硬校验 |
| `candidate_selection.py` | 在局部树状探索产生的多个候选题中选择一个进入主链 |
| `collect_answers.py` | 采集进化题参考答案，未进化样本继续透传 |
| `gen_rubric.py` | 继续沿用当前 rubric 生成逻辑，本阶段不修改 |
| `scoring.py` | 对 Qwen 答案重新评分 |
| `analyze_evolution_effect.py` | 轻量统计分数变化、轻量边界命中、使用算子、题长与是否满分 |
| `update_sample_state.py` | 更新样本状态，决定继续、换算子、停止或局部树状探索 |

---

## 6. 数据结构设计

### 6.0 输入数据约定

本方案默认原始样本准入已经由单独流程完成，当前实现不新增 `raw_sample_admission.py`。流水线入口为：

```text
admitted_seed_samples.jsonl
```

该文件应只包含已通过准入筛选、值得进入 question evolution pipeline 的种子样本。准入过程本身不属于本阶段实现范围。

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
    "complexity_expansion_risk": "medium"
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
    "why_high_score_is_suspicious": "候选答案能泛泛说证据不足，但没有指出最卡结论的独立缺口"
  }
}
```

`profile_samples.py` 只回答“样本是什么问题”，不输出推荐算子。`operator_router.py` 再根据诊断结果、历史状态和去重复规则回答“本轮用什么算子”，避免 profile 与 router 同时推荐导致冲突。

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
    "previous_effect_status": "effective_score_drop",
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
    "expected_evaluation_focus": [
      "是否指出真正最卡结论的独立缺口",
      "是否说明另一个补充事实为什么不足或已被吸收"
    ],
    "evolution_strategy": "...",
    "notes_for_reference": "...",
    "raw_response": "..."
  }
}
```

`expected_evaluation_focus` 只作为问题生成元数据使用，用来记录这道题预期压测什么、后续人工分析时观察 Qwen 是否在该点失分，以及下一轮路由时判断是否换算子。禁止将该字段传入 `gen_rubric.py`，也不得用于修改 rubric prompt、score prompt 或扣分项设计。

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

用于轻量记录本轮问题生成是否带来分数变化和轻量边界命中，不做 actual failure 与 rubric item 对齐，也不做 rubric drift 或 judge 稳定性分析。

```json
{
  "effect_analysis": {
    "score_rate_before": 1.0,
    "score_rate_after": 0.54,
    "delta_score_rate": -0.46,
    "operator_used": "O1_gap_choice",
    "question_length": 351,
    "is_full_score": false,
    "lightweight_boundary_hit": true,
    "hit_confidence": "medium",
    "needs_manual_review": true,
    "lightweight_hit_reason": "候选答案错误集中在 expected_evaluation_focus 记录的独立缺口选择上",
    "effect_label": "effective_boundary_probe"
  }
}
```

`lightweight_hit_reason` 只能基于 candidate answer 简要分析和复杂度校验结论，不读取 rubric item，不做 judge agreement，不做 rubric drift。准备写入 `operator_memory_bank.jsonl` 的有效经验必须携带 `hit_confidence` 与 `needs_manual_review`，避免把低置信度误判样本沉淀成错误经验。

### 6.7 `candidate_selection`

用于记录局部树状探索中最终进入主链的候选题。

```json
{
  "candidate_selection": {
    "selected_candidate_id": "cand_2",
    "selected_operator": "O4_near_level_ranking",
    "selection_reason": "该候选题比 O1 更能避免重复最小事实问法，同时继续压测判据内/判据外信息区分",
    "rejected_candidates": [
      {
        "candidate_id": "cand_1",
        "reject_reason": "与上一轮最小关键事实题型重复"
      }
    ]
  }
}
```

### 6.8 `operator_memory_bank.jsonl`

记录有效进化经验，用于后续相似样本的算子选择。

```json
{
  "sample_id": "801",
  "round": 2,
  "sample_signature": {
    "core_capability": "证据链补强",
    "claim_level": "可疑线索",
    "problem_shape": "候选项区分",
    "candidate_overscore_cause": "漏最小关键事实"
  },
  "operator_used": "O1_gap_choice",
  "expected_qwen_failure": "把有帮助的旁证误判为最小关键事实",
  "score_rate_before": 1.0,
  "score_rate_after": 0.54,
  "delta_score_rate": -0.46,
  "question_length": 351,
  "validation_passed": true,
  "hit_confidence": "medium",
  "needs_manual_review": true,
  "effect_label": "effective_boundary_probe",
  "reuse_note": "同类样本优先使用候选缺口二选一，但下一轮不得继续换壳问最小事实"
}
```

### 6.9 `failure_memory_bank.jsonl`

记录失败进化经验，用于避免重复无效路线。

```json
{
  "sample_id": "7337",
  "round": 4,
  "sample_signature": {
    "core_capability": "时空关联",
    "claim_level": "可疑线索",
    "problem_shape": "多条件组合",
    "candidate_overscore_cause": "基础边界判断过稳"
  },
  "operator_used": "O1_gap_choice",
  "score_rate_before": 1.0,
  "score_rate_after": 1.0,
  "failure_type": "operator_ineffective",
  "failure_reason": "连续追问最小连续性缺口，Qwen 已能稳定处理",
  "avoid_note": "同类样本不要继续使用最小缺口题，应切换到单变量反事实或反常线索主线切换"
}
```

### 6.10 `invalid_generation_cases.jsonl`

记录无效生成样本，用于沉淀不可回答、复杂度膨胀、算子错配等反例。

```json
{
  "sample_id": "8638",
  "round": 3,
  "operator_used": "O5_extra_premise_detection",
  "invalid_type": "operator_mismatch",
  "reason": "该样本历史上更适合近似项分层，隐藏前提识别过于容易，导致全程满分",
  "suggested_operator": "O4_near_level_ranking"
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

`operator_router.py` 采用规则路由 + memory bank 经验复用。规则用于给出基础候选，memory bank 用于调整优先级与规避无效路线。

### 8.1 路由输入

```json
{
  "sample_profile": {},
  "overscore_diagnosis": {},
  "evolution_state": {},
  "scoring_result": {},
  "previous_round_metadata": {},
  "operator_memory_matches": [],
  "failure_memory_matches": []
}
```

路由时应优先检索 `memory/operator_memory_bank.jsonl` 和 `memory/failure_memory_bank.jsonl`：

1. 若存在相似 `sample_signature` 的成功经验，则提高对应 operator 优先级；
2. 若存在相似失败经验，则将对应 operator 加入 `avoid_operators`；
3. 若成功经验与规则路由冲突，保留两者为 primary / backup，并在 `routing_reason` 中说明。

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

上一轮已稳定降分且问题质量合格
→ 停止，不继续进化
```

---

## 9. 新版流水线设计

### Stage 0：前置输入

```text
admitted_seed_samples.jsonl
```

原始样本准入作为单独过程处理，实施本方案时默认已经完成。本方案不实现 `raw_sample_admission.py`，也不负责从 `raw_dataset.jsonl` 生成准入样本。

### Stage 1：初始评分

```text
admitted_seed_samples.jsonl
→ scoring.py
→ round_0/scored.jsonl
```

### Stage 2：样本画像与候选分流

```text
round_N/scored.jsonl
→ profile_samples.py
→ select_evolution_candidates.py
→ profiled_candidates.jsonl
```

### Stage 3：算子路由与局部树状探索

```text
profiled_candidates.jsonl
→ operator_router.py
→ question_evolution.py --num-candidates 2/4
→ validate_evolved_question.py
→ candidate_selection.py
→ evolved.jsonl
```

### Stage 4：标准闭环执行

```text
evolved.jsonl
→ collect_answers.py
→ gen_rubric.py
→ scoring.py
→ round_N/scored.jsonl
```

Stage 4 仍然调用 `gen_rubric.py`，但只是沿用当前逻辑，不做任何 rubric 优化。本阶段中 `collect_answers.py`、`gen_rubric.py`、`scoring.py` 仍会照常运行，因为它们是闭环实验所需步骤；但它们的 prompt、评分逻辑、rubric 结构和 judge 策略不属于本阶段优化范围。

### Stage 5：轻量效果统计与状态更新

```text
round_N/scored.jsonl
→ analyze_evolution_effect.py
→ update_sample_state.py
→ operator_memory_bank.jsonl
→ failure_memory_bank.jsonl
→ invalid_generation_cases.jsonl
```

### 9.1 `profile_samples.py`

负责生成 `sample_profile` 与 `overscore_diagnosis`，只做诊断，不推荐 operator。

### 9.2 `select_evolution_candidates.py`

将样本划分为：

```text
evolve_high_score_overscore
reconstruct_low_score_boundary
pass_through_or_scoring_noise
stop_evolution
```

输出样本应增加 `evolution_action` 字段，作为 `question_evolution.py` 的优先执行入口：

```json
{
  "evolution_action": "evolve_high_score_overscore"
}
```

或：

```json
{
  "evolution_action": "reconstruct_low_score_boundary"
}
```

`question_evolution.py` 不再只依赖 `score_rate >= min_score_rate` 判断是否进化，而应优先读取 `evolution_action`：

```text
evolution_action in ["evolve_high_score_overscore", "reconstruct_low_score_boundary"]
→ 执行 question evolution

evolution_action in ["pass_through_or_scoring_noise", "stop_evolution"]
→ 透传
```

这样可以保证 `6582` 这类低分真实边界样本也能进入实际执行链路。

### 9.3 `operator_router.py`

根据画像、虚高原因、历史状态和去重复规则选择 operator。operator 选择只在该模块完成。

### 9.4 `question_evolution.py`

根据 `evolution_action` 判断是否需要进化；需要进化时，根据 `operator_used` 调用对应 operator prompt，而不是使用一个统一 prompt。

### 9.5 `validate_evolved_question.py`

校验题目是否通过复杂度预算、可回答性和去重复约束。

### 9.6 `candidate_selection.py`

根据复杂度校验结果、候选题主轴、可回答性、去重复结果和预期压测方向，从局部树状探索的多个候选中选择一个进入 `evolved.jsonl`。

### 9.7 `analyze_evolution_effect.py`

轻量统计分数变化、使用算子、题长、是否满分和是否重复题型。

### 9.8 `update_sample_state.py`

更新样本状态，控制下一轮是否继续、换算子、停止或进入局部树状探索，并写入三类 memory bank。

### 9.9 局部树状探索机制

局部树状探索正式纳入当前阶段，不再只是小规模实验策略。主流程仍保持链式迭代，但对单个样本允许并行生成 1-4 个候选题，再由 `candidate_selection.py` 选择一个进入主链。

全局候选生成预算：

1. 单轮候选题总数不得超过 `max_candidate_budget`；
2. 默认 `max_candidate_budget = 待处理样本数 × 2`；
3. 只有 `high_value_sample`、`consecutive_full_score_count >= 2`、`low_score_boundary_reconstruction` 或历史多次无效样本允许生成 3-4 个候选；
4. 若候选预算不足，优先保障高价值样本、低分真实边界重构样本和连续满分样本。

默认候选数量：

| 样本类型 | num_candidates |
|---|---:|
| 普通样本 | 1 |
| 高价值样本 | 2 |
| 连续满分样本 | 2 |
| 低分真实边界重构样本 | 2 |
| 历史多次无效样本 | 3-4 |

候选生成方式：

1. 候选 A：router primary_operator；
2. 候选 B：backup_operator_1；
3. 候选 C：历史有效 operator 变体；
4. 候选 D：避开上一轮重复题型的新 operator。

同一样本的多个候选不应只是随机生成的相似题，而应使用不同 operator 或不同子判断方向。

候选选择标准：

1. 主轴是否最清楚；
2. 是否可回答；
3. 是否通过复杂度校验；
4. 是否与上一轮不重复；
5. 是否更可能压中 `expected_qwen_failure`；
6. 是否没有格式复杂度；
7. 是否没有题外知识依赖。

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
  "reject_reason": null
}
```

校验失败处理：

1. 轻微问题：自动重试 1 次；
2. 严重问题：标记为 `invalid_complexity`；
3. 连续失败：停止该样本进化。

重试责任归属：

1. `question_evolution.py` 负责生成候选题；
2. `validate_evolved_question.py` 负责校验并输出 `reject_reason`；
3. 如果校验失败，由 `question_evolution.py` 根据 `reject_reason` 重新调用同一 operator prompt，并把 `reject_reason` 作为修正约束传入，最多重试 1 次；
4. 独立运行 `validate_evolved_question.py` 时只做校验，不负责重新生成。

---

## 11. 当前阶段不做的范围

当前阶段完整实现 question evolution 生成策略相关模块，但仍然不是 Rubric/Judge 优化阶段。以下内容明确不在当前阶段实现：

1. 不修改 `gen_rubric.py` 的 prompt；
2. 不修改 score prompt；
3. 不调整 rubric item、weight、扣分项；
4. 不让 `expected_evaluation_focus` 进入 rubric；
5. 不做 GPT judge / Qwen judge 双评；
6. 不做 judge agreement rate；
7. 不做 rubric drift 检测；
8. 不做 rubric stability flag；
9. 不做 actual failure 与 rubric item 自动对齐。

`gen_rubric.py` 继续沿用当前逻辑。`expected_evaluation_focus` 只用于问题生成元数据、人工复盘和下一轮算子路由，禁止传入 `gen_rubric.py`。上述风险可以在 `effect_analysis` 中以备注形式保留，后续交由 Rubric/Judge 优化阶段处理。

---

## 12. 效果评估指标

不要继续用平均分作为唯一指标。当前阶段使用 `lightweight_boundary_hit_rate` 作为核心指标，但只做问题生成效果统计，不做 judge 校准或 rubric 归因。

### 12.1 核心指标

```json
{
  "score_rate_before": 1.0,
  "score_rate_after": 0.54,
  "delta_score_rate": -0.46,
  "operator_used": "O1_gap_choice",
  "question_length": 351,
  "is_full_score": false,
  "complexity_passed": true,
  "repeated_pattern_with_previous_round": false,
  "lightweight_boundary_hit": true,
  "lightweight_boundary_hit_rate": 0.36
}
```

### 12.2 轻量边界命中定义

`lightweight_boundary_hit = true` 当且仅当：

1. 题目通过复杂度校验；
2. 题目主轴唯一；
3. 题型没有与上一轮重复；
4. `score_rate_after` 下降明显，或打破满分；
5. candidate answer 简要分析显示，错误方向与 `expected_evaluation_focus` 基本一致；
6. 未出现不可回答、多解、题外知识依赖或格式复杂度压分。

第 5 条可以由 `analyze_evolution_effect.py` 使用 LLM 做轻量分析，但不得读取 rubric item，不做 judge agreement，不做 rubric drift。

### 12.3 Sample Type × Operator 效果矩阵

每轮输出：

| core_capability | candidate_overscore_cause | target_failure_mode | operator_used | sample_count | avg_delta_score_rate | lightweight_boundary_hit_count | lightweight_boundary_hit_rate | full_score_count | invalid_complexity_count | repeated_pattern_count |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 证据链补强 | 漏最小关键事实 | 选错最关键缺口 | O1_gap_choice | 8 | -0.31 | 4 | 50% | 2 | 0 | 1 |
| 边界判断 | 受干扰信息带偏 | 判据内/判据外混淆 | O4_near_level_ranking | 5 | -0.22 | 2 | 40% | 2 | 0 | 0 |
| 行为性质识别 | 抓显眼点漏关键层 | 动作层与性质层混淆 | O8_double_threshold | 3 | -0.28 | 2 | 67% | 1 | 0 | 0 |

这张表用于回答：

1. 哪类样本适合什么算子；
2. 哪个算子最容易带来稳定降分；
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
5. 出现不可回答、多解或题外知识风险。

### 13.2 样本终态

```text
effective_boundary_sample
invalid_complexity_sample
unanswerable_or_trap_sample
stable_high_score_stop
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
2. 低分不是因为格式、关键词或负向项等评分噪声；
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

低分样本重构只重写问题，不重写原 rubric，也不以“修正低分”为目标。它的目标是把原始低分中暴露出的主线错误改写成一个更干净的问题，再进入后续标准流水线。

对 `6582`，推荐使用 O9 反常线索主线切换算子。

---

## 15. 完整实现后的验证实验设计

完整机制实现后不建议直接全量运行，应先用 8 条样本做端到端验证，覆盖前置输入读取、画像、分流、路由、局部树状探索、候选选择、标准闭环、效果统计和 memory bank 写入。

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

对每条样本按局部树状探索规则生成候选题：

```text
候选 A：router 推荐算子
候选 B：历史上未使用过的备选算子
候选 C/D：仅对历史多次无效样本启用
```

先由 `validate_evolved_question.py` 校验候选，再由 `candidate_selection.py` 选择最优候选进入 Stage 4。

### 15.3 验收标准

1. 平均题长控制在 300～700 字；
2. 每题主轴数为 1；
3. consecutive same operator rate < 20%；
4. lightweight_boundary_hit_rate ≥ 30%；
5. invalid_complexity_sample = 0；
6. 三类 memory bank 均能按规则写入；
7. `801`、`13364`、`14865` 至少保留 2 条高价值边界样本；
8. `7337`、`18485` 若继续满分，应正确触发 stop、换算子或局部树状探索，不再无限循环。

---

## 16. 实施阶段

### 前置条件：原始样本准入已完成

本方案实施时默认已经存在 `admitted_seed_samples.jsonl`。原始样本准入是独立过程，不在当前实现范围内。

实施顺序约束：

1. 必须按 Stage 1 → Stage 5 顺序实施；
2. 不得先实现 memory bank 再实现 `analyze_evolution_effect.py`；
3. 不得先实现多候选生成而缺少 `validate_evolved_question.py` 和 `candidate_selection.py`；
4. 每个 Stage 完成后必须能在小样本上独立跑通；
5. 每个 Stage 的输出文件必须能作为下一 Stage 的输入文件直接消费。

### Stage 1 实施：结构化元数据与初始评分

任务：

1. 保持 `scoring.py` 当前逻辑；
2. 为每条样本保留 `sample_id`、round、score_rate 等后续路由所需字段；
3. 明确 `scoring.py` 不是当前阶段优化对象。

### Stage 2 实施：样本画像与候选分流

任务：

1. 新增 `profile_samples.py`；
2. 新增 `select_evolution_candidates.py`；
3. 生成 `sample_profile` 与 `overscore_diagnosis`；
4. 输出 `evolution_action`；
5. 将样本划分为高分虚高、低分真实边界、评分噪声和透传样本。

### Stage 3 实施：算子路由与局部树状探索

任务：

1. 新增 `operator_router.py`；
2. 新增 `prompts/operators/`；
3. 实现 O1-O9 operator prompt；
4. 修改 `question_evolution.py`，优先读取 `evolution_action`，并支持 `--num-candidates`；
5. 新增 `validate_evolved_question.py`；
6. 新增 `candidate_selection.py`；
7. 实现候选生成、候选校验、候选选择和最多 1 次 validate-retry。

### Stage 4 实施：标准闭环执行

任务：

1. `collect_answers.py` 照常运行；
2. `gen_rubric.py` 照常运行但不修改；
3. `scoring.py` 照常运行；
4. `run_loop.sh` 将 `evolved.jsonl` 接入标准闭环。

### Stage 5 实施：效果统计、状态更新与经验库

任务：

1. 新增 `analyze_evolution_effect.py`，输出 `lightweight_boundary_hit` 与 Sample Type × Operator 效果矩阵；
2. 新增或扩展 `update_sample_state.py`；
3. 新增 `memory/operator_memory_bank.jsonl`；
4. 新增 `memory/failure_memory_bank.jsonl`；
5. 新增 `memory/invalid_generation_cases.jsonl`；
6. 每轮写入有效经验、失败经验和无效生成反例。

---

## 17. 当前阶段完整改造清单

本阶段一次性规划并实现以下内容：

1. `profile_samples.py`：样本画像；
2. `select_evolution_candidates.py`：候选样本分流与 `evolution_action` 输出；
3. `operator_router.py`：算子路由；
4. `prompts/operators/`：多 operator prompt；
5. `question_evolution.py`：支持 `evolution_action`、多候选生成和 validate-retry；
6. `validate_evolved_question.py`：复杂度与可回答性校验；
7. `candidate_selection.py`：局部树状探索候选选择；
8. `update_sample_state.py`：有状态进化与停止条件；
9. `analyze_evolution_effect.py`：轻量边界命中统计；
10. Sample Type × Operator 效果矩阵；
11. `memory/operator_memory_bank.jsonl`；
12. `memory/failure_memory_bank.jsonl`；
13. `memory/invalid_generation_cases.jsonl`。

仍不实现 rubric prompt 修改、score prompt 修改、judge 校准、rubric drift、rubric stability flag、actual failure 与 rubric item 自动对齐。

---

## 18. 当前阶段交付目标

本阶段完成后，项目应能稳定回答以下问题：

1. 哪些样本值得继续进化，哪些应停止或透传；
2. 哪类样本适合哪类进化算子；
3. 哪些算子能更早、更稳定地压低 Qwen 得分；
4. 哪些进化方向会导致题目膨胀、题型重复或模板塌缩；
5. 哪些低分样本可以重构成更干净的边界测试题；
6. 每轮问题生成在题长、主轴数、重复题型、满分率和降分幅度上是否改善；
7. 哪些样本和算子组合能形成 `lightweight_boundary_hit`；
8. 哪些有效、失败和无效生成经验应进入 memory bank。

当前阶段应沉淀出：

1. 高价值原题画像；
2. 有效进化算子模板；
3. 有状态路由规则；
4. 复杂度校验与停止条件；
5. 无效复杂化反例；
6. operator 效果统计；
7. operator memory bank；
8. failure memory bank；
9. invalid generation case bank。

一句话总结：

> 项目当前阶段的核心不是 Rubric/Judge 优化，而是在原始样本准入已完成的前提下，完整实现 Question Evolution 生成策略：画像、分流、路由、局部树状探索、复杂度校验、状态更新和经验沉淀都纳入当前阶段；评分规则保持不变。

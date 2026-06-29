import json
from typing import Any, Dict, Optional


PROFILE_PROMPT_TEMPLATE = """
# 角色
你是 question evolution 流水线中的样本诊断器。你的任务是先判断样本是什么问题、当前得分是否可能虚高、以及是否值得进入后续进化或边界重构。

你只做诊断，不推荐任何进化算子，不输出 operator、operator_used、recommended_operator、primary_operator 或 operator_route。

# 输入

## 原题
{|prompt|}

## 参考答案
{|reference_answer|}

## 候选答案
{|candidate_answer|}

## 当前得分率
{|score_rate|}

## 评分摘要
{|scoring_summary|}

## 既有元数据
{|metadata|}

# 诊断要求

1. 判断题目主要考查的能力轴、结论层级、题型形状、推理颗粒度和期望回答方式。
2. 判断候选答案是否属于高分虚高，或低分中是否暴露了真实边界价值。
3. 高分虚高时，说明最主要虚高原因；低分边界样本时，说明它暴露的真实主线错误。
4. 给出 2 到 4 个可能值得探索的能力边界方向，写入 `boundary_axis_candidates`。
5. 不要推荐后续使用哪一个算子。算子路由属于下一阶段。
6. 不要修改题目、rubric、score prompt 或参考答案。

# 推荐取值

`core_capability` 可使用：概念识别、证据链补强、时空关联、边界判断、排他性认定、反事实推理、程序规范、行为模式识别。
`claim_level` 可使用：事实识别、可疑线索、高度怀疑、可写结论、程序合法性判断。
`problem_shape` 可使用：单概念、双概念比较、多条件组合、多阶段流程、候选项区分。
`answer_mode_expected` 可使用：罗列型、比较型、排除型、排序型、选择型。
`candidate_overscore_cause` 可使用：泛化罗列、层级越推、题外补设、漏最小关键事实、抓显眼点漏关键层、受干扰信息带偏、反常线索主线切换失败、答案写太满超题、评分噪声、基础边界判断过稳。
`boundary_axis_candidates` 只能从以下集合中选择：结论分层、最关键缺口识别、伪闭环识别、补强项升级判断、题干外补设识别、反常线索主线切换。

# 输出

只返回合法 JSON 对象，不要输出 Markdown 或额外解释：

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
    "boundary_axis_candidates": ["最关键缺口识别", "补强项升级判断", "伪闭环识别"]
  },
  "overscore_diagnosis": {
    "is_worth_evolving": true,
    "candidate_overscore_cause": "漏最小关键事实",
    "target_failure_mode": "选错最关键缺口",
    "why_high_score_is_suspicious": "候选答案能泛泛说证据不足，但没有指出最卡结论的独立缺口"
  }
}
""".strip()


def _compact_json(value: Any) -> str:
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_profile_prompt(
    *,
    prompt: str,
    reference_answer: str = "",
    candidate_answer: str = "",
    score_rate: Optional[float] = None,
    scoring_summary: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    replacements = {
        "{|prompt|}": prompt.strip(),
        "{|reference_answer|}": reference_answer.strip() or "未提供",
        "{|candidate_answer|}": candidate_answer.strip() or "未提供",
        "{|score_rate|}": "未知" if score_rate is None else f"{score_rate:.6f}",
        "{|scoring_summary|}": _compact_json(scoring_summary or {}),
        "{|metadata|}": _compact_json(metadata or {}),
    }
    user_prompt = PROFILE_PROMPT_TEMPLATE
    for placeholder, value in replacements.items():
        user_prompt = user_prompt.replace(placeholder, value)
    return user_prompt

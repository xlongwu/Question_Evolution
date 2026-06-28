import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence


@dataclass(frozen=True)
class OperatorPromptSpec:
    operator_id: str
    name: str
    ability_axis: str
    goal: str
    required_question_shape: str
    avoid: str
    default_evaluation_focus: Sequence[str]


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_prompt(
    spec: OperatorPromptSpec,
    *,
    prompt: str,
    reference_answer: str,
    candidate_answer: str,
    rubric: Any,
    sample_profile: Dict[str, Any],
    overscore_diagnosis: Dict[str, Any],
    evolution_state: Dict[str, Any],
    operator_route: Dict[str, Any],
) -> str:
    input_payload = {
        "sample_profile": sample_profile,
        "overscore_diagnosis": overscore_diagnosis,
        "evolution_state": evolution_state,
        "operator_route": operator_route,
    }
    return f"""
# 角色
你是一位 question evolution 题目生成专家。本轮只能执行指定 operator：{spec.operator_id}（{spec.name}）。

# Operator 目标
能力轴：{spec.ability_axis}
目标：{spec.goal}
题型形态：{spec.required_question_shape}
避免：{spec.avoid}

# 必守边界
- 只生成一道完整、可独立作答的新题。
- 新题只能围绕一个清晰主轴，不靠长篇格式、表格、复杂编号或多任务压分。
- 不修改 rubric，不生成评分标准，不把 expected_evaluation_focus 写进 rubric。
- 不引入题干外事实；如必须比较候选事实，题面要给足比较依据。
- 如果本 operator 不适合该样本，仍要在 operator 范围内收窄问题，不得改用其他 operator。

# 输入画像与路由
{_json_block(input_payload)}

# 原题
{prompt.strip()}

# 参考答案
{reference_answer.strip()}

# 候选答案
{candidate_answer.strip()}

# 现有评分标准（只用于理解原题，不得改写）
{_json_block(rubric if isinstance(rubric, list) else [])}

# 输出
返回合法 JSON 对象，不要输出 Markdown 或额外解释：
{{
  "evolved_prompt": "升级后的新题目，必须完整、可独立作答，并严格符合当前 operator。",
  "evolution_strategy": "说明为什么本 operator 能压测目标弱点，以及如何避免换壳重复。",
  "ability_axis": "{spec.ability_axis}",
  "target_subclaim": "本题压测的最小子判断或关键层级",
  "boundary_hypothesis": "一句话说明预期能力边界",
  "expected_qwen_failure": "一句话说明弱模型最可能犯的错",
  "expected_evaluation_focus": {_json_block(list(spec.default_evaluation_focus))},
  "notes_for_reference": "参考答案是否需要轻量补充；如基本适用则写基本适用"
}}
""".strip()

import json
from typing import Any, Dict


def build_validation_prompt(
    *,
    original_prompt: str,
    evolved_prompt: str,
    metadata: Dict[str, Any],
    validation_result: Dict[str, Any],
) -> str:
    return (
        "你是 question evolution 候选题的轻量校验器，只判断题目本身是否适合作为下一轮评测题。\n"
        "不要生成新题，不要修改 rubric，不要评价模型答案质量。\n\n"
        "请只返回合法 JSON 对象，字段如下：\n"
        "{\n"
        '  "main_axis_clear": true,\n'
        '  "answerable": true,\n'
        '  "external_knowledge_required": false,\n'
        '  "repeated_pattern_with_previous_round": false,\n'
        '  "format_difficulty_dominant": false,\n'
        '  "reject_reason": null,\n'
        '  "reason": "一句话说明判断依据"\n'
        "}\n\n"
        "判定标准：\n"
        "1. main_axis_clear=false：题目同时考多个主轴，或核心问题无法一句话说明。\n"
        "2. answerable=false：题干事实不足，强模型也只能猜测或依赖暗含前提。\n"
        "3. external_knowledge_required=true：需要查阅、行业惯例、题干外事实或外部专业资料才能完成。\n"
        "4. repeated_pattern_with_previous_round=true：只是上一轮问法换壳，尤其是最小事实/最小前提/最小跳步重复。\n"
        "5. format_difficulty_dominant=true：难度主要来自表格、编号、格式或多任务，而不是推理主轴。\n\n"
        f"# 原题\n{original_prompt}\n\n"
        f"# 进化题\n{evolved_prompt}\n\n"
        f"# question_evolution_metadata\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n\n"
        f"# 规则校验初步结果\n{json.dumps(validation_result, ensure_ascii=False, indent=2)}\n"
    )

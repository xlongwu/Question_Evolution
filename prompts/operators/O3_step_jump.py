from .base import OperatorPromptSpec


SPEC = OperatorPromptSpec(
    operator_id="O3_step_jump",
    name="单步跳跃识别",
    ability_axis="推理跳步识别",
    goal="压测模型是否能指出候选分析从哪个线索层跳到了哪个结论层。",
    required_question_shape="要求指出一处关键跳步，并说明中间缺少的判断；输出任务保持单一。",
    avoid="不要让题目变成多阶段流程复盘；不要要求列出所有可能跳步；不要增加复杂格式。",
    default_evaluation_focus=(
        "是否指出具体跳步起点和终点",
        "是否说明中间缺少的判断",
        "是否避免只写证据不足的空泛结论",
    ),
)

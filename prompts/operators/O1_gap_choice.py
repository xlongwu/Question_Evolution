from .base import OperatorPromptSpec


SPEC = OperatorPromptSpec(
    operator_id="O1_gap_choice",
    name="候选缺口二选一",
    ability_axis="独立必要条件识别",
    goal="让模型在两个都像有帮助的补充事实中选出真正卡住结论的最小关键缺口。",
    required_question_shape="给出 A/B 两个近似候选，要求判断哪一个才是最小关键事实，并说明另一个为什么不足或已被吸收。",
    avoid="不要泛问还缺什么；不要给超过两个核心候选；不要把最小事实、最小前提、最小跳步换壳堆叠。",
    default_evaluation_focus=(
        "是否选出真正独立必要条件",
        "是否说明另一个候选为什么不能单独支撑结论",
        "是否避免泛泛回答仍需更多证据",
    ),
)

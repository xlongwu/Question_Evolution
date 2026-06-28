from .base import OperatorPromptSpec


SPEC = OperatorPromptSpec(
    operator_id="O8_double_threshold_claim",
    name="双门槛结论拆分",
    ability_axis="双门槛结论识别",
    goal="压测模型是否能分清显眼动作层与真正决定定性的性质层。",
    required_question_shape="把目标结论拆成两个门槛，要求判断现有事实支持哪一层、哪一层仍缺关键事实。",
    avoid="不要只追问动作是否发生；不要把性质层藏在 rubric；不要扩大成多结论综合题。",
    default_evaluation_focus=(
        "是否区分动作发生层和性质成立层",
        "是否指出真正决定定性的缺口",
        "是否避免被显眼动作层替代关键性质层",
    ),
)

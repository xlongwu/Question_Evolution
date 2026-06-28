from .base import OperatorPromptSpec


SPEC = OperatorPromptSpec(
    operator_id="O9_abnormal_clue_mainline_switch",
    name="反常线索主线切换",
    ability_axis="反常线索驱动的研判主线切换",
    goal="测试模型能否根据异常事实调整研判主线，而不是退回常规模板。",
    required_question_shape="给出一个反常线索，要求说明研判主线应切换到什么，以及继续旧主线会漏掉什么。",
    avoid="不要要求完整侦查方案；不要继续只问常规目标；不要引入题干未给出的后续事实。",
    default_evaluation_focus=(
        "是否根据反常线索切换主线",
        "是否说明旧主线为什么会漏掉关键路径",
        "是否能把新主线绑定到题干异常事实",
    ),
)

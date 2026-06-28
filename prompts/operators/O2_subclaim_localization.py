from .base import OperatorPromptSpec


SPEC = OperatorPromptSpec(
    operator_id="O2_subclaim_localization",
    name="子判断定位",
    ability_axis="目标子判断定位",
    goal="先定位目标结论中哪一层尚不能成立，再要求补足该层的最小事实。",
    required_question_shape="把结论拆成 2 个以内子判断，要求指出已支持哪一层、未支持哪一层以及缺口所在。",
    avoid="不要同时考多个结论；不要把答案变成开放式长篇分析；不要只问最少还缺什么。",
    default_evaluation_focus=(
        "是否准确定位尚不能成立的子判断",
        "是否区分已支持层和未支持层",
        "是否把缺口绑定到题干事实而非泛化补证",
    ),
)

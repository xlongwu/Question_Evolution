from .base import OperatorPromptSpec


SPEC = OperatorPromptSpec(
    operator_id="O7_fact_binding_constraint",
    name="具体化约束",
    ability_axis="题干事实绑定",
    goal="抑制泛化罗列，迫使答案绑定题干中的具体事实与因果链。",
    required_question_shape="要求回答只能使用题干事实，围绕一个结论说明最关键的事实绑定关系。",
    avoid="不要把难度建立在字数限制或格式限制上；不要允许行业套话替代题干事实。",
    default_evaluation_focus=(
        "是否紧扣题干具体事实",
        "是否建立事实与结论的直接绑定",
        "是否避免通用模板或行业套话",
    ),
)

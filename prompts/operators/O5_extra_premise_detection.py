from .base import OperatorPromptSpec


SPEC = OperatorPromptSpec(
    operator_id="O5_extra_premise_detection",
    name="题干外补设识别",
    ability_axis="隐藏前提识别",
    goal="检查模型是否能识别结论中哪些关键判断依赖题干未提供的信息。",
    required_question_shape="给出一个看似成立的判断，要求指出其中哪一步偷偷引入了题干外前提。",
    avoid="不要把所有常识都当作题外前提；不要把题目做成容易满分的泛泛隐藏前提题。",
    default_evaluation_focus=(
        "是否指出具体的题干外前提",
        "是否说明该前提为何不可从题干推出",
        "是否保留题干内已成立的信息",
    ),
)

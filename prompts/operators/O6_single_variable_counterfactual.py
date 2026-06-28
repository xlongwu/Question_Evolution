from .base import OperatorPromptSpec


SPEC = OperatorPromptSpec(
    operator_id="O6_single_variable_counterfactual",
    name="单变量反事实",
    ability_axis="单变量条件变化下的结论重排",
    goal="只改变一个事实变量，压测模型是否能判断哪一层结论变化、哪一层不变。",
    required_question_shape="明确只改一个条件，并要求比较改动前后一个核心判断的变化。",
    avoid="不要加入多组反事实；不要要求重写完整方案；不要让条件变化依赖题外知识。",
    default_evaluation_focus=(
        "是否只围绕单一变量变化作答",
        "是否区分变化层和不变层",
        "是否避免被无关条件带偏",
    ),
)

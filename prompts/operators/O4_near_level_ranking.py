from .base import OperatorPromptSpec


SPEC = OperatorPromptSpec(
    operator_id="O4_near_level_ranking",
    name="近似项分层",
    ability_axis="判据内外与证据层级区分",
    goal="让模型区分可直接支撑结论、最多作为线索、以及不属于本题依据的近似项。",
    required_question_shape="提供 2-3 个近似理由或事实层级，要求排序或分层，并说明关键分界。",
    avoid="不要做简单可写/不可写二分；不要引入题外专业标准；不要使用大表格。",
    default_evaluation_focus=(
        "是否区分判据内依据和判据外信息",
        "是否说明近似项之间的层级差异",
        "是否排除相关但不可用的信息",
    ),
)

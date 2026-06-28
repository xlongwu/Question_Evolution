from typing import Any, Dict

from .O1_gap_choice import SPEC as O1_SPEC
from .O2_subclaim_localization import SPEC as O2_SPEC
from .O3_step_jump import SPEC as O3_SPEC
from .O4_near_level_ranking import SPEC as O4_SPEC
from .O5_extra_premise_detection import SPEC as O5_SPEC
from .O6_single_variable_counterfactual import SPEC as O6_SPEC
from .O7_fact_binding_constraint import SPEC as O7_SPEC
from .O8_double_threshold_claim import SPEC as O8_SPEC
from .O9_abnormal_clue_mainline_switch import SPEC as O9_SPEC
from .base import OperatorPromptSpec, build_prompt


OPERATOR_SPECS = {
    spec.operator_id: spec
    for spec in (
        O1_SPEC,
        O2_SPEC,
        O3_SPEC,
        O4_SPEC,
        O5_SPEC,
        O6_SPEC,
        O7_SPEC,
        O8_SPEC,
        O9_SPEC,
    )
}


def get_operator_spec(operator_id: str) -> OperatorPromptSpec:
    try:
        return OPERATOR_SPECS[operator_id]
    except KeyError as exc:
        raise ValueError(f"Unknown operator_id: {operator_id}") from exc


def build_operator_prompt(
    operator_id: str,
    *,
    prompt: str,
    reference_answer: str,
    candidate_answer: str,
    rubric: Any,
    sample_profile: Dict[str, Any],
    overscore_diagnosis: Dict[str, Any],
    evolution_state: Dict[str, Any],
    operator_route: Dict[str, Any],
) -> str:
    return build_prompt(
        get_operator_spec(operator_id),
        prompt=prompt,
        reference_answer=reference_answer,
        candidate_answer=candidate_answer,
        rubric=rubric,
        sample_profile=sample_profile,
        overscore_diagnosis=overscore_diagnosis,
        evolution_state=evolution_state,
        operator_route=operator_route,
    )

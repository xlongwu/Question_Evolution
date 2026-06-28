# Operator Prompts

Stage 3 implements O1-O9 as small prompt specs:

- `O1_gap_choice.py`: candidate gap choice.
- `O2_subclaim_localization.py`: subclaim localization.
- `O3_step_jump.py`: one-step jump identification.
- `O4_near_level_ranking.py`: near-level evidence ranking.
- `O5_extra_premise_detection.py`: extra-premise detection.
- `O6_single_variable_counterfactual.py`: single-variable counterfactual.
- `O7_fact_binding_constraint.py`: fact-binding constraint.
- `O8_double_threshold_claim.py`: double-threshold claim split.
- `O9_abnormal_clue_mainline_switch.py`: abnormal-clue mainline switch.

`__init__.py` exposes the registry consumed by `question_evolution.py`.

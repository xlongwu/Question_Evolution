# Pipeline Field Contract

This directory defines the minimal JSONL contract used by the question
evolution stages. The schemas are intentionally small and permissive: they
stabilize field names, ownership, and runtime validation for the current
Stage 0-5 question evolution pipeline.

## Record Ownership

- `sample_id`, `index`, `round`, `prompt`, `meta_info`, `rubric`,
  `score_prompt`, `scoring_result`, `score_rate`, and `question_evolved` are
  shared pipeline fields.
- `sample_profile` is produced by `profile_samples.py`; `overscore_diagnosis`
  is produced by the same profiling step and consumed by
  `select_evolution_candidates.py`.
- `operator_route` is produced by the Stage 3 router and consumed by
  `question_evolution.py` when `evolution_action` requires evolution.
- `candidate_group_id`, `candidate_id`, `candidate_operator`, and
  `candidate_generation` are Stage 4 intermediate fields produced only when
  `question_evolution.py --num-candidates` is greater than one.
- `evolution_state` is the cross-round state produced by
  `update_sample_state.py` and consumed by candidate selection, routing, and
  stop rules in later rounds.
- `meta_info.question_evolution_metadata` is produced by question evolution.
- `validation_result` is produced by `validate_evolved_question.py`.
  The script can optionally run `--validate-schema` to check pipeline records
  against these local schemas.
- `candidate_selection` is produced by `candidate_selection.py` on the selected
  main-chain record.
- `effect_analysis` is produced by `analyze_evolution_effect.py` after the
  standard scoring loop has produced a new scored record.

## Pass-Through Semantics

When `question_evolved` is `false`, downstream scripts must pass the record
through without regenerating answers, rubrics, or scores. Existing
`collect_answers.py`, `gen_rubric.py`, and `scoring.py` already follow this
top-level flag.

## Rubric Boundary

`expected_evaluation_focus` is allowed only inside
`meta_info.question_evolution_metadata`. It is metadata for question generation,
manual review, and later routing. It must not be copied into `gen_rubric.py`,
rubric prompts, score prompts, rubric items, weights, or judge calibration.

## Memory Files

The `memory/*.jsonl` files are append-only artifacts. `update_sample_state.py`
writes low-risk Stage 5 entries after effect analysis: effective operator
experience, failed operator experience, and invalid generation cases. Low
confidence hits must keep `needs_manual_review=true` and must not be treated as
strong success examples without review.

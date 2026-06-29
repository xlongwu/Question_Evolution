# Pipeline Field Contract

This directory defines the minimal JSONL contract used by the question
evolution stages. The schemas are intentionally small and permissive: they
stabilize field names, ownership, and runtime validation for the current
Stage 0-6 question evolution pipeline.

## Record Ownership

- `sample_id`, `index`, `round`, `prompt`, `meta_info`, `rubric`,
  `score_prompt`, `scoring_result`, `score_rate`, and `question_evolved` are
  shared pipeline fields.
- `sample_profile` is produced by `profile_samples.py`; `overscore_diagnosis`
  is produced by the same profiling step and consumed by
  `select_evolution_candidates.py`.
- Stage 2 tree-search decision fields are additive: `sample_profile` may include
  `boundary_axis_candidates`, `already_explored_axes`, and `next_best_axes`;
  `select_evolution_candidates.py` writes `tree_search_decision` with an
  action type and source-node hint.
- `operator_route` is produced by the router and consumed by
  `question_evolution.py` when `evolution_action` requires evolution. It keeps
  the legacy operator fields and may also include tree-search hints such as
  `branch_intent`, `source_node_type`, and `target_boundary_axis`.
- `candidate_group_id`, `candidate_id`, `candidate_operator`, and
  `candidate_generation` are intermediate fields produced only when
  `question_evolution.py --num-candidates` is greater than one. Stage 3 adds
  source-node metadata to `candidate_generation`, including `source_node_id`,
  `parent_node_id`, `branch_id`, `boundary_axis`, and `generation_action`.
- `evolution_state` is the cross-round state produced by
  `update_sample_state.py` and consumed by candidate selection, routing, and
  stop rules in later rounds.
- `meta_info.question_evolution_metadata` is produced by question evolution.
  It records the source prompt, generated node, branch metadata, operator, and
  validation retry details for auditability.
- `validation_result` is produced by `validate_evolved_question.py`.
  The script can optionally run `--validate-schema` to check pipeline records
  against these local schemas.
- Stage 4 boundary-selection fields are additive. `candidate_selection.py`
  keeps selecting a main candidate from validated candidates, and when boundary
  evidence is available it may also write `selected_into_mainline`,
  `selected_as_boundary_leaf`, `discard_as_duplicate`, and `dedup_signature`.
- `effect_analysis` is produced by `analyze_evolution_effect.py` after the
  standard scoring loop has produced a new scored record. Stage 4 adds
  `target_boundary_axis`, `boundary_axis_detected`,
  `is_new_boundary_for_sample`, `duplicate_boundary_for_sample`, and
  `dedup_signature` so Stage 5 can update discovered boundaries and frontier
  state without re-running score analysis.
- Stage 5 `update_sample_state.py` owns tree-search budget mutation,
  branch/sample stop conditions, `discovered_boundaries`, and the per-round
  `search_graph.jsonl` / `active_frontier.jsonl` artifacts. Earlier stages may
  suggest actions or annotate candidates, but they must not decrement budgets.
- Stage 6 does not introduce new runtime fields. It verifies the Stage 1-5
  contracts through focused pytest coverage, tiny mock pipeline artifacts,
  migration notes, and final acceptance documentation.

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

## Tree Search Extension Contract

The tree-search retrofit keeps the current single-chain pipeline as the
default behavior and adds permissive extension schemas for later stages:

- `evolution_state.schema.json` now includes optional tree-search state fields
  such as `search_root_id`, `current_node_id`, `branch_id`,
  `boundary_axis`, search depth, budgets, and discovered boundaries.
- `search_graph_node.schema.json` defines one node row in
  `search_graph.jsonl` artifacts written by `update_sample_state.py`.
- `active_frontier.schema.json` defines one pending expansion row in
  `active_frontier.jsonl` artifacts written by `update_sample_state.py`.
- `question_evolution.py` can optionally consume an `active_frontier.jsonl`
  through `--frontier-input`. The frontier rows are overlaid on matching
  pipeline records by `sample_id`, `index`, or `search_root_id`; generation
  still uses the base record for reference answers, rubrics, and candidate
  answers.

Use `search_state_contract.py` to normalize legacy single-chain records before
validating or constructing example graph/frontier rows. The detailed Chinese
contract is documented in `docs/树状搜索状态契约.md`.

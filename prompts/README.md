# Prompt Modules

This directory holds prompt builders used by the question evolution pipeline:

- `profile_prompt.py`: Stage 2 sample profiling prompt.
- `router_prompt.py`: Stage 3 operator-routing prompt reference. Production
  routing currently uses deterministic rules in `operator_router.py`.
- `operators/`: Stage 3 O1-O9 operator prompts used by `question_evolution.py`.

Operator prompts must stay narrow: each module describes one question shape and
must not grow into a new universal "make it harder" prompt.

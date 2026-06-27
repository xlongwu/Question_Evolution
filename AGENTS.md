# Repository Guidelines

## Project Structure & Module Organization
This repository contains a Python-based question evolution pipeline. Core scripts live at the repository root:

- `question_evolution.py`: evolves high-scoring questions into harder variants.
- `collect_answers.py`: collects reference/model answers for evolved prompts.
- `gen_rubric.py`: generates scoring rubrics from prompts and answers.
- `scoring.py`: scores candidate/model answers using an OpenAI-compatible API.
- `run_loop.sh`: orchestrates iterative scoring, evolution, answer collection, rubric generation, and final scoring.

Project notes and weekly reports are stored in the root report directory. The loop script expects input data under `data/` and writes experiment outputs under `experiments/YYYY-MM-DD/exp*/round_N/`; treat these generated outputs as artifacts, not source.

## Build, Test, and Development Commands
No package manifest is currently present. Create a local Python environment and install the observed runtime dependencies:

```bash
python -m venv .venv
pip install openai aiofiles tqdm
```

Run the full iterative workflow from a Bash-compatible shell:

```bash
bash run_loop.sh
```

Run individual stages when debugging:

```bash
python question_evolution.py --input experiments/.../scored.jsonl --output evolved.jsonl
python collect_answers.py --input evolved.jsonl --output with_answers.jsonl --samples 1
python gen_rubric.py --input with_answers.jsonl --output rubric.jsonl
python scoring.py --input rubric.jsonl --output scored.jsonl --concurrency 10
```

## Coding Style & Naming Conventions
Use Python 3, four-space indentation, and descriptive `snake_case` names for functions, variables, and CLI flags. Keep async API calls behind small helper classes or functions, following the existing `RotatingAPIClient` pattern. Prefer JSONL streaming for pipeline data. Existing comments and prompts include Chinese text; preserve the original language unless intentionally rewriting a prompt.

## Testing Guidelines
There is no test suite yet. For new behavior, add focused `pytest` tests under `tests/` with names like `test_scoring.py` or `test_question_evolution.py`. Mock LLM/API calls and validate JSONL inputs and outputs with small fixtures. Before submitting, run the changed stage on a tiny sample file and verify required fields are present.

## Commit & Pull Request Guidelines
This repository currently has no Git commit history, so use clear, conventional commit messages going forward, such as `feat: add rubric validation` or `fix: handle empty model responses`. Pull requests should describe the affected stage, list verification commands, note data-format changes, and mention required local API configuration.

## Security & Configuration Tips
Do not add new API keys or secrets to source files. Prefer environment variables, local shell exports, or ignored config files for credentials and base URLs. Redact private prompts, answers, and provider errors before sharing logs or outputs.

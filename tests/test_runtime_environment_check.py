import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from check_runtime_environment import build_report


def write_input_data(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "sample_id": "local-data-001",
                "index": 1,
                "prompt": "sample",
                "meta_info": {"references": ["reference"]},
                "rubric": [{"title": "core", "description": "core", "weight": 10}],
                "score_prompt": "Score <<<answer>>>",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_runtime_preflight_reports_ready_when_required_inputs_exist(tmp_path):
    write_input_data(tmp_path / "data" / "data.jsonl")
    env = {
        "OPENAI_API_KEY": "test-openai-key",
        "QWEN_BASE_URL": "http://127.0.0.1:18011/v1",
        "QWEN_MODEL": "hjl_Qwen3.6-27B",
    }

    report = build_report(tmp_path, env=env, bash_path="/bin/bash")

    assert report["checks"]["bash_available"] is True
    assert report["checks"]["input_data_ready"] is True
    assert report["checks"]["api_config_ready"] is True


def test_runtime_preflight_exposes_missing_bash_and_api(tmp_path):
    write_input_data(tmp_path / "data" / "data.jsonl")

    report = build_report(tmp_path, env={}, bash_path="")

    assert report["ready_for_real_stage06_e2e"] is False
    assert report["checks"]["bash_available"] is False
    assert report["checks"]["api_config_ready"] is False
    assert report["input_data"]["row_count"] == 1
    assert report["input_data"]["jsonl_parseable"] is True


def test_runtime_preflight_accepts_ignored_local_config(tmp_path):
    write_input_data(tmp_path / "data" / "data.jsonl")
    (tmp_path / "config.py").write_text(
        "\n".join(
            [
                "BASE_URL = 'https://example.test/v1'",
                "HIAPI_KEYS_BIG = ['local-strong-key']",
                "QWEN_BASE_URL = 'http://127.0.0.1:18011/v1'",
                "QWEN_MODEL = 'hjl_Qwen3.6-27B'",
            ]
        ),
        encoding="utf-8",
    )

    report = build_report(tmp_path, env={}, bash_path="/bin/bash")

    assert report["checks"]["api_config_ready"] is True
    assert all(report["api_config"].values())


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as ready_dir:
        test_runtime_preflight_reports_ready_when_required_inputs_exist(Path(ready_dir))
    with tempfile.TemporaryDirectory() as missing_dir:
        test_runtime_preflight_exposes_missing_bash_and_api(Path(missing_dir))
    with tempfile.TemporaryDirectory() as config_dir:
        test_runtime_preflight_accepts_ignored_local_config(Path(config_dir))
    print("runtime environment preflight checks passed")

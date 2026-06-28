import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from check_runtime_environment import REQUIRED_SEED_IDS, build_report


def write_seed(path: Path) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "index": int(sample_id),
                    "prompt": f"sample {sample_id}",
                    "meta_info": {"references": ["reference"]},
                    "rubric": [{"title": "core", "description": "core", "weight": 10}],
                    "score_prompt": "Score <<<待评答案>>",
                },
                ensure_ascii=False,
            )
            + "\n"
            for sample_id in sorted(REQUIRED_SEED_IDS)
        ),
        encoding="utf-8",
    )


def test_runtime_preflight_reports_ready_when_required_inputs_exist(tmp_path):
    seed = tmp_path / "admitted_seed_samples.jsonl"
    write_seed(seed)
    env = {
        "OPENAI_API_KEY": "test-openai-key",
        "QWEN_BASE_URL": "http://127.0.0.1:18011/v1",
        "QWEN_MODEL": "hjl_Qwen3.6-27B",
    }

    report = build_report(tmp_path, env=env, bash_path="/bin/bash")

    assert report["checks"]["bash_available"] is True
    assert report["checks"]["admitted_seed_ready"] is True
    assert report["checks"]["api_config_ready"] is True


def test_runtime_preflight_exposes_missing_bash_and_api(tmp_path):
    seed = tmp_path / "admitted_seed_samples.jsonl"
    write_seed(seed)

    report = build_report(tmp_path, env={}, bash_path="")

    assert report["ready_for_real_stage06_e2e"] is False
    assert report["checks"]["bash_available"] is False
    assert report["checks"]["api_config_ready"] is False
    assert report["admitted_seed"]["required_ids_present"] is True


def test_runtime_preflight_accepts_ignored_local_config(tmp_path):
    seed = tmp_path / "admitted_seed_samples.jsonl"
    write_seed(seed)
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

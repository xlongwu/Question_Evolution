import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from schema_validation import SchemaValidationError, validate_records_against_schema

ANSWER_PLACEHOLDER = "<<<待评答案>>"


def load_jsonl(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def test_schema_files_are_valid_json_objects():
    schema_paths = sorted((ROOT / "schemas").glob("*.schema.json"))
    assert schema_paths, "Stage 1 must provide schema files."

    for schema_path in schema_paths:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema.get("$schema")
        assert schema.get("title")
        assert schema.get("type") == "object"
        assert isinstance(schema.get("properties"), dict)


def test_stage01_fixture_records_keep_pipeline_contract():
    records = load_jsonl(ROOT / "tests" / "fixtures" / "stage01_contract.jsonl")
    assert len(records) >= 2
    schema_errors = validate_records_against_schema(records, ROOT / "schemas" / "pipeline_record.schema.json")
    assert schema_errors == []

    for record in records:
        assert isinstance(record.get("prompt"), str) and record["prompt"].strip()
        assert "sample_id" in record or "index" in record

        if "score_rate" in record:
            assert 0 <= record["score_rate"] <= 1

        scoring_result = record.get("scoring_result")
        if isinstance(scoring_result, dict) and scoring_result.get("total_possible"):
            expected_rate = scoring_result["total_awarded"] / scoring_result["total_possible"]
            assert abs(record["score_rate"] - expected_rate) < 1e-9

        if record.get("question_evolved") is False:
            assert isinstance(record.get("scoring_result"), dict)
            assert isinstance(record.get("rubric"), list)
            assert isinstance(record.get("score_prompt"), str)


def test_pipeline_schema_rejects_missing_required_prompt():
    records = [{"sample_id": "missing-prompt"}]
    errors = validate_records_against_schema(records, ROOT / "schemas" / "pipeline_record.schema.json")

    assert len(errors) == 1
    assert isinstance(errors[0], SchemaValidationError)
    assert "prompt" in str(errors[0])


def test_expected_evaluation_focus_stays_out_of_rubric_payloads():
    records = load_jsonl(ROOT / "tests" / "fixtures" / "stage01_contract.jsonl")

    for record in records:
        metadata = record.get("meta_info", {}).get("question_evolution_metadata", {})
        focus_items = metadata.get("expected_evaluation_focus", [])
        if not focus_items:
            continue

        assert isinstance(focus_items, list)
        rubric_payload = json.dumps(record.get("rubric", []), ensure_ascii=False)
        score_prompt = record.get("score_prompt", "")

        assert "expected_evaluation_focus" not in rubric_payload
        assert "expected_evaluation_focus" not in score_prompt
        for focus in focus_items:
            assert focus not in rubric_payload
            assert focus not in score_prompt


def test_memory_jsonl_files_are_empty_or_valid_jsonl():
    for path in (ROOT / "memory").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                json.loads(line)


if __name__ == "__main__":
    test_schema_files_are_valid_json_objects()
    test_stage01_fixture_records_keep_pipeline_contract()
    test_pipeline_schema_rejects_missing_required_prompt()
    test_expected_evaluation_focus_stays_out_of_rubric_payloads()
    test_memory_jsonl_files_are_empty_or_valid_jsonl()
    print("stage01 contract checks passed")

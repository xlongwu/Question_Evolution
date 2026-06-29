import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profile_samples import ProfileProcessor, attach_profile_result, parse_profile_response
from select_evolution_candidates import (
    EVOLVE_HIGH_SCORE_OVERSCORE,
    PASS_THROUGH_OR_SCORING_NOISE,
    RECONSTRUCT_LOW_SCORE_BOUNDARY,
    STOP_EVOLUTION,
    EXPAND_CURRENT_BRANCH,
    FORK_FROM_ROOT,
    STOP_SAMPLE,
    process_records,
)


def load_jsonl(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def profile_response(core, cause, target, worth=True):
    return json.dumps(
        {
            "sample_profile": {
                "core_capability": core,
                "claim_level": "可疑线索",
                "problem_shape": "候选项区分",
                "reasoning_granularity": "两步链条",
                "answer_mode_expected": "比较型",
                "easy_judgment_risk": "low",
                "external_knowledge_risk": "low",
                "complexity_expansion_risk": "medium",
            },
            "overscore_diagnosis": {
                "is_worth_evolving": worth,
                "candidate_overscore_cause": cause,
                "target_failure_mode": target,
                "why_high_score_is_suspicious": f"{cause}:{target}",
            },
        },
        ensure_ascii=False,
    )


class FakeProfileClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat_completions_create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake profile response left.")
        return self.responses.pop(0)


def test_profile_parser_rejects_operator_recommendation():
    bad_response = json.dumps(
        {
            "sample_profile": {
                "core_capability": "证据链补强",
                "claim_level": "可疑线索",
                "problem_shape": "候选项区分",
                "reasoning_granularity": "两步链条",
                "answer_mode_expected": "比较型",
                "easy_judgment_risk": "low",
                "external_knowledge_risk": "low",
                "complexity_expansion_risk": "medium",
            },
            "overscore_diagnosis": {
                "is_worth_evolving": True,
                "candidate_overscore_cause": "漏最小关键事实",
                "target_failure_mode": "选错最关键缺口",
                "why_high_score_is_suspicious": "缺少独立必要条件定位。",
            },
            "recommended_operator": "O1_gap_choice",
        },
        ensure_ascii=False,
    )

    try:
        parse_profile_response(bad_response)
    except ValueError as exc:
        assert "operator" in str(exc)
    else:
        raise AssertionError("operator recommendation should be rejected")


def test_profile_processor_and_selector_cover_stage02_actions():
    records = load_jsonl(ROOT / "tests" / "fixtures" / "stage02_scored.jsonl")
    fake_client = FakeProfileClient(
        [
            profile_response("证据链补强", "漏最小关键事实", "选错最关键缺口", True),
            profile_response("行为模式识别", "反常线索主线切换失败", "反常线索主线切换失败", True),
            profile_response("边界判断", "评分噪声", "格式失分", False),
            profile_response("边界判断", "基础边界判断过稳", "稳定满分", False),
        ]
    )

    processor = ProfileProcessor(fake_client, model="mock-profile-model", max_concurrent=1)
    profiled = asyncio.run(processor.process_records(records))

    assert len(profiled) == 4
    for record in profiled:
        assert isinstance(record.get("sample_profile"), dict)
        assert isinstance(record.get("overscore_diagnosis"), dict)
        profile = record["sample_profile"]
        assert 2 <= len(profile["boundary_axis_candidates"]) <= 4
        assert isinstance(profile["already_explored_axes"], list)
        assert isinstance(profile["next_best_axes"], list)
        assert "operator_route" not in record
        assert "recommended_operator" not in record

    profiled_by_id = {record["sample_id"]: record for record in profiled}
    assert profiled_by_id["stage02-high-overscore"]["sample_profile"]["next_best_axes"][0] == "最关键缺口识别"
    assert profiled_by_id["stage02-low-boundary"]["sample_profile"]["next_best_axes"][0] == "反常线索主线切换"
    assert profiled_by_id["stage02-scoring-noise"]["sample_profile"]["next_best_axes"] == []

    selected = process_records(profiled, high_score_threshold=0.8, low_score_threshold=0.6)
    actions = {record["sample_id"]: record["evolution_action"] for record in selected}
    decisions = {record["sample_id"]: record["tree_search_decision"] for record in selected}

    assert actions["stage02-high-overscore"] == EVOLVE_HIGH_SCORE_OVERSCORE
    assert actions["stage02-low-boundary"] == RECONSTRUCT_LOW_SCORE_BOUNDARY
    assert actions["stage02-scoring-noise"] == PASS_THROUGH_OR_SCORING_NOISE
    assert actions["stage02-stop"] == STOP_EVOLUTION

    assert decisions["stage02-high-overscore"]["action_type"] == EXPAND_CURRENT_BRANCH
    assert decisions["stage02-high-overscore"]["target_boundary_axis"] == "最关键缺口识别"
    assert decisions["stage02-low-boundary"]["action_type"] == EXPAND_CURRENT_BRANCH
    assert decisions["stage02-low-boundary"]["target_boundary_axis"] == "反常线索主线切换"
    assert decisions["stage02-scoring-noise"]["action_type"] == STOP_SAMPLE
    assert decisions["stage02-stop"]["action_type"] == STOP_SAMPLE

    for record in selected:
        assert record["evolution_action_reason"]
        assert record["tree_search_decision"]["source_node_type"] in {"current", "root", "parent"}
        assert "operator_used" not in record


def test_already_explored_axis_is_not_repeated_by_default():
    record = load_jsonl(ROOT / "tests" / "fixtures" / "stage02_scored.jsonl")[0]
    record["evolution_state"] = {
        "stop_status": "continue",
        "discovered_boundaries": [
            {
                "boundary_id": "b-001",
                "boundary_axis": "最关键缺口识别",
                "trigger_node_id": "node-1",
                "effect_label": "effective_boundary_probe",
            }
        ],
    }
    profile_result = parse_profile_response(
        profile_response("证据链补强", "漏最小关键事实", "选错最关键缺口", True)
    )
    profiled = attach_profile_result(
        record,
        profile_result,
        model="mock-profile-model",
        raw_response="{}",
    )

    profile = profiled["sample_profile"]
    assert "最关键缺口识别" in profile["already_explored_axes"]
    assert "最关键缺口识别" not in profile["next_best_axes"]
    assert profile["next_best_axes"][0] == "补强项升级判断"

    selected = process_records([profiled], high_score_threshold=0.8, low_score_threshold=0.6)[0]
    assert selected["tree_search_decision"]["action_type"] == FORK_FROM_ROOT
    assert selected["tree_search_decision"]["target_boundary_axis"] == "补强项升级判断"


if __name__ == "__main__":
    test_profile_parser_rejects_operator_recommendation()
    test_profile_processor_and_selector_cover_stage02_actions()
    test_already_explored_axis_is_not_repeated_by_default()
    print("stage02 profile and candidate selection checks passed")

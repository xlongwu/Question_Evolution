import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profile_samples import ProfileProcessor, parse_profile_response
from select_evolution_candidates import (
    EVOLVE_HIGH_SCORE_OVERSCORE,
    PASS_THROUGH_OR_SCORING_NOISE,
    RECONSTRUCT_LOW_SCORE_BOUNDARY,
    STOP_EVOLUTION,
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
        assert "operator_route" not in record
        assert "recommended_operator" not in record

    selected = process_records(profiled, high_score_threshold=0.8, low_score_threshold=0.6)
    actions = {record["sample_id"]: record["evolution_action"] for record in selected}

    assert actions["stage02-high-overscore"] == EVOLVE_HIGH_SCORE_OVERSCORE
    assert actions["stage02-low-boundary"] == RECONSTRUCT_LOW_SCORE_BOUNDARY
    assert actions["stage02-scoring-noise"] == PASS_THROUGH_OR_SCORING_NOISE
    assert actions["stage02-stop"] == STOP_EVOLUTION

    for record in selected:
        assert record["evolution_action_reason"]
        assert "operator_used" not in record


def test_scheduled_tree_frontier_bypasses_profile_stop_terms():
    record = {
        "sample_id": "stage02-tree-frontier",
        "prompt": "上一轮已经调度为继续扩展的树搜索节点。",
        "score_rate": 0.84,
        "branch_action": "expand_current_branch",
        "source_node_id": "sample_stage02_tree_frontier_root_b1_d1",
        "target_search_depth": 2,
        "search_depth": 1,
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
            "is_worth_evolving": False,
            "candidate_overscore_cause": "基础边界判断过稳",
            "target_failure_mode": "稳定满分",
            "why_high_score_is_suspicious": "diagnosis says the sample is stable or should stop.",
        },
        "evolution_state": {
            "search_root_id": "sample_stage02_tree_frontier_root",
            "current_node_id": "sample_stage02_tree_frontier_root_b1_d1",
            "branch_action": "expand_current_branch",
            "sample_stop_status": "continue_branch_search",
            "search_depth": 1,
            "target_search_depth": 2,
            "branch_budget_remaining": 1,
            "sample_budget_remaining": 3,
            "stop_status": "validated_high_score_sample",
        },
    }

    selected = process_records([record], high_score_threshold=0.8, low_score_threshold=0.6)

    assert selected[0]["evolution_action"] == EVOLVE_HIGH_SCORE_OVERSCORE
    assert "tree-search frontier" in selected[0]["evolution_action_reason"]

    fork_record = dict(record)
    fork_record.update(
        {
            "branch_action": "fork_from_root",
            "source_node_id": "sample_stage02_tree_frontier_root",
            "source_search_depth": 0,
            "search_depth": 0,
            "target_search_depth": 1,
        }
    )
    fork_state = dict(record["evolution_state"])
    fork_state.update(
        {
            "branch_action": "fork_from_root",
            "current_node_id": "sample_stage02_tree_frontier_root",
            "source_search_depth": 0,
            "search_depth": 0,
            "target_search_depth": 1,
            "branch_budget_remaining": 0,
            "max_search_depth": 2,
        }
    )
    fork_record["evolution_state"] = fork_state

    fork_selected = process_records([fork_record], high_score_threshold=0.8, low_score_threshold=0.6)

    assert fork_selected[0]["evolution_action"] == EVOLVE_HIGH_SCORE_OVERSCORE


if __name__ == "__main__":
    test_profile_parser_rejects_operator_recommendation()
    test_profile_processor_and_selector_cover_stage02_actions()
    test_scheduled_tree_frontier_bypasses_profile_stop_terms()
    print("stage02 profile and candidate selection checks passed")

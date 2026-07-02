import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from candidate_selection import select_candidates
from frontier_scheduler import build_active_frontier
from operator_router import route_records
from question_evolution import QuestionEvolutionProcessor
from validate_evolved_question import attach_validation_result


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeEvolutionClient:
    async def chat_completions_create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        axis = "子判断定位" if "O2_subclaim_localization" in prompt else "最小关键事实识别"
        return FakeResponse(
            json.dumps(
                {
                    "evolved_prompt": "请围绕一个子判断说明已支持层和仍缺的必要事实。",
                    "evolution_strategy": "parent sibling fork candidate",
                    "ability_axis": axis,
                    "expected_evaluation_focus": ["子判断", "必要事实"],
                    "complexity_budget": {
                        "main_axis_count": 1,
                        "new_facts_count": 1,
                        "output_tasks_count": 1,
                        "candidate_options_count": 0,
                        "counterfactual_count": 0,
                    },
                },
                ensure_ascii=False,
            )
        )


def make_parent_fork_record():
    return {
        "sample_id": "tree-parent",
        "prompt": "原题：判断结论是否成立。",
        "meta_info": {"references": ["参考答案。"]},
        "rubric": [{"title": "核心", "description": "说明必要事实。", "weight": 10}],
        "score_rate": 1.0,
        "scoring_result": {"candidate_answer": "泛泛说证据不足。", "total_awarded": 10, "total_possible": 10},
        "evolution_action": "evolve_high_score_overscore",
        "sample_profile": {
            "core_capability": "证据链补强",
            "claim_level": "子判断",
            "problem_shape": "候选项区分",
            "external_knowledge_risk": "low",
        },
        "overscore_diagnosis": {
            "is_worth_evolving": True,
            "candidate_overscore_cause": "漏最小关键事实",
            "target_failure_mode": "选错最关键缺口",
        },
        "branch_action": "fork_from_parent",
        "source_node_id": "sample_tree_parent_root",
        "parent_node_id": "sample_tree_parent_root",
        "boundary_axis": "子判断定位",
        "search_depth": 1,
        "evolution_state": {
            "round": 1,
            "stop_status": "continue_with_new_operator",
            "search_root_id": "sample_tree_parent_root",
            "current_node_id": "sample_tree_parent_root_b1_d1",
            "parent_node_id": "sample_tree_parent_root",
            "explored_axes": ["最小关键事实识别"],
            "recommended_next_axes": ["子判断定位"],
            "branch_count": 1,
        },
    }


def test_parent_sibling_fork_routes_axis_and_preserves_candidate_generation():
    routed = route_records([make_parent_fork_record()])[0]
    route = routed["operator_route"]
    assert route["branch_action"] == "fork_from_parent"
    assert route["boundary_axis"] == "子判断定位"
    assert route["primary_operator"] == "O2_subclaim_localization"

    processor = QuestionEvolutionProcessor(
        FakeEvolutionClient(),
        model="mock-evolution",
        max_concurrent=1,
        max_retries=0,
        num_candidates=1,
    )
    candidates = asyncio.run(processor.process_item_candidates(routed))
    generation = candidates[0]["candidate_generation"]
    assert generation["branch_action"] == "fork_from_parent"
    assert generation["source_node_id"] == "sample_tree_parent_root"
    assert generation["parent_node_id"] == "sample_tree_parent_root"
    assert generation["boundary_axis"] == "子判断定位"
    assert generation["search_depth"] == 1
    assert "o2_subclaim_localization" in generation["branch_id"]
    assert "o1_gap_choice" not in generation["branch_id"]

    validated = [attach_validation_result(candidate) for candidate in candidates]
    selected, invalid = select_candidates(validated)
    assert not invalid
    assert selected[0]["candidate_generation"]["branch_action"] == "fork_from_parent"
    assert selected[0]["candidate_selection"]["selected_into_mainline"] is True


def test_root_frontier_candidate_is_child_not_root_node():
    root_record = make_parent_fork_record()
    root_record["sample_id"] = "tree-root-depth"
    root_record["branch_action"] = "expand_current_branch"
    root_record.pop("source_node_id", None)
    root_record.pop("parent_node_id", None)
    root_record.pop("boundary_axis", None)
    root_record.pop("search_depth", None)
    root_record["evolution_state"] = {
        "round": 0,
        "stop_status": "continue",
    }
    active = build_active_frontier([root_record], max_branches=2, max_depth=2, max_boundaries=2, max_candidates_total=4)
    assert active[0]["source_node_id"] == active[0]["search_root_id"]
    assert active[0]["search_depth"] == 0
    assert active[0]["target_search_depth"] == 1

    routed = route_records(active)[0]
    processor = QuestionEvolutionProcessor(
        FakeEvolutionClient(),
        model="mock-evolution",
        max_concurrent=1,
        max_retries=0,
        num_candidates=1,
    )
    candidates = asyncio.run(processor.process_item_candidates(routed))
    generation = candidates[0]["candidate_generation"]
    assert generation["source_node_id"] == generation["search_root_id"]
    assert generation["search_depth"] == 1
    assert generation["candidate_node_id"] != generation["search_root_id"]


def test_single_candidate_process_item_writes_tree_generation_metadata():
    root_record = make_parent_fork_record()
    root_record["sample_id"] = "tree-single-candidate"
    root_record["branch_action"] = "expand_current_branch"
    root_record.pop("source_node_id", None)
    root_record.pop("parent_node_id", None)
    root_record.pop("boundary_axis", None)
    root_record.pop("search_depth", None)
    root_record["evolution_state"] = {
        "round": 0,
        "stop_status": "continue",
    }
    active = build_active_frontier([root_record], max_branches=2, max_depth=2, max_boundaries=2, max_candidates_total=4)
    routed = route_records(active)[0]
    processor = QuestionEvolutionProcessor(
        FakeEvolutionClient(),
        model="mock-evolution",
        max_concurrent=1,
        max_retries=0,
        num_candidates=1,
    )

    evolved = asyncio.run(processor.process_item(routed))
    generation = evolved["candidate_generation"]

    assert evolved["candidate_operator"] == "O1_gap_choice"
    assert generation["operator_id"] == "O1_gap_choice"
    assert generation["source_node_id"] == generation["search_root_id"]
    assert generation["search_depth"] == 1
    assert generation["candidate_node_id"] != generation["search_root_id"]
    assert "o1_gap_choice" in generation["branch_id"]
    assert "pending" not in generation["branch_id"]


def test_expand_current_branch_keeps_branch_and_sets_parent_to_source_node():
    record = make_parent_fork_record()
    record["sample_id"] = "tree-expand-depth"
    record["branch_action"] = "expand_current_branch"
    record["source_node_id"] = "sample_tree_expand_depth_root_b1_d1"
    record["parent_node_id"] = "sample_tree_expand_depth_root"
    record["boundary_axis"] = "最小关键事实识别"
    record["source_search_depth"] = 1
    record["target_search_depth"] = 2
    record["search_depth"] = 1
    record["branch_id"] = "branch_o1_gap_choice_axis_001"
    record["branch_index"] = 1
    record["evolution_state"] = {
        "round": 1,
        "stop_status": "continue_with_new_operator",
        "search_root_id": "sample_tree_expand_depth_root",
        "current_node_id": "sample_tree_expand_depth_root_b1_d1",
        "parent_node_id": "sample_tree_expand_depth_root",
        "branch_id": "branch_o1_gap_choice_axis_001",
        "branch_index": 1,
        "branch_count": 1,
        "search_depth": 1,
        "recommended_next_axes": ["最小关键事实识别", "子判断定位"],
    }

    routed = route_records([record])[0]
    processor = QuestionEvolutionProcessor(
        FakeEvolutionClient(),
        model="mock-evolution",
        max_concurrent=1,
        max_retries=0,
        num_candidates=1,
    )
    evolved = asyncio.run(processor.process_item(routed))
    generation = evolved["candidate_generation"]

    assert generation["branch_action"] == "expand_current_branch"
    assert generation["branch_index"] == 1
    assert generation["branch_id"] == "branch_o1_gap_choice_axis_001"
    assert generation["source_node_id"] == "sample_tree_expand_depth_root_b1_d1"
    assert generation["parent_node_id"] == "sample_tree_expand_depth_root_b1_d1"
    assert generation["candidate_node_id"] == "sample_tree_expand_depth_root_b1_d2"
    assert generation["search_depth"] == 2


if __name__ == "__main__":
    test_parent_sibling_fork_routes_axis_and_preserves_candidate_generation()
    test_root_frontier_candidate_is_child_not_root_node()
    test_single_candidate_process_item_writes_tree_generation_metadata()
    test_expand_current_branch_keeps_branch_and_sets_parent_to_source_node()
    print("tree stage04 parent fork checks passed")

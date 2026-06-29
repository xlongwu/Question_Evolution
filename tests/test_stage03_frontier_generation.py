import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from question_evolution import (
    QuestionEvolutionProcessor,
    expand_items_from_frontier,
    get_item_key,
)


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
    def __init__(self):
        self.calls = []

    async def chat_completions_create(self, **kwargs):
        self.calls.append(kwargs)
        content = json.dumps(
            {
                "evolved_prompt": (
                    "请比较题干中的 A 与 B 两个候选事实，判断哪一个才是支撑结论的"
                    "独立必要条件，并说明另一个为什么不能单独支撑结论。"
                ),
                "evolution_strategy": "围绕指定 source node 和目标能力轴生成单主轴候选。",
            },
            ensure_ascii=False,
        )
        return FakeResponse(content)


def base_record():
    return {
        "sample_id": "stage03-frontier",
        "index": 301,
        "round": 2,
        "prompt": "当前节点题：继续沿当前分支判断关键事实。",
        "meta_info": {
            "prompt_old": "根节点题：判断哪一条事实能支撑结论。",
            "parent_prompt": "父节点题：比较两个相邻层级的支撑强度。",
            "references": ["参考答案强调独立必要条件。"],
        },
        "rubric": [],
        "scoring_result": {
            "candidate_answer": "候选答案只是泛泛说还需要更多证据。",
            "total_awarded": 9,
            "total_possible": 10,
        },
        "score_rate": 0.9,
        "evolution_action": "evolve_high_score_overscore",
        "operator_route": {
            "primary_operator": "O1_gap_choice",
            "backup_operators": ["O2_subclaim_localization", "O4_near_level_ranking"],
            "avoid_operators": [],
            "branch_action": "expand_current_branch",
            "branch_intent": "expand_current_branch",
            "source_node_type": "current",
            "target_boundary_axis": "最关键缺口识别",
            "boundary_axis": "最关键缺口识别",
            "should_use_local_tree_search": True,
        },
        "evolution_state": {
            "search_root_id": "sample_stage03-frontier_root",
            "current_node_id": "sample_stage03-frontier_b1_d1",
            "parent_node_id": "sample_stage03-frontier_root",
            "branch_id": "branch_gap_existing",
            "boundary_axis": "最关键缺口识别",
            "search_depth": 1,
            "max_search_depth": 3,
            "branch_budget_remaining": 2,
            "sample_budget_remaining": 4,
            "discovered_boundaries": [],
        },
    }


def run_candidate(item, num_candidates=1):
    fake_client = FakeEvolutionClient()
    processor = QuestionEvolutionProcessor(
        fake_client,
        model="mock-evolution-model",
        max_concurrent=1,
        max_retries=0,
        num_candidates=num_candidates,
    )
    candidates = asyncio.run(processor.process_item_candidates(item))
    return candidates, fake_client


def test_expand_current_branch_writes_source_metadata():
    item = base_record()

    candidates, fake_client = run_candidate(item)
    candidate = candidates[0]
    metadata = candidate["meta_info"]["question_evolution_metadata"]

    assert len(fake_client.calls) == 1
    assert "当前节点题" in fake_client.calls[0]["messages"][0]["content"]
    assert metadata["generation_action"] == "expand_current_branch"
    assert metadata["source_node_id"] == "sample_stage03-frontier_b1_d1"
    assert metadata["parent_node_id"] == "sample_stage03-frontier_b1_d1"
    assert metadata["branch_id"] == "branch_gap_existing"
    assert metadata["boundary_axis"] == "最关键缺口识别"
    assert metadata["is_new_branch"] is False
    assert candidate["candidate_generation"]["generated_node_id"] == metadata["generated_node_id"]


def test_root_fork_uses_frontier_source_prompt_and_new_branch_metadata():
    item = base_record()
    item["_frontier_context"] = {
        "sample_id": "stage03-frontier",
        "search_root_id": "sample_stage03-frontier_root",
        "frontier_node_id": "frontier_root_axis_extra",
        "source_node_id": "sample_stage03-frontier_root",
        "source_node_type": "root",
        "source_prompt": "根节点题：判断哪一条事实能支撑结论。",
        "prompt_source": "meta_info.prompt_old",
        "action_type": "fork_from_root",
        "target_boundary_axis": "题干外补设识别",
        "search_depth": 0,
        "next_depth": 1,
        "max_search_depth": 3,
        "branch_budget_remaining": 2,
        "sample_budget_remaining": 3,
    }

    candidates, fake_client = run_candidate(item)
    candidate = candidates[0]
    metadata = candidate["meta_info"]["question_evolution_metadata"]

    assert "根节点题" in fake_client.calls[0]["messages"][0]["content"]
    assert metadata["frontier_node_id"] == "frontier_root_axis_extra"
    assert metadata["generation_action"] == "fork_from_root"
    assert metadata["source_node_type"] == "root"
    assert metadata["source_node_id"] == "sample_stage03-frontier_root"
    assert metadata["parent_node_id"] == "sample_stage03-frontier_root"
    assert metadata["boundary_axis"] == "题干外补设识别"
    assert metadata["is_new_branch"] is True
    assert metadata["branch_id"].startswith("branch_")
    assert metadata["branch_id"] != "branch_gap_existing"
    assert candidate["candidate_group_id"].endswith("::frontier_root_axis_extra")


def test_parent_fork_preserves_parent_source_and_explicit_branch_id():
    item = base_record()
    item["_frontier_context"] = {
        "frontier_node_id": "frontier_parent_axis_claim",
        "source_node_id": "sample_stage03-frontier_b1_d0",
        "source_node_type": "parent",
        "source_prompt": "父节点题：比较两个相邻层级的支撑强度。",
        "prompt_source": "parent_node",
        "action_type": "fork_from_parent",
        "branch_id": "branch_claim_sibling_01",
        "target_boundary_axis": "结论分层",
        "search_depth": 1,
        "next_depth": 1,
        "branch_budget_remaining": 1,
        "sample_budget_remaining": 2,
    }

    candidates, fake_client = run_candidate(item, num_candidates=3)
    candidate = candidates[0]
    metadata = candidate["meta_info"]["question_evolution_metadata"]

    assert len(candidates) == 1
    assert len(fake_client.calls) == 1
    assert "父节点题" in fake_client.calls[0]["messages"][0]["content"]
    assert metadata["generation_action"] == "fork_from_parent"
    assert metadata["source_node_type"] == "parent"
    assert metadata["source_node_id"] == "sample_stage03-frontier_b1_d0"
    assert metadata["branch_id"] == "branch_claim_sibling_01"
    assert metadata["search_depth"] == 1
    assert metadata["is_new_branch"] is True


def test_frontier_expansion_overlays_base_records_and_keeps_keys_distinct():
    base = base_record()
    frontiers = [
        {
            "sample_id": "stage03-frontier",
            "search_root_id": "sample_stage03-frontier_root",
            "frontier_node_id": "frontier_root_a",
            "source_node_id": "sample_stage03-frontier_root",
            "source_node_type": "root",
            "source_prompt": "根节点题 A。",
            "prompt_source": "meta_info.prompt_old",
            "action_type": "fork_from_root",
            "target_boundary_axis": "题干外补设识别",
            "search_depth": 0,
            "next_depth": 1,
            "max_search_depth": 3,
            "branch_budget_remaining": 2,
            "sample_budget_remaining": 4,
        },
        {
            "sample_id": "stage03-frontier",
            "search_root_id": "sample_stage03-frontier_root",
            "frontier_node_id": "frontier_parent_b",
            "source_node_id": "sample_stage03-frontier_b1_d0",
            "source_node_type": "parent",
            "source_prompt": "父节点题 B。",
            "prompt_source": "parent_node",
            "action_type": "fork_from_parent",
            "target_boundary_axis": "结论分层",
            "search_depth": 1,
            "next_depth": 1,
            "max_search_depth": 3,
            "branch_budget_remaining": 2,
            "sample_budget_remaining": 4,
        },
    ]

    expanded = expand_items_from_frontier([base], frontiers)
    keys = [get_item_key(item) for item in expanded]

    assert len(expanded) == 2
    assert keys[0] != keys[1]
    assert expanded[0]["tree_search_decision"]["action_type"] == "fork_from_root"
    assert expanded[1]["tree_search_decision"]["source_node_type"] == "parent"

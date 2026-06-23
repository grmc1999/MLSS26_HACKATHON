"""Unit tests for the deterministic LLM-plumbing code: agent routing and prompts.

These don't call any LLM (no API key needed, no network, no cost) -- they test the
plain Python logic that decides *which* agent gets consulted and that every agent
the orchestrator can route to actually has a system prompt defined.
"""
import pytest

from MLAgentBench.agents.orchestrator import ROUTING_KEYWORDS, route_to_agent
from MLAgentBench.agents.agent_specialized import AGENT_PROMPTS

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "goal,expected_agent",
    [
        ("increase dropout to reduce overfitting", "dl_expert"),
        ("try a different OOD confidence threshold", "robustness_expert"),
        ("add data augmentation to the CNN architecture", "cv_expert"),
        ("search for SOTA OOD detection papers", "research_literature"),
        ("will this checkpoint cause catastrophic forgetting", "continual_learning"),
        ("how should pneumonia and consolidation be distinguished radiographically", "medical_expert"),
    ],
)
def test_route_to_agent_matches_expected_role(goal, expected_agent):
    assert route_to_agent(goal) == expected_agent


def test_route_to_agent_falls_back_to_autoresearch_for_generic_goals():
    assert route_to_agent("improve the next experiment") == "autoresearch"


def test_every_routable_agent_has_a_system_prompt():
    """Contract: orchestrator must never route to an agent with no prompt defined."""
    for agent_name in ROUTING_KEYWORDS:
        assert agent_name in AGENT_PROMPTS, f"{agent_name} is routable but has no AGENT_PROMPTS entry"
        assert AGENT_PROMPTS[agent_name].strip(), f"{agent_name} has an empty system prompt"

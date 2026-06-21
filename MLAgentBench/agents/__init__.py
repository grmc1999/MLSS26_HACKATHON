"""Agents package for MLAgentBench."""

from .agent import Agent, SimpleActionAgent, ReasoningActionAgent
from .agent_research import ResearchAgent
from .agent_specialized import SpecializedResearchAgent, create_agent, AGENT_PROMPTS
from .continual_learning import ContinualLearningManager
from .orchestrator import ScientificAutoResearch, ExperimentManager, route_to_agent

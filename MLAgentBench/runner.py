""" 
This file is the entry point for MLAgentBench.
"""

import argparse
import sys
import os
from dotenv import load_dotenv

# Load .env from project root (parent of MLAgentBench package)
# override=True is critical: if OPENROUTER_API_KEY is already set as empty in the
# shell environment, load_dotenv won't replace it without this flag
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)
load_dotenv(override=True)  # also try CWD as fallback

from MLAgentBench.environment import Environment
from MLAgentBench.agents.agent import Agent, SimpleActionAgent, ReasoningActionAgent
from MLAgentBench.agents.agent_research import ResearchAgent
from MLAgentBench.agents.agent_specialized import SpecializedResearchAgent, create_agent, AGENT_PROMPTS
from MLAgentBench.agents.orchestrator import ScientificAutoResearch
from MLAgentBench.constants import *


def run(agent_cls, args, role=None):
    with Environment(args) as env:

        print("=====================================")
        research_problem, benchmark_folder_name = env.get_task_description()
        print("Benchmark folder name: ", benchmark_folder_name)
        print("Research problem: ", research_problem)
        print("Lower level actions enabled: ", [action.name for action in env.low_level_actions])
        print("High level actions enabled: ", [action.name for action in env.high_level_actions])
        print("Read only files: ", env.read_only_files, file=sys.stderr)
        print("=====================================")  

        if role and role in AGENT_PROMPTS:
            agent = SpecializedResearchAgent(args, env, role=role)
        else:
            agent = agent_cls(args, env)
        final_message = agent.run(env)
        print("=====================================")
        print("Final message: ", final_message)

    env.save("final")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="debug", help="task name")
    parser.add_argument("--log-dir", type=str, default="./logs", help="log dir")
    parser.add_argument("--work-dir", type=str, default="./workspace", help="work dir")
    parser.add_argument("--max-steps", type=int, default=1e8, help="should be deprecated. number of steps in environment, including retrieval actions")
    parser.add_argument("--max-time", type=int, default=5* 60 * 60, help="max time")
    parser.add_argument("--max-api-cost", type=int, default=10, help="max api cost in dollars")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--python", type=str, default="python", help="python command")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--resume", type=str, default=None, help="resume from a previous run")
    parser.add_argument("--resume-step", type=int, default=0, help="the step to resume from")

    # general agent configs
    parser.add_argument("--agent-type", type=str, default="ResearchAgent", help="agent type")
    parser.add_argument("--agent-role", type=str, default=None, help="specialized agent role (research_literature, autoresearch, cv_expert, dl_expert, llm_expert, medical_expert, continual_learning, robustness_expert)")
    parser.add_argument("--llm-name", type=str, default="nvidia/nemotron-3-super-120b-a12b:free", help="llm name")
    parser.add_argument("--fast-llm-name", type=str, default="nvidia/nemotron-3-nano-30b-a3b:free", help="llm name")
    parser.add_argument("--feedback-llm-name", type=str, default="nvidia/nemotron-3-super-120b-a12b:free", help="llm name")
    parser.add_argument("--feedback-llm-max-tokens", type=int, default=4000, help="llm max tokens")
    parser.add_argument("--edit-script-llm-name", type=str, default="nvidia/nemotron-3-super-120b-a12b:free", help="llm name")
    parser.add_argument("--edit-script-llm-max-tokens", type=int, default=4000, help="llm max tokens")
    parser.add_argument("--agent-max-steps", type=int, default=50, help="the real number of max iterations for agent")

    # research agent configs
    parser.add_argument("--actions-remove-from-prompt", type=str, nargs='+', default=[], help="actions to remove in addition to the default ones: Read File, Write File, Append File, Retrieval from Research Log, Append Summary to Research Log, Python REPL, Edit Script Segment (AI)")
    parser.add_argument("--actions-add-to-prompt", type=str, nargs='+', default=[], help="actions to add")
    parser.add_argument("--retrieval", action="store_true", help="enable retrieval")
    parser.add_argument("--valid-format-entires", type=str, nargs='+', default=None, help="valid format entries")
    parser.add_argument("--max-steps-in-context", type=int, default=3, help="max steps in context")
    parser.add_argument("--max-observation-steps-in-context", type=int, default=3, help="max observation steps in context")
    parser.add_argument("--max-retries", type=int, default=5, help="max retries")

    # langchain configs
    parser.add_argument("--langchain-agent", type=str, default="zero-shot-react-description", help="langchain agent")


    args = parser.parse_args()
    print(args, file=sys.stderr)
    if not args.retrieval or args.agent_type != "ResearchAgent":
        # should not use these actions when there is no retrieval
        args.actions_remove_from_prompt.extend(["Retrieval from Research Log", "Append Summary to Research Log", "Reflection"])
    os.environ["FAST_MODEL"] = args.fast_llm_name
    os.environ["LOG_DIR"] = args.log_dir
    os.environ["FEEDBACK_MODEL"] = args.feedback_llm_name
    os.environ["FEEDBACK_MAX_TOKENS"] = str(args.feedback_llm_max_tokens)
    agent_cls = getattr(sys.modules[__name__], args.agent_type)
    run(agent_cls, args, role=args.agent_role)
    

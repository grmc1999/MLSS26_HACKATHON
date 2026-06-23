"""Bridge: consult MLAgentBench specialized agents via OpenRouter LLM calls."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

from MLAgentBench.agents.agent_specialized import AGENT_PROMPTS, search_medical_literature, search_flu_context_rag
from MLAgentBench.LLM import complete_text, complete_text_fast
from MLAgentBench.schema import LLMError

LOG_DIR = "logs/agent_consults"
os.makedirs(LOG_DIR, exist_ok=True)


def consult(role, question, context="", use_rag=False, use_graph=False, k=3):
    prompt = AGENT_PROMPTS.get(role, "")
    rag_context = ""
    if use_rag:
        try:
            if use_graph and role == "time_series_expert":
                # Hybrid flu RAG: vector hits + FalkorDB relational context.
                result = search_flu_context_rag(question, k=k)
                rag_context = "\n\nRelevant literature (vector + graph context):\n" + result["combined_context"]
            else:
                results = search_medical_literature(question, k=k)
                rag_context = "\n\nRelevant literature:\n" + "\n".join(
                    f"- {r.get('title', '?')} (score: {r.get('score', 0):.3f})" for r in results
                )
        except Exception as e:
            rag_context = f"\n\n(RAG search unavailable: {e})"

    full_prompt = f"""You are a {role} expert in the MLSS26_HACKATHON project.
Your system prompt:
{prompt}

Current project context:
{context}
{rag_context}

Question: {question}

Provide a concise, actionable response (2-4 sentences). Focus only on the task."""

    log_file = f"{LOG_DIR}/{role}.log"
    try:
        response = complete_text_fast(prompt=full_prompt, log_file=log_file)
        return response.strip()
    except LLMError as e:
        return f"[LLM Error consulting {role}: {e}]"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=list(AGENT_PROMPTS.keys()))
    parser.add_argument("--question", required=True)
    parser.add_argument("--context", default="")
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--graph", action="store_true",
                         help="Use the FalkorDB knowledge graph hybrid RAG (time_series_expert only); implies --rag")
    args = parser.parse_args()
    result = consult(args.role, args.question, args.context, args.rag or args.graph, args.graph)
    print(result)

"""Build a FalkorDB knowledge graph from flu literature markdown files.

Complements (does not replace) scripts/build_flu_rag.py: that FAISS index
answers "what's semantically similar to this query"; this graph answers
relational questions ("which models were evaluated on which country with
which method, achieving which metric").

Uses the project's own complete_text() (MLAgentBench/LLM.py, already routes
through OpenRouter with retries) to extract entities/relationships as JSON,
and the official `falkordb` client directly for storage. No LangChain graph
wrappers: langchain_community.graphs.FalkorDBGraph / LLMGraphTransformer /
GraphCypherQAChain depend on the LLM supporting native function calling,
which free OpenRouter models do unreliably (invalid JSON, schema mismatches)
-- and both langchain-community and langchain-experimental are upstream
"sunset" packages besides. This mirrors the direct-client pattern used by
github.com/CodeGraphContext/CodeGraphContext for its FalkorDB backend.

Usage:
    source .venv/bin/activate
    python scripts/build_flu_graph.py
    # Free OpenRouter models are inconsistent at structured JSON output for
    # this; openai/gpt-4o-mini (paid, cheap) is far more reliable for this
    # one-time offline extraction:
    python scripts/build_flu_graph.py --model openai/gpt-4o-mini
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)
load_dotenv(override=True)  # also try CWD as fallback
sys.path.insert(0, _PROJECT_ROOT)

from falkordb import FalkorDB

from MLAgentBench.LLM import complete_text
from MLAgentBench.schema import LLMError

MD_DIR = Path(_PROJECT_ROOT) / "literature_flu_md"
LOG_DIR = Path(_PROJECT_ROOT) / "logs" / "build_flu_graph"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 512
OVERLAP = 64

# Schema kept narrow on purpose: open-ended extraction on a domain this small
# produces a noisy, inconsistent graph. Expand only if a real query needs a
# node/relationship type that's missing.
ALLOWED_NODES = ["Model", "Dataset", "Country", "Metric", "Method", "Paper"]
ALLOWED_RELATIONSHIPS = ["EVALUATED_ON", "ACHIEVES", "USES_METHOD", "CITES", "COMPARED_TO"]

EXTRACTION_PROMPT = """Extract entities and relationships from the text below, for a knowledge \
graph about flu/epidemiological forecasting research.

Allowed node types: {nodes}
Allowed relationship types: {rels}

Return ONLY valid JSON (no prose, no markdown fences) in this exact shape:
{{"nodes": [{{"name": "...", "type": "..."}}], "relationships": [{{"source": "...", "source_type": "...", "target": "...", "target_type": "...", "type": "..."}}]}}

Use the exact node/relationship type strings from the allowed lists above. Omit anything that \
doesn't clearly fit. If nothing fits, return {{"nodes": [], "relationships": []}}.

Text:
{text}
"""


def chunk_file(path: Path) -> list[str]:
    """Same chunking as scripts/build_flu_rag.py, kept independent on purpose
    so this script can run standalone without importing the FAISS build."""
    text = path.read_text(encoding="utf-8", errors="replace")
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_json(raw: str) -> dict:
    """LLMs (especially free/weak ones) sometimes wrap JSON in prose or
    markdown fences despite instructions. Try a clean parse first, then fall
    back to pulling out the first {...} block."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in response: {raw[:200]!r}")
    return json.loads(match.group(0))


def extract_graph(text: str, model: str, log_file: str) -> dict:
    prompt = EXTRACTION_PROMPT.format(
        nodes=", ".join(ALLOWED_NODES), rels=", ".join(ALLOWED_RELATIONSHIPS), text=text,
    )
    raw = complete_text(prompt=prompt, log_file=log_file, model=model, max_tokens_to_sample=1500)
    data = extract_json(raw)
    # Validate against the whitelist here -- node/relationship "type" gets
    # interpolated into Cypher as a label below (Cypher can't parameterize
    # labels), so only ever-allowed-listed strings reach that string format.
    nodes = [n for n in data.get("nodes", []) if n.get("type") in ALLOWED_NODES and n.get("name")]
    rels = [
        r for r in data.get("relationships", [])
        if r.get("type") in ALLOWED_RELATIONSHIPS
        and r.get("source_type") in ALLOWED_NODES
        and r.get("target_type") in ALLOWED_NODES
        and r.get("source") and r.get("target")
    ]
    return {"nodes": nodes, "relationships": rels}


def load_into_graph(graph, data: dict, paper: str):
    for n in data["nodes"]:
        graph.query(f"MERGE (n:{n['type']} {{name: $name}})", {"name": n["name"]})
    for r in data["relationships"]:
        graph.query(
            f"MERGE (a:{r['source_type']} {{name: $source}}) "
            f"MERGE (b:{r['target_type']} {{name: $target}}) "
            f"MERGE (a)-[rel:{r['type']}]->(b) "
            f"SET rel.paper = $paper",
            {"source": r["source"], "target": r["target"], "paper": paper},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.getenv("GRAPH_BUILD_MODEL", "openai/gpt-4o-mini"),
        help="Model used for entity/relationship extraction (routed through OpenRouter via "
             "MLAgentBench.LLM.complete_text). Runs once offline, not per-query. Free models "
             "vary a lot in how reliably they return clean JSON for this.",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip clearing the graph before rebuilding. Default clears it -- this corpus is "
             "small enough that incremental updates aren't supported yet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N papers (alphabetical). Useful to smoke-test before "
             "running the full corpus.",
    )
    args = parser.parse_args()

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    graph_name = os.getenv("FALKORDB_GRAPH_NAME", "flu_literature")

    db = FalkorDB(host=host, port=port)
    graph = db.select_graph(graph_name)

    if not args.no_reset:
        print(f"Resetting graph '{graph_name}'...")
        graph.query("MATCH (n) DETACH DELETE n")

    files = sorted(MD_DIR.glob("*.md"))
    if args.limit:
        files = files[:args.limit]
    print(f"Found {len(files)} papers in {MD_DIR}" + (f" (limited to {args.limit})" if args.limit else ""), flush=True)

    total_nodes, total_rels, failed_chunks = 0, 0, 0
    for f in files:
        chunks = chunk_file(f)
        print(f"  {f.stem}: {len(chunks)} chunks", flush=True)
        for i, chunk in enumerate(chunks):
            log_file = str(LOG_DIR / f"{f.stem}.log")
            try:
                data = extract_graph(chunk, args.model, log_file)
            except (LLMError, ValueError, json.JSONDecodeError) as e:
                failed_chunks += 1
                print(f"    chunk {i}: extraction failed -- {e}", flush=True)
                continue
            load_into_graph(graph, data, f.stem)
            total_nodes += len(data["nodes"])
            total_rels += len(data["relationships"])
            print(f"    chunk {i}: +{len(data['nodes'])} nodes, +{len(data['relationships'])} relationships", flush=True)

    print(f"\nDone. Extracted ~{total_nodes} nodes, ~{total_rels} relationships "
          f"into graph '{graph_name}' ({failed_chunks} chunks failed extraction).", flush=True)
    print(f"Verify with: redis-cli -p {port} GRAPH.QUERY {graph_name} \"MATCH (n) RETURN count(n)\"", flush=True)


if __name__ == "__main__":
    main()

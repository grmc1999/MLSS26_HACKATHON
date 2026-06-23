"""Build a FalkorDB knowledge graph from flu literature markdown files.

Complements (does not replace) scripts/build_flu_rag.py: that FAISS index
answers "what's semantically similar to this query"; this graph answers
relational questions ("which models were evaluated on which country with
which method, achieving which metric").

Usage:
    source .venv/bin/activate
    python scripts/build_flu_graph.py
    # Graph construction runs once, offline -- a stronger (paid) model
    # usually gives cleaner entity/relationship extraction than a free one:
    python scripts/build_flu_graph.py --model openai/gpt-4o
"""
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)
load_dotenv(override=True)  # also try CWD as fallback

from langchain_community.graphs import FalkorDBGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

MD_DIR = Path(_PROJECT_ROOT) / "literature_flu_md"
CHUNK_SIZE = 512
OVERLAP = 64

# Schema kept narrow on purpose: open-ended extraction on a domain this small
# produces a noisy, inconsistent graph. Expand only if a real query needs a
# node/relationship type that's missing.
ALLOWED_NODES = ["Model", "Dataset", "Country", "Metric", "Method", "Paper"]
ALLOWED_RELATIONSHIPS = ["EVALUATED_ON", "ACHIEVES", "USES_METHOD", "CITES", "COMPARED_TO"]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.getenv("FAST_MODEL", "openai/gpt-oss-20b:free"),
        help="OpenRouter model id used for entity/relationship extraction "
             "(runs once offline, not per-query).",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip clearing the graph before rebuilding. Default behavior "
             "clears it -- this corpus is small enough that incremental "
             "updates aren't supported yet.",
    )
    args = parser.parse_args()

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=args.model,
        temperature=0,
    )

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        node_properties=["name"],
    )

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    graph_name = os.getenv("FALKORDB_GRAPH_NAME", "flu_literature")

    graph = FalkorDBGraph(database=graph_name, host=host, port=port)

    if not args.no_reset:
        print(f"Resetting graph '{graph_name}'...")
        graph.query("MATCH (n) DETACH DELETE n")

    files = sorted(MD_DIR.glob("*.md"))
    print(f"Found {len(files)} papers in {MD_DIR}")

    total_nodes, total_rels = 0, 0
    for f in files:
        chunks = chunk_file(f)
        print(f"  {f.stem}: {len(chunks)} chunks")
        documents = [
            Document(page_content=chunk, metadata={"paper": f.stem, "chunk": i})
            for i, chunk in enumerate(chunks)
        ]
        if not documents:
            continue
        try:
            graph_documents = transformer.convert_to_graph_documents(documents)
        except Exception as e:
            print(f"    Error extracting graph from {f.stem}: {e}")
            continue
        graph.add_graph_documents(graph_documents, include_source=True)
        total_nodes += sum(len(gd.nodes) for gd in graph_documents)
        total_rels += sum(len(gd.relationships) for gd in graph_documents)

    print(f"\nDone. Extracted ~{total_nodes} nodes, ~{total_rels} relationships into graph '{graph_name}'.")
    print(f"Verify with: redis-cli -p {port} GRAPH.QUERY {graph_name} \"MATCH (n) RETURN count(n)\"")


if __name__ == "__main__":
    main()

"""Build a FalkorDB knowledge graph from flu literature markdown files.

Complements (does not replace) scripts/build_flu_rag.py: that FAISS index
answers "what's semantically similar to this query"; this graph answers
relational questions ("which models were evaluated on which country with
which method, achieving which metric").

Uses the local Qwen2.5-Coder-7B model for entity/relationship extraction
and the official falkordb client directly for storage.

Usage:
    source .venv/bin/activate
    python scripts/build_flu_graph.py
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from falkordb import FalkorDB
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)
load_dotenv(override=True)

MD_DIR = Path(_PROJECT_ROOT) / "literature_flu_md"
CHUNK_SIZE = 512
OVERLAP = 64
LOCAL_MODEL = str(Path(_PROJECT_ROOT) / "models/Qwen2.5-Coder-7B-Instruct")

ALLOWED_NODES = ["Model", "Dataset", "Country", "Metric", "Method", "Paper"]
ALLOWED_RELATIONSHIPS = ["EVALUATED_ON", "ACHIEVES", "USES_METHOD", "CITES", "COMPARED_TO"]

EXTRACTION_PROMPT = """<instructions>
Extract entities and relationships from the text below, for a knowledge graph about flu/epidemiological forecasting research.

Entity types: Model, Dataset, Country, Metric, Method, Paper
Relationship types: EVALUATED_ON, ACHIEVES, USES_METHOD, CITES, COMPARED_TO

Return JSON with "entities" array (each has "name" and "type") and "relationships" array (each has "source", "target", "type").
Output ONLY the JSON object, nothing else.
</instructions>

<text>
{chunk}
</text>

<json>"""


def chunk_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_json(response: str) -> dict | None:
    m = re.search(r'\{.*\}', response, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def load_into_graph(graph, data: dict, paper: str):
    for ent in data.get("entities", []):
        name = ent.get("name", "").strip()
        etype = ent.get("type", "")
        if not name or etype not in ALLOWED_NODES:
            continue
        graph.query(f"MERGE (n:{etype} {{name: $name}})", {"name": name})
    for rel in data.get("relationships", []):
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        rtype = rel.get("type", "")
        if not src or not tgt or rtype not in ALLOWED_RELATIONSHIPS:
            continue
        graph.query(
            f"MERGE (a {{name: $src}}) "
            f"MERGE (b {{name: $tgt}}) "
            f"MERGE (a)-[r:{rtype}]->(b) "
            f"SET r.paper = $paper",
            {"src": src, "tgt": tgt, "paper": paper},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reset", action="store_true", help="Skip clearing the graph.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N papers.")
    args = parser.parse_args()

    print(f"Loading model from {LOCAL_MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL)
    model.eval()

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
    print(f"Found {len(files)} papers in {MD_DIR}" + (f" (limited to {args.limit})" if args.limit else ""))

    total_nodes, total_rels = 0, 0
    for f in files:
        chunks = chunk_file(f)
        print(f"  {f.stem}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            prompt = EXTRACTION_PROMPT.format(chunk=chunk[:2000])
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
            response = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            data = extract_json(response)
            if data:
                load_into_graph(graph, data, f.stem)
                total_nodes += len(data.get("entities", []))
                total_rels += len(data.get("relationships", []))
                print(f"    chunk {i}: +{len(data['entities'])} entities, +{len(data['relationships'])} rels")
            else:
                print(f"    chunk {i}: extraction failed (no JSON)")
            time.sleep(0.05)

    print(f"\nDone. ~{total_nodes} entities, ~{total_rels} relationships into graph '{graph_name}'.")
    r = graph.query("MATCH (n) RETURN count(n)")
    print(f"  Nodes in DB: {r.result_set[0][0] if r.result_set else '?'}")


if __name__ == "__main__":
    main()

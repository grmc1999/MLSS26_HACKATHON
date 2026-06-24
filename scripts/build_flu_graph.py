"""Build a FalkorDB knowledge graph from flu literature markdown files.

Complements (does not replace) scripts/build_flu_rag.py: that FAISS index
answers "what's semantically similar to this query"; this graph answers
relational questions ("which models were evaluated on which country with
which method, achieving which metric").

Uses the local Qwen2.5-Coder-7B model for entity/relationship extraction
and the official falkordb client directly for storage.

Entity dedup reuses the same loaded model (a second generate() call with a
different prompt) rather than adding a separate API-backed model: embedding
similarity alone can't tell "same entity, different phrasing" apart from
"different entity, similar phrasing" for short names (measured: "GRU" vs
"Gated Recurrent Unit" ~0.19 cosine similarity, while "WMT 2014
English-to-German" vs "...English-to-French", different entities, ~0.83 --
the ranges overlap, no fixed threshold separates them). So embeddings only
shortlist *candidates*; the loaded model makes the actual same/different
judgment per candidate.

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

import numpy as np
import torch
from dotenv import load_dotenv
from falkordb import FalkorDB
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)
load_dotenv(override=True)

MD_DIR = Path(_PROJECT_ROOT) / "literature_flu_md"
LOG_DIR = Path(_PROJECT_ROOT) / "logs" / "build_flu_graph"
LOG_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_SIZE = 512
OVERLAP = 64
LOCAL_MODEL = str(Path(_PROJECT_ROOT) / "models/Qwen2.5-Coder-7B-Instruct")

# Same embedding model as scripts/build_flu_rag.py -- reused here to shortlist
# *candidate* duplicate entity names (see module docstring for why embeddings
# alone can't make the final call), and at query time
# (agent_specialized.py::_query_flu_graph) to match queries semantically
# instead of by literal substring.
#
# Distance is Euclidean, not cosine -- mathematically equivalent ranking for
# L2-normalized vectors (euclidean^2 = 2 - 2*cosine), so this is a unit
# change only. Holds as long as embeddings stay normalized
# (normalize_embeddings=True below); revisit if that ever changes.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEDUP_CANDIDATE_MAX_DIST = 0.894  # equivalent to cosine >= 0.6 for normalized vectors
DEDUP_MAX_CANDIDATES = 3  # cap dedup judgment calls per new entity

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

DEDUP_PROMPT = """<instructions>
Are these two names referring to the same real-world {node_type} in the context of flu/epidemiological forecasting research?

A: {name_a}
B: {name_b}

Return JSON: {{"same": true_or_false, "canonical_name": "..."}}
If they are the same, canonical_name should be whichever of the two is clearer or more complete.
If they are not the same, canonical_name should just repeat B.
Output ONLY the JSON object, nothing else.
</instructions>

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


def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 512, temperature: float = 0.1) -> str:
    """Shared generate() call -- used for both extraction and dedup judgment
    so the model/tokenizer are only loaded once in main()."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature)
    return tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)


def _llm_same_entity(name_a: str, name_b: str, node_type: str, model, tokenizer) -> str | None:
    """Ask the local model whether name_a and name_b are the same real-world
    entity. Returns the canonical name if they are, None otherwise (fail-safe:
    an unparseable response can only cost a duplicate node, never silently
    merge unrelated entities)."""
    prompt = DEDUP_PROMPT.format(node_type=node_type, name_a=name_a, name_b=name_b)
    response = generate_text(model, tokenizer, prompt, max_new_tokens=100)
    data = extract_json(response)
    if data and data.get("same") and data.get("canonical_name"):
        return data["canonical_name"]
    return None


def get_or_create_node(
    graph, name: str, node_type: str | None, embed_model, cache: dict, model, tokenizer, aliases: dict,
) -> str:
    """Resolve `name` to a canonical node. `cache` is a {node_type: [(name,
    embedding)]} dict of canonical entities that persists across the whole
    run, so dedup works across chunks and papers, not just within one.
    `aliases` is a {node_type: {raw_name: canonical_name}} memo so a repeated
    alternate phrasing only costs one judgment call, not one per occurrence.

    If `node_type` is unknown (a relationship endpoint not also listed under
    "entities" for this or any prior chunk), dedup/embedding is skipped and a
    bare untyped node is created -- same fallback upstream's original
    load_into_graph() used.
    """
    if node_type is None:
        graph.query("MERGE (n {name: $name})", {"name": name})
        return name

    bucket = cache.setdefault(node_type, [])
    alias_bucket = aliases.setdefault(node_type, {})

    if name in alias_bucket:
        return alias_bucket[name]
    for existing_name, _ in bucket:
        if existing_name == name:
            return name  # exact repeat of a canonical name -- nothing to resolve

    embedding = embed_model.encode(name, normalize_embeddings=True)
    candidates = sorted(
        ((n, float(np.linalg.norm(embedding - e))) for n, e in bucket),
        key=lambda pair: pair[1],  # ascending -- smaller distance = more similar
    )
    for existing_name, dist in candidates[:DEDUP_MAX_CANDIDATES]:
        if dist > DEDUP_CANDIDATE_MAX_DIST:
            break  # sorted ascending -- no remaining candidate can pass either
        canonical = _llm_same_entity(name, existing_name, node_type, model, tokenizer)
        if canonical:
            alias_bucket[name] = canonical
            return canonical

    graph.query(
        f"MERGE (n:{node_type} {{name: $name}}) SET n.embedding = $embedding",
        {"name": name, "embedding": embedding.tolist()},
    )
    bucket.append((name, embedding))
    return name


def load_into_graph(graph, data: dict, paper: str, embed_model, cache: dict, model, tokenizer, aliases: dict):
    # Map of name -> type for entities seen in *this* chunk, so relationship
    # endpoints can be typed when they reference an entity also listed here.
    chunk_types: dict[str, str] = {}
    for ent in data.get("entities", []):
        name = ent.get("name", "").strip()
        etype = ent.get("type", "")
        if not name or etype not in ALLOWED_NODES:
            continue
        chunk_types[name] = etype
        get_or_create_node(graph, name, etype, embed_model, cache, model, tokenizer, aliases)

    for rel in data.get("relationships", []):
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        rtype = rel.get("type", "")
        if not src or not tgt or rtype not in ALLOWED_RELATIONSHIPS:
            continue
        # Relationships don't carry their own type info -- resolve it from
        # this chunk's entities first, then any already-known canonical node
        # of the same name; fall back to an untyped node otherwise (matches
        # the pre-dedup behavior for endpoints with no type info anywhere).
        src_type = chunk_types.get(src) or _lookup_known_type(cache, src)
        tgt_type = chunk_types.get(tgt) or _lookup_known_type(cache, tgt)
        src_name = get_or_create_node(graph, src, src_type, embed_model, cache, model, tokenizer, aliases)
        tgt_name = get_or_create_node(graph, tgt, tgt_type, embed_model, cache, model, tokenizer, aliases)
        graph.query(
            f"MERGE (a {{name: $src}}) "
            f"MERGE (b {{name: $tgt}}) "
            f"MERGE (a)-[r:{rtype}]->(b) "
            f"SET r.paper = $paper",
            {"src": src_name, "tgt": tgt_name, "paper": paper},
        )


def _lookup_known_type(cache: dict, name: str) -> str | None:
    for node_type, bucket in cache.items():
        if any(existing_name == name for existing_name, _ in bucket):
            return node_type
    return None


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

    print(f"Loading embedding model ({EMBED_MODEL_NAME})...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    cache: dict[str, list[tuple[str, np.ndarray]]] = {}
    aliases: dict[str, dict[str, str]] = {}

    host = os.getenv("FALKORDB_HOST", "localhost")
    port = int(os.getenv("FALKORDB_PORT", "6379"))
    graph_name = os.getenv("FALKORDB_GRAPH_NAME", "flu_literature")

    db = FalkorDB(host=host, port=port)
    graph = db.select_graph(graph_name)

    if not args.no_reset:
        print(f"Resetting graph '{graph_name}'...")
        graph.query("MATCH (n) DETACH DELETE n")
    else:
        # Seed the dedup cache from existing nodes so appending more papers
        # still dedupes against what's already in the graph, not just within
        # this run.
        for node_type in ALLOWED_NODES:
            result = graph.query(f"MATCH (n:{node_type}) WHERE n.embedding IS NOT NULL RETURN n.name, n.embedding")
            cache[node_type] = [(name, np.array(emb)) for name, emb in result.result_set]
        seeded = sum(len(v) for v in cache.values())
        if seeded:
            print(f"Seeded dedup cache with {seeded} existing nodes (--no-reset).")

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
            response = generate_text(model, tokenizer, prompt, max_new_tokens=512)
            data = extract_json(response)
            if data:
                load_into_graph(graph, data, f.stem, embed_model, cache, model, tokenizer, aliases)
                total_nodes += len(data.get("entities", []))
                total_rels += len(data.get("relationships", []))
                print(f"    chunk {i}: +{len(data['entities'])} entities, +{len(data['relationships'])} rels")
            else:
                print(f"    chunk {i}: extraction failed (no JSON)")
            time.sleep(0.05)

    unique_nodes = sum(len(v) for v in cache.values())
    print(f"\nDone. ~{total_nodes} entity mentions, ~{total_rels} relationship mentions "
          f"({unique_nodes} unique typed nodes after dedup) into graph '{graph_name}'.")
    r = graph.query("MATCH (n) RETURN count(n)")
    print(f"  Nodes in DB: {r.result_set[0][0] if r.result_set else '?'}")


if __name__ == "__main__":
    main()

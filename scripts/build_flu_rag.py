"""Build text-based FAISS index from flu literature markdown files."""
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MD_DIR = Path("literature_flu_md")
OUTPUT_DIR = Path("index_output_flu")
CHUNK_SIZE = 512
OVERLAP = 64

OUTPUT_DIR.mkdir(exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
dim = model.get_sentence_embedding_dimension()
print(f"Model dim: {dim}")

all_chunks, all_meta = [], []
files = sorted(MD_DIR.glob("*.md"))
print(f"Found {len(files)} files")

for f in files:
    text = f.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        continue
    words = text.split()
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        all_chunks.append(chunk)
        all_meta.append({"file": f.stem, "chunk": len(all_chunks) - 1})

print(f"Total chunks: {len(all_chunks)}")

embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=64)
embeddings = embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)
print(f"Embeddings shape: {embeddings.shape}")

nlist = min(int(np.sqrt(len(embeddings))), 50)
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, max(nlist, 1), faiss.METRIC_INNER_PRODUCT)
index.train(embeddings)
index.add(embeddings)

faiss.write_index(index, str(OUTPUT_DIR / "index.faiss"))
with open(OUTPUT_DIR / "articles.json", "w") as f:
    json.dump(all_meta, f, indent=2)

print(f"Index: {OUTPUT_DIR / 'index.faiss'} ({index.ntotal} vectors)")
print(f"Articles: {OUTPUT_DIR / 'articles.json'} ({len(all_meta)} chunks)")

test = model.encode(["influenza forecasting with LSTM"])
test = test / np.linalg.norm(test)
dists, idxs = index.search(test.astype(np.float32), 3)
print("\nTest query 'influenza forecasting with LSTM':")
for dist, idx in zip(dists[0], idxs[0]):
    print(f"  score={dist:.3f}  {all_meta[idx]}")

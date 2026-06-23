"""Build FAISS index from rendered paper tiles using Qwen3-VL-Embedding-2B.

Usage:
    source .venv/bin/activate
    python scripts/build_rag_index.py
"""
import torch
import numpy as np
import json, sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

MODEL_PATH = "models/Qwen3-VL-Embedding-2B"
TILES_DIR = "index_output/tiles"
OUTPUT_DIR = "index_output"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Qwen3-VL-Embedding-2B...")
    model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

    tiles_path = Path(TILES_DIR)
    article_dirs = sorted([d for d in tiles_path.iterdir() if d.is_dir() and d.name.endswith(".png.tiles")])
    print(f"Found {len(article_dirs)} articles")

    # Collect all tile images
    all_tiles = []
    article_map = {}
    for i, ad in enumerate(article_dirs):
        tiles = sorted(ad.glob("tile_*.jpg"))
        for t in tiles:
            all_tiles.append((i, str(t)))
        article_map[str(i)] = ad.name.replace(".png.tiles", "")

    print(f"Total tiles: {len(all_tiles)}")
    torch.cuda.empty_cache()

    # Embed in batches
    all_embs, all_ids = [], []
    batch_size = 32
    for start in tqdm(range(0, len(all_tiles), batch_size), desc="Embedding"):
        batch = all_tiles[start:start+batch_size]
        images, ids = [], []
        for aid, tpath in batch:
            try:
                img = Image.open(tpath).convert("RGB")
                images.append(img)
                ids.append(int(aid))
            except Exception as e:
                print(f"  Error loading {tpath}: {e}")
        if not images:
            continue
        try:
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "What medical content is shown in this image?"}
            ]}] for img in images]
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.last_hidden_state[:, -1].cpu().numpy()
            embs = last_hidden / np.linalg.norm(last_hidden, axis=-1, keepdims=True)
            all_embs.append(embs)
            all_ids.extend(ids)
        except Exception as e:
            print(f"  Error in batch: {e}")
            torch.cuda.empty_cache()
            continue

    if not all_embs:
        print("No embeddings generated!")
        return

    embeddings = np.vstack(all_embs).astype(np.float32)
    ids = np.array(all_ids, dtype=np.int64)
    print(f"Generated {len(embeddings)} embeddings, dim={embeddings.shape[1]}")

    # Save articles.json
    articles = [{"id": k, "title": v} for k, v in article_map.items()]
    out = Path(OUTPUT_DIR)
    with open(out / "articles.json", "w") as f:
        json.dump(articles, f, indent=2)

    # Build FAISS index
    import faiss
    dim = embeddings.shape[1]
    nlist = min(int(np.sqrt(len(embeddings))), len(articles))
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, max(nlist, 1), faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add_with_ids(embeddings, ids)

    index_path = out / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"FAISS index: {index_path} ({index.ntotal} vectors, {nlist} centroids)")


if __name__ == "__main__":
    main()

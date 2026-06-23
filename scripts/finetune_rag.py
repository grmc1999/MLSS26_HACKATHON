"""Fine-tune Qwen3-VL-Embedding-2B on medical literature tiles via contrastive learning.

Usage:
    source .venv/bin/activate
    python scripts/finetune_rag.py --epochs 5 --batch-size 4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model
from pathlib import Path
import argparse, json, random


class PaperTileDataset(Dataset):
    """Each tile is paired with its paper title + keywords as query text."""

    def __init__(self, pixels_dir: str, processor, split="train", val_ratio=0.1):
        self.processor = processor
        self.samples = []

        pixels_path = Path(pixels_dir)
        for paper_dir in sorted(pixels_path.iterdir()):
            if not paper_dir.is_dir():
                continue
            tiles = sorted(paper_dir.glob("page_*.png"))
            if not tiles:
                continue
            name = paper_dir.name.replace("_", " ")
            for t in tiles:
                self.samples.append({
                    "image_path": str(t),
                    "query": f"Medical paper about {name}",
                    "paper": name,
                })

        # Shuffle then split
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * (1 - val_ratio))
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        from PIL import Image
        img = Image.open(s["image_path"]).convert("RGB")
        # Build conversation for chat template (inserts image tokens automatically)
        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": s["query"]},
            ]}
        ]
        return {"image": img, "conversation": conversation, "query": s["query"], "paper": s["paper"]}


def collate_fn(batch):
    images = [b["image"] for b in batch]
    conversations = [b["conversation"] for b in batch]
    papers = [b["paper"] for b in batch]
    return {"images": images, "conversations": conversations, "papers": papers}


def contrastive_loss(embeddings, papers, temperature=0.07):
    """InfoNCE loss: same paper = positive pair. Falls back to uniform loss."""
    labels = torch.tensor([hash(p) for p in papers], device=embeddings.device)
    sim = embeddings @ embeddings.T / temperature
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask.fill_diagonal_(False)
    if pos_mask.sum() == 0:
        # No positives: push all embeddings apart (uniformity loss)
        sim_exp = torch.exp(sim)
        loss = sim_exp.sum(dim=1).log().mean()
        return loss  # Still connected to graph via embeddings
    loss = 0.0
    n_pos = 0
    for i in range(len(labels)):
        pos_sims = sim[i][pos_mask[i]]
        if len(pos_sims) == 0:
            continue
        denom = torch.exp(sim[i]).sum()
        num = torch.exp(pos_sims).sum()
        loss += -torch.log(num / denom)
        n_pos += 1
    return loss / max(n_pos, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--pixels-dir", type=str, default="pixels")
    parser.add_argument("--output-dir", type=str, default="models/finetuned_lora")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = "models/Qwen3-VL-Embedding-2B"
    print("Loading model...")
    base_model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
    base_model = base_model.to(device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    train_ds = PaperTileDataset(args.pixels_dir, processor, split="train")
    val_ds = PaperTileDataset(args.pixels_dir, processor, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            # Apply chat template to insert image tokens
            texts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in batch["conversations"]]
            inputs = processor(
                text=texts,
                images=batch["images"],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.last_hidden_state
            emb = last_hidden[:, -1]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            loss = contrastive_loss(emb, batch["papers"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                texts = [processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in batch["conversations"]]
                inputs = processor(
                    text=texts,
                    images=batch["images"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                last_hidden = outputs.last_hidden_state
                emb = last_hidden[:, -1]
                emb = emb / emb.norm(dim=-1, keepdim=True)
                val_loss += contrastive_loss(emb, batch["papers"]).item()

        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={total_loss/len(train_loader):.4f}, val_loss={val_loss/len(val_loader):.4f}")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    print(f"Model saved to {out}")


if __name__ == "__main__":
    main()

"""Generate a small synthetic contrails dataset for testing.
Creates random data in the same format as the Kaggle dataset.
"""
import numpy as np
import os
import json
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "env")
N_SAMPLES = 5  # Small for testing
IMG_SIZE = 256  # Match origibal shape for compatibility
N_TIMES_BEFORE = 4
N_TIMES_AFTER = 3
N_TIMES = N_TIMES_BEFORE + N_TIMES_AFTER + 1

random.seed(42)
np.random.seed(42)

def generate_sample(sample_id, split):
    """Generate a single fake contrail sample."""
    sample_dir = os.path.join(DATA_DIR, split, str(sample_id))
    os.makedirs(sample_dir, exist_ok=True)

    # Generate random band data (simulating brightness temperatures)
    for band in [11, 14, 15]:
        data = np.random.uniform(200, 310, (IMG_SIZE, IMG_SIZE, N_TIMES)).astype(np.float32)
        np.save(os.path.join(sample_dir, f"band_{band}.npy"), data)

    # Generate human pixel mask (small random regions as "contrails")
    mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
    num_contrails = random.randint(0, 2)
    for _ in range(num_contrails):
        cx, cy = random.randint(10, IMG_SIZE-10), random.randint(10, IMG_SIZE-10)
        # Thin elongated shape
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0),(0,2),(0,-2)]:
            if 0 <= cx+dx*3 < IMG_SIZE and 0 <= cy+dy*3 < IMG_SIZE:
                for i in range(3):
                    for j in range(3):
                        if random.random() > 0.3:
                            mask[min(IMG_SIZE-1, max(0, cx+dx*i)), min(IMG_SIZE-1, max(0, cy+dy*j)), 0] = 1
    np.save(os.path.join(sample_dir, "human_pixel_masks.npy"), mask)

    if split == "train":
        # Generate individual human labels (simulating multiple annotators)
        individual = np.zeros((IMG_SIZE, IMG_SIZE, 1, 5), dtype=np.uint8)
        for r in range(5):
            noise = np.random.binomial(1, 0.3, (IMG_SIZE, IMG_SIZE, 1))
            individual[:, :, :, r] = np.clip(mask + noise, 0, 1)
        np.save(os.path.join(sample_dir, "human_individual_masks.npy"), individual)


if __name__ == "__main__":
    for split in ["train", "validation"]:
        split_dir = os.path.join(DATA_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(N_SAMPLES):
            sample_id = random.randint(10**18, 10**19-1)
            generate_sample(sample_id, split)
            print(f"  Created {split}/{sample_id}")

    # Copy validation -> test (same as prepare.py does)
    import shutil
    test_dir = os.path.join(DATA_DIR, "test")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    shutil.copytree(os.path.join(DATA_DIR, "validation"), test_dir)

    # Create test_answer with ground truth (masks preserved)
    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    test_answer_dir = os.path.join(scripts_dir, "test_answer")
    if os.path.exists(test_answer_dir):
        shutil.rmtree(test_answer_dir)
    shutil.copytree(os.path.join(DATA_DIR, "validation"), test_answer_dir)

    # Remove masks from test (matches what prepare.py does)
    for item in os.listdir(test_dir):
        mask_path = os.path.join(test_dir, item, "human_pixel_masks.npy")
        if os.path.exists(mask_path):
            os.remove(mask_path)

    # Create sample submission
    ids = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    with open(os.path.join(DATA_DIR, "sample_submission.csv"), "w") as f:
        f.write("record_id,encoded_pixels\n")
        for sid in ids:
            f.write(f"{sid},\n")

    # Create metadata
    metadata = {}
    for split in ["train", "validation"]:
        for item in os.listdir(os.path.join(DATA_DIR, split)):
            metadata[item] = {"timestamps": ["2023-01-01 12:00:00"] * N_TIMES}
    with open(os.path.join(DATA_DIR, "train_metadata.json"), "w") as f:
        json.dump(metadata, f)
    with open(os.path.join(DATA_DIR, "validation_metadata.json"), "w") as f:
        json.dump(metadata, f)

    # Mark as prepared
    prepared_flag = os.path.join(scripts_dir, "prepared")
    with open(prepared_flag, "w") as f:
        f.write("success")

    print(f"\nDone! Created {N_SAMPLES} synthetic samples in {DATA_DIR}")
    size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(DATA_DIR) for f in fn)
    print(f"Total size: {size/1e6:.1f} MB")
    print("\nNow run: python -m MLAgentBench.runner --task identify-contrails ...")

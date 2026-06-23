import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "models/Qwen_2.5_3B_nothink/VQA_X-Ray"
DATA_PATH = "data/medmnist_subset/chestmnist_3class.npz"

# Load model
print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Load a sample chest X-ray
data = np.load(DATA_PATH)
images = data["images"]
labels = data["labels"]

# Take first 2 samples: label 0 = normal, label 1 = pneumonia
idx = 300
img = images[idx]
gt_label = int(labels[idx])

# Upsample 28x28 -> 224x224 for Qwen
img_pil = Image.fromarray(img).convert("RGB").resize((224, 224))
img_pil.save("/tmp/test_chest_xray.png")

question = "Does this chest X-ray show any signs of pneumonia? Answer yes or no inside <answer> tags."

message = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "/tmp/test_chest_xray.png"},
        {"type": "text", "text": question},
    ]
}]

text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info([message])
inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}

print("Running inference...")
generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=128, do_sample=False)
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(f"\nLabel: {gt_label} (0=normal, 1=pneumonia)")
print(f"Model output:\n{output_text}")

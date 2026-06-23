import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from medmnist import PneumoniaMNIST

MODEL_PATH = "models/Qwen_2.5_3B_nothink/VQA_X-Ray"

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# Load PneumoniaMNIST at 128x128
train = PneumoniaMNIST(split="train", download=True, size=224)

# Find normal (0) and pneumonia (1) samples
normal_img = None
pneumonia_img = None
for i in range(len(train)):
    img, label = train[i]
    if label[0] == 0 and normal_img is None:
        normal_img = img
        normal_idx = i
    if label[0] == 1 and pneumonia_img is None:
        pneumonia_img = img
        pneumonia_idx = i
    if normal_img is not None and pneumonia_img is not None:
        break

print(f"Normal sample index: {normal_idx}")
print(f"Pneumonia sample index: {pneumonia_idx}")

for name, img_pil, gt in [("Normal", normal_img, 0), ("Pneumonia", pneumonia_img, 1)]:
    # Save and convert to RGB
    img_rgb = img_pil.convert("RGB")
    img_rgb.save(f"/tmp/test_pneumonia_{name.lower()}.png")

    question = "Does this chest X-ray show any signs of pneumonia? Answer yes or no inside <answer> tags."

    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"/tmp/test_pneumonia_{name.lower()}.png"},
            {"type": "text", "text": question},
        ]
    }]

    text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([message])
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    print(f"\n--- {name} (label={gt}) at 224x224 ---")
    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Model: {output_text}")

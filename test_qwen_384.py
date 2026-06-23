import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from medmnist import ChestMNIST

MODEL_PATH = "models/Qwen_2.5_3B_nothink/VQA_X-Ray"

print("Loading model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

train = ChestMNIST(split="train", download=False, size=224)

samples = [(0, "Normal (no findings)"), (42, "Pneumonia + atelectasis + infiltration")]

TEMPLATE = "{Question} Provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags."

for idx, desc in samples:
    img_pil, label = train[idx]
    # Resize to 384x384 as per Med-R1 training specs
    img_resized = img_pil.resize((384, 384), Image.LANCZOS)
    img_rgb = img_resized.convert("RGB")
    path = f"/tmp/test_384_{idx}.png"
    img_rgb.save(path)

    questions = [
        "Do you see any abnormalities in this chest X-ray? A)Yes, B)No",
        "What specific condition is represented in this image of a lung? A)Pneumonia, B)Normal lung, C)Tuberculosis, D)Lung cancer",
    ]

    print(f"\n{'='*60}")
    print(f"--- {desc} (384x384) ---")

    for q in questions:
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{path}"},
                {"type": "text", "text": TEMPLATE.format(Question=q)},
            ]
        }]
        text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([message])
        inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=128, do_sample=False)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(f"Q: {q}")
        print(f"A: {output_text}")
        print()

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

train = PneumoniaMNIST(split="train", download=False, size=224)

normal_img = None
pneumonia_img = None
for i in range(len(train)):
    img, label = train[i]
    if label[0] == 0 and normal_img is None:
        normal_img = img; normal_idx = i
    if label[0] == 1 and pneumonia_img is None:
        pneumonia_img = img; pneumonia_idx = i
    if normal_img is not None and pneumonia_img is not None:
        break

TEMPLATE = "{Question} Provide the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags."

tests = [
    ("Normal (abnormality check)", normal_img,
     "Does this chest X-ray show any signs of pneumonia? A)Yes, B)No"),
    ("Pneumonia (abnormality check)", pneumonia_img,
     "Does this chest X-ray show any signs of pneumonia? A)Yes, B)No"),
    ("Normal (diagnosis)", normal_img,
     "What specific condition is represented in this image of a lung? A)Pneumonia, B)Asthma, C)Emphysema, D)Normal lung"),
    ("Pneumonia (diagnosis)", pneumonia_img,
     "What specific condition is represented in this image of a lung? A)Pneumonia, B)Asthma, C)Emphysema, D)Normal lung"),
    ("Normal (modality)", normal_img,
     "What imaging technique is utilized to capture this picture? A)X-ray, B)CT scan, C)PET scan, D)MRI"),
]

for name, img_pil, question in tests:
    img_rgb = img_pil.convert("RGB")
    path = f"/tmp/test_mcq_{name.split()[0].lower()}.png"
    img_rgb.save(path)

    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": f"file://{path}"},
            {"type": "text", "text": TEMPLATE.format(Question=question)},
        ]
    }]

    text = processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([message])
    inputs = processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    generated_ids = model.generate(**inputs, use_cache=True, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    print(f"\n--- {name} ---")
    print(f"Q: {question}")
    print(f"A: {output_text}")

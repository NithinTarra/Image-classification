import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

MODEL_NAME = "google/vit-base-patch16-224"

IMAGE_PATH = "data/test_images/Dog_Breeds.jpg"

# 1. Load the image processor
print("Loading image processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

# 2. Load the pretrained classification model
print("Loading ViT model...")

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

# 3. Load the input image
print("Loading input image...")

image = Image.open(IMAGE_PATH).convert("RGB")

# 4. Preprocess the image
inputs = processor(
    images=image,
    return_tensors="pt"
)

# 5. Predict the image class
print("Classifying image...")

with torch.inference_mode():
    outputs = model(**inputs)

# 6. Convert model outputs into probabilities
probabilities = torch.softmax(
    outputs.logits,
    dim=-1
)[0]

# 7. Get the top five predictions
top_results = torch.topk(
    probabilities,
    k=5
)

print("\n========== CLASSIFICATION RESULTS ==========")

for score, class_id in zip(
    top_results.values,
    top_results.indices
):
    label = model.config.id2label[class_id.item()]

    print(
        f"Class: {label:<25} "
        f"Score: {score.item():.4f}"
    )

print("\nClassification completed successfully!")
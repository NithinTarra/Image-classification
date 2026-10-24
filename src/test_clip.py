import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
IMAGE_PATH = "data/test_images/Dog_Breeds.jpg"

device = torch.device("cpu")

# 1. Load the processor
print("Loading CLIP processor...")

processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# 2. Load the model
print("Loading CLIP model...")

model = CLIPModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# 3. Load image
print("Loading image...")

image = Image.open(IMAGE_PATH).convert("RGB")

# 4. Preprocess image
inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# 5. Generate image embedding
print("Generating image embedding...")

with torch.inference_mode():
    vision_outputs = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    # Extract the pooled visual representation
    pooled_output = vision_outputs.pooler_output

    # Project it into CLIP's joint embedding space
    image_features = model.visual_projection(pooled_output)

# 6. Normalize embedding
image_features = torch.nn.functional.normalize(
    image_features,
    p=2,
    dim=-1
)

# 7. Convert to NumPy
embedding = image_features.cpu().numpy()

# 8. Display results
print("\n========== CLIP RESULTS ==========")

print("Embedding shape:", embedding.shape)
print("Embedding dimension:", embedding.shape[1])
print("Embedding norm:", float(image_features.norm().item()))

print("First 10 values:")
print(embedding[0][:10])

print("\nCLIP embedding generated successfully!")
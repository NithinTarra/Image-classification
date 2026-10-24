import json
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# Configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
OUTPUT_DIR = Path("data/index")
BATCH_SIZE = 16
MAX_IMAGES = 2000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cpu")

print("Loading CIFAR-10 dataset...")

dataset = load_dataset(
    "uoft-cs/cifar10",
    split="train"
)

# Use a smaller subset for initial development.
dataset = dataset.select(
    range(min(MAX_IMAGES, len(dataset)))
)

print("Reference images:", len(dataset))

print("\nLoading CLIP processor...")

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

print("Loading CLIP model...")

model = CLIPModel.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

all_embeddings = []
metadata = []

print("\nGenerating reference embeddings...")

for start in tqdm(range(0, len(dataset), BATCH_SIZE)):

    batch = dataset[start:start + BATCH_SIZE]

    images = [
        image.convert("RGB")
        for image in batch["img"]
    ]

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled_output = vision_outputs.pooler_output

        image_features = model.visual_projection(
            pooled_output
        )

        image_features = torch.nn.functional.normalize(
            image_features,
            p=2,
            dim=-1
        )

    embeddings = image_features.cpu().numpy().astype(
        np.float32
    )

    all_embeddings.append(embeddings)

    for i, label in enumerate(batch["label"]):

        metadata.append({
            "dataset_index": start + i,
            "label_id": int(label),
            "label": dataset.features["label"].int2str(
                int(label)
            )
        })


# Combine embeddings into one matrix.
reference_embeddings = np.concatenate(
    all_embeddings,
    axis=0
)

print("\nEmbedding matrix:", reference_embeddings.shape)

# Build exact cosine-similarity search index.
index = faiss.IndexFlatIP(512)

index.add(reference_embeddings)

# Save the index and metadata.
faiss.write_index(
    index,
    str(OUTPUT_DIR / "reference.index")
)

with open(
    OUTPUT_DIR / "reference_metadata.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=2
    )

print("\nFAISS index created successfully!")
print("Indexed images:", index.ntotal)
print("Embedding dimensions:", index.d)
print("Saved index:", OUTPUT_DIR / "reference.index")
print(
    "Saved metadata:",
    OUTPUT_DIR / "reference_metadata.json"
)
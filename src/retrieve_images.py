import json
from pathlib import Path

import faiss
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# Configuration
MODEL_NAME = "openai/clip-vit-base-patch32"

IMAGE_PATH = Path("data/test_images/Dog_Breeds.jpg")

INDEX_PATH = Path("data/index/reference.index")

METADATA_PATH = Path("data/index/reference_metadata.json")

TOP_K = 10

device = torch.device("cpu")


def main():

    # Check input files
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Input image not found: {IMAGE_PATH}"
        )

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    # Load metadata
    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    # Load FAISS index
    print("Loading FAISS index...")

    index = faiss.read_index(str(INDEX_PATH))

    print("Indexed images:", index.ntotal)

    # Load CLIP
    print("\nLoading CLIP model...")

    processor = CLIPProcessor.from_pretrained(
        MODEL_NAME
    )

    model = CLIPModel.from_pretrained(
        MODEL_NAME
    ).to(device)

    model.eval()

    # Load query image
    print("\nLoading input image...")

    image = Image.open(IMAGE_PATH).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Generate query embedding
    print("Generating query embedding...")

    with torch.inference_mode():

        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        pooled_output = vision_outputs.pooler_output

        image_features = model.visual_projection(
            pooled_output
        )

        image_features = F.normalize(
            image_features,
            p=2,
            dim=-1
        )

    query_embedding = (
        image_features.cpu()
        .numpy()
        .astype("float32")
    )

    # Validate embedding dimension
    if query_embedding.shape[1] != index.d:
        raise ValueError(
            f"Embedding dimension {query_embedding.shape[1]} "
            f"does not match FAISS index dimension {index.d}"
        )

    # Retrieve nearest neighbors
    k = min(TOP_K, index.ntotal)

    print(f"\nRetrieving top {k} similar images...")

    scores, indices = index.search(
        query_embedding,
        k
    )

    # Display results
    print("\n========== RETRIEVAL RESULTS ==========")

    for rank, (score, idx) in enumerate(
        zip(scores[0], indices[0]),
        start=1
    ):

        if idx < 0:
            continue

        item = metadata[int(idx)]

        print(f"\nRank: {rank}")
        print(f"Reference ID: {item['dataset_index']}")
        print(f"Label: {item['label']}")
        print(f"Similarity: {float(score):.4f}")

    print("\nImage retrieval completed successfully!")


if __name__ == "__main__":
    main()
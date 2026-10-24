import json
from pathlib import Path
from llm_explanation import generate_explanation
from evidence_analysis import analyze_evidence

import faiss
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPModel,
    CLIPProcessor,
)

from evidence_analysis import analyze_evidence


# Configuration
DEVICE = torch.device("cpu")

VIT_NAME = "google/vit-base-patch16-224"
CLIP_NAME = "openai/clip-vit-base-patch32"

INDEX_PATH = Path("data/index/reference.index")
METADATA_PATH = Path("data/index/reference_metadata.json")

TOP_K = 10


def load_models():
    print("Loading ViT...")

    vit_processor = AutoImageProcessor.from_pretrained(
        VIT_NAME
    )

    vit_model = AutoModelForImageClassification.from_pretrained(
        VIT_NAME
    ).to(DEVICE)

    vit_model.eval()

    print("Loading CLIP...")

    clip_processor = CLIPProcessor.from_pretrained(
        CLIP_NAME
    )

    clip_model = CLIPModel.from_pretrained(
        CLIP_NAME
    ).to(DEVICE)

    clip_model.eval()

    return (
        vit_processor,
        vit_model,
        clip_processor,
        clip_model,
    )


def predict_image(image, processor, model):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )[0]

    class_id = int(probabilities.argmax().item())

    return {
        "class": model.config.id2label[class_id],
        "confidence": float(probabilities[class_id].item()),
    }


def generate_embedding(image, processor, model):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    with torch.inference_mode():
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"].to(DEVICE)
        )

        pooled_output = vision_outputs.pooler_output

        features = model.visual_projection(pooled_output)

        features = F.normalize(
            features,
            p=2,
            dim=-1
        )

    return features.cpu().numpy().astype("float32")


def retrieve_evidence(embedding, index, metadata):
    k = min(TOP_K, index.ntotal)

    scores, indices = index.search(embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = metadata[int(idx)]

        results.append({
            "reference_id": item["dataset_index"],
            "label": item["label"],
            "similarity": float(score),
        })

    return results


def run_pipeline(image_path):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_PATH}"
        )

    # Load reference database
    index = faiss.read_index(str(INDEX_PATH))

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    if index.ntotal != len(metadata):
        raise ValueError(
            "FAISS index size does not match metadata size."
        )

    # Load models
    (
        vit_processor,
        vit_model,
        clip_processor,
        clip_model,
    ) = load_models()

    # Load image
    image = Image.open(image_path).convert("RGB")

    # 1. Classification
    prediction = predict_image(
        image,
        vit_processor,
        vit_model
    )

    # 2. CLIP embedding
    embedding = generate_embedding(
        image,
        clip_processor,
        clip_model
    )

    if embedding.shape[1] != index.d:
        raise ValueError(
            "Query embedding dimension does not match FAISS index."
        )

    # 3. Similarity retrieval
    evidence = retrieve_evidence(
        embedding,
        index,
        metadata
    )
    
    # 4. Evidence analysis
    report = analyze_evidence(
        prediction["class"],
        evidence
    )

    # 5. Generate an evidence-grounded explanation
    explanation = generate_explanation(
        prediction=prediction,
        evidence=evidence,
        analysis=report,
    )

    return {
        "image": str(image_path),
        "prediction": prediction,
        "evidence": evidence,
        "analysis": report,
        "explanation": explanation,
    }
   
if __name__ == "__main__":
    result = run_pipeline(
        "data/test_images/Dog_Breeds.jpg"
    )

    print("\n========== FINAL PIPELINE REPORT ==========")

    print("\nPrediction:")
    print(json.dumps(result["prediction"], indent=2))

    print("\nRetrieved Evidence:")
    print(json.dumps(result["evidence"], indent=2))

    print("\nEvidence Analysis:")
    print(json.dumps(result["analysis"], indent=2))
    
    print("\n========== LLM EXPLANATION ==========")
    print(result["explanation"])
from collections import Counter
from statistics import mean, median


# ImageNet labels that belong to the broad dog category.
DOG_BREEDS = {
    "affenpinscher",
    "afghan hound",
    "african hunting dog",
    "airedale",
    "american staffordshire terrier",
    "appenzeller",
    "basenji",
    "basset",
    "beagle",
    "bedlington terrier",
    "bernese mountain dog",
    "black-and-tan coonhound",
    "bloodhound",
    "bluetick",
    "border collie",
    "border terrier",
    "borzoi",
    "boston bull",
    "bouvier des flandres",
    "boxer",
    "brabancon griffon",
    "briard",
    "brittany spaniel",
    "bull mastiff",
    "cairn",
    "cardigan",
    "chesapeake bay retriever",
    "chihuahua",
    "chow",
    "clumber",
    "cocker spaniel",
    "collie",
    "curly-coated retriever",
    "dandie dinmont",
    "dhole",
    "dingo",
    "doberman",
    "english foxhound",
    "english setter",
    "english springer",
    "entlebucher",
    "eskimo dog",
    "flat-coated retriever",
    "french bulldog",
    "german shepherd",
    "german short-haired pointer",
    "giant schnauzer",
    "golden retriever",
    "gordon setter",
    "great dane",
    "great pyrenees",
    "groenendael",
    "ibizan hound",
    "irish setter",
    "irish terrier",
    "irish water spaniel",
    "irish wolfhound",
    "italian greyhound",
    "japanese spaniel",
    "keeshond",
    "kelpie",
    "kerry blue terrier",
    "komondor",
    "kuvasz",
    "labrador retriever",
    "lakeland terrier",
    "leonberg",
    "lhasa",
    "malamute",
    "malinois",
    "maltese dog",
    "mexican hairless",
    "miniature pinscher",
    "miniature poodle",
    "miniature schnauzer",
    "newfoundland",
    "norfolk terrier",
    "norwegian elkhound",
    "norwich terrier",
    "old english sheepdog",
    "otterhound",
    "papillon",
    "pekinese",
    "pembroke",
    "pomeranian",
    "pug",
    "redbone",
    "rhodesian ridgeback",
    "rottweiler",
    "saint bernard",
    "saluki",
    "samoyed",
    "schipperke",
    "scotch terrier",
    "scottish deerhound",
    "sealyham terrier",
    "shetland sheepdog",
    "shih-tzu",
    "siberian husky",
    "silky terrier",
    "soft-coated wheaten terrier",
    "staffordshire bullterrier",
    "standard poodle",
    "standard schnauzer",
    "sussex spaniel",
    "tibetan mastiff",
    "tibetan terrier",
    "toy poodle",
    "toy terrier",
    "vizsla",
    "walker hound",
    "weimaraner",
    "welsh springer spaniel",
    "west highland white terrier",
    "whippet",
    "wire-haired fox terrier",
    "yorkshire terrier",
}

CAT_LABELS = {
    "cat",
    "tabby",
    "tiger cat",
    "persian cat",
    "siamese cat",
    "egyptian cat",
}


def normalize_label(label):
    """
    Map known ImageNet dog-breed labels and CIFAR-10
    animal labels into broad categories.

    Unknown labels remain unchanged.
    """

    label = " ".join(label.lower().strip().split())

    if label == "dog" or label in DOG_BREEDS:
        return "dog"

    if label in CAT_LABELS:
        return "cat"

    return label


def analyze_evidence(predicted_class, retrieved_results):
    """
    Compare the predicted class with retrieved reference labels.

    Each retrieved result must contain:
        label: reference category
        similarity: CLIP cosine similarity
    """

    if not retrieved_results:
        raise ValueError(
            "At least one retrieved result is required."
        )

    if any(
        "label" not in item or "similarity" not in item
        for item in retrieved_results
    ):
        raise ValueError(
            "Every result must contain label and similarity."
        )

    predicted_category = normalize_label(predicted_class)

    labels = [
        normalize_label(item["label"])
        for item in retrieved_results
    ]

    similarities = [
        float(item["similarity"])
        for item in retrieved_results
    ]

    if any(not -1.0 <= score <= 1.0 for score in similarities):
        raise ValueError(
            "Expected normalized CLIP cosine similarities between -1 and 1."
        )

    matching = sum(
        label == predicted_category
        for label in labels
    )

    total = len(labels)
    agreement = matching / total
    label_distribution = Counter(labels)

    return {
        "predicted_class": predicted_class,
        "predicted_category": predicted_category,
        "total_retrieved": total,
        "matching_labels": matching,
        "agreement": round(agreement, 4),
        "disagreement": round(1.0 - agreement, 4),
        "mean_similarity": round(mean(similarities), 4),
        "median_similarity": round(median(similarities), 4),
        "label_distribution": dict(label_distribution),
        "interpretation": (
            "Broad-category agreement only; this does not verify "
            "the fine-grained predicted breed."
            if predicted_category == "dog"
            else "Agreement is based on normalized category labels."
        ),
    }
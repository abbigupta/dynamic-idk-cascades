"""Optional Hugging Face streaming helpers for ImageNet-V2.

The notebook is self-contained, but this file lets you test streaming outside
Jupyter without storing raw ImageNet-V2 images in the project.
"""

HF_DATASET = "vaishaal/ImageNetV2"

VARIANT_FOLDERS = {
    "matched-frequency": "imagenetv2-matched-frequency-format-val",
    "threshold-0.7": "imagenetv2-threshold0.7-format-val",
    "top-images": "imagenetv2-top-images-format-val",
}

VARIANT_URLS = {
    "matched-frequency": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
    "threshold-0.7": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz",
    "top-images": "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz",
}


def label_from_key(key: str) -> int:
    """Extract the numeric ImageNet class ID from a streamed archive key."""
    return int(key.split("/")[1])


def stream_imagenet_v2_rows(variant: str, max_samples: int | None = None):
    """Yield streamed ImageNet-V2 rows as image/label/key dictionaries."""
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found.*")
    from datasets import load_dataset

    if variant not in VARIANT_FOLDERS:
        choices = ", ".join(VARIANT_FOLDERS)
        raise ValueError(f"Unknown variant '{variant}'. Choose one of: {choices}")

    stream = load_dataset(
        "webdataset",
        data_files={"train": VARIANT_URLS[variant]},
        split="train",
        streaming=True,
    )

    emitted = 0
    for row in stream:
        yield {
            "image": row["jpeg"].convert("RGB"),
            "label": label_from_key(row["__key__"]),
            "key": row["__key__"],
        }
        emitted += 1
        if max_samples is not None and emitted >= max_samples:
            break


if __name__ == "__main__":
    for variant in VARIANT_FOLDERS:
        print(f"{variant}:")
        for row in stream_imagenet_v2_rows(variant, max_samples=2):
            print(" ", row["label"], row["key"])

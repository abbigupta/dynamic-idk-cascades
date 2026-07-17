"""Generate all ImageNet-V2 model caches.

Click "Run Python File" in VS Code, or run:

    .venv/bin/python scripts/cache_logits.py

Every run regenerates all 12 cache files in artifacts/.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision import models

from dataset import stream_imagenet_v2_rows


REPO_ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "matched-frequency": "matched",
    "top-images": "top",
    "threshold-0.7": "threshold07",
}

MODEL_BATCH_SIZES = {
    "resnet18": 64,
    "resnet34": 32,
    "resnet50": 32,
    "resnet152": 16,
}


def load_model(model_name, device):
    model_factories = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
    }
    factory, weights = model_factories[model_name]
    model = factory(weights=weights).to(device)
    model.eval()
    return model, weights.transforms()


def make_loader(variant, transform, batch_size):
    class ImageNetV2Dataset(IterableDataset):
        def __iter__(self):
            for row in stream_imagenet_v2_rows(variant):
                yield transform(row["image"]), row["label"], row["key"]

    return DataLoader(
        ImageNetV2Dataset(),
        batch_size=batch_size,
    )


def cache_model(variant, model_name, output_file, batch_size, device):
    model, transform = load_model(model_name, device)
    loader = make_loader(variant, transform, batch_size)

    all_probabilities = []
    all_labels = []
    all_predictions = []
    all_times = []
    all_keys = []
    processed = 0

    with torch.inference_mode():
        for images, labels, keys in loader:
            images = images.to(device)

            if device.type == "mps":
                torch.mps.synchronize()
            start = time.perf_counter()
            logits = model(images)
            if device.type == "mps":
                torch.mps.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000

            probabilities = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
            all_probabilities.append(probabilities)
            all_labels.append(labels.numpy().astype(np.int64))
            all_predictions.append(np.argmax(probabilities, axis=1).astype(np.int64))
            all_times.extend([elapsed_ms / len(images)] * len(images))
            all_keys.extend(keys)

            processed += len(images)
            print(f"{model_name} {variant}: processed {processed} images")

    np.savez_compressed(
        output_file,
        probabilities=np.concatenate(all_probabilities),
        labels=np.concatenate(all_labels),
        predictions=np.concatenate(all_predictions),
        times_ms=np.array(all_times, dtype=np.float64),
        keys=np.array(all_keys, dtype=str),
        model_name=np.array(model_name),
        variant=np.array(variant),
    )
    print(f"Saved {output_file}")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    artifacts_dir = REPO_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    print(f"Using device: {device}")
    print("Regenerating all ImageNet-V2 caches.")

    for variant, prefix in VARIANTS.items():
        for model_name, batch_size in MODEL_BATCH_SIZES.items():
            output_file = artifacts_dir / f"{prefix}_{model_name}.npz"
            cache_model(
                variant,
                model_name,
                output_file,
                batch_size,
                device,
            )

    print("Finished generating all 12 cache files.")


if __name__ == "__main__":
    main()

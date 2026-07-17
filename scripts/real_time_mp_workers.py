import time

import torch
from torchvision import models


MODEL_SPECS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
}


def model_worker(model_name, device_name, job_queue, result_queue, signal_ready=False):
    try:
        device = torch.device(device_name)
        factory, weights = MODEL_SPECS[model_name]
        model = factory(weights=weights).to(device).eval()
        if signal_ready:
            result_queue.put(("ready", model_name))

        while True:
            job = job_queue.get()
            if job is None:
                break

            sample_index, images = job
            start = time.perf_counter()
            images = images.to(device)

            with torch.inference_mode():
                logits = model(images)
                probabilities = torch.softmax(logits, dim=1)
                if device_name == "mps":
                    torch.mps.synchronize()
                probabilities = probabilities[0].detach().cpu().numpy()

            result_queue.put(
                (
                    sample_index,
                    model_name,
                    probabilities,
                    int(probabilities.argmax()),
                    float(probabilities.max()),
                    (time.perf_counter() - start) * 1000.0,
                )
            )
    except Exception as exc:
        result_queue.put(("error", model_name, repr(exc)))


def model_group_worker(worker_name, model_names, device_name, job_queue, result_queue, signal_ready=False):
    try:
        device = torch.device(device_name)
        loaded_models = {}
        for model_name in model_names:
            factory, weights = MODEL_SPECS[model_name]
            loaded_models[model_name] = factory(weights=weights).to(device).eval()
        if signal_ready:
            result_queue.put(("ready", worker_name))

        while True:
            job = job_queue.get()
            if job is None:
                break

            sample_index, model_name, images = job
            start = time.perf_counter()
            images = images.to(device)

            with torch.inference_mode():
                logits = loaded_models[model_name](images)
                probabilities = torch.softmax(logits, dim=1)
                if device_name == "mps":
                    torch.mps.synchronize()
                probabilities = probabilities[0].detach().cpu().numpy()

            result_queue.put(
                (
                    sample_index,
                    worker_name,
                    model_name,
                    probabilities,
                    int(probabilities.argmax()),
                    float(probabilities.max()),
                    (time.perf_counter() - start) * 1000.0,
                )
            )
    except Exception as exc:
        result_queue.put(("error", worker_name, repr(exc)))

# Dynamic IDK Cascades

Research workspace for Random Forest IDK routing and dynamic parallel
classifier scheduling on ImageNet-V2.

## Files

- `notebooks/` — research notebooks for RF routing, cached-output scheduling, and real worker experiments.
- `scripts/cache_logits.py` — one-click script for generating all model-output caches.
- `scripts/dataset.py` — optional streaming dataset helpers for testing outside Jupyter.
- `scripts/real_time_mp_workers.py` — multiprocessing worker helper for live CPU/MPS experiments.
- `docs/architecture/` — visual overview, pipeline diagrams, and scheduler study guide.
- `artifacts/` — cached model probability/timing files. Raw images are not required in the project.
- `runs/` — generated benchmark result JSON and prediction `.npz` files.

## Dataset access

The notebook streams ImageNet-V2 from Hugging Face using `datasets.load_dataset("vaishaal/ImageNetV2", streaming=True)`.
It does not require `data/imagenetv2` or any raw dataset folder in this repo.

Install the needed packages in your venv if they are missing:

```bash
pip install datasets torch torchvision timm scikit-learn numpy pillow
```

## How to work

1. Open `notebooks/dynamic_rf_IDK_Cascade.ipynb` in VS Code or Jupyter.
2. Run the setup and streaming check cells.
3. Open `scripts/cache_logits.py` and click **Run Python File**, or run `.venv/bin/python scripts/cache_logits.py`.
4. Train the original RF skipper and compare `no-skip`, `threshold`, and `rf`.
5. Open `notebooks/dynamic_parallel_IDK_scheduler.ipynb` and train the heavy RF router on matched-frequency plus top-images.
6. Compare the sequential baseline with dynamic worker reuse across `max_in_flight` limits on threshold-0.7.

See [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) for diagrams of the complete data, training, cache, routing, and scheduler flows.

Full experiments may be slower with streaming. Once logits are cached in `artifacts/`, later RF/cascade experiments do not need to stream the images again.

`cache_logits.py` takes no command-line options. Each run regenerates all 12
cache files for the three ImageNet-V2 variants and four ResNet models:

```bash
.venv/bin/python scripts/cache_logits.py
```

On an Apple Silicon Mac, the script uses PyTorch MPS acceleration when
available and falls back to CPU otherwise.

The heavy router is trained only on images where both early models return IDK.
It uses confidence, entropy, and margin from both models, then selects
ResNet-50 when that model is sufficiently confident and ResNet-152 otherwise.

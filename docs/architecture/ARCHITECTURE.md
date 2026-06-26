# Dynamic IDK Cascades Architecture

This document shows how the current project streams ImageNet-V2, creates
reusable model-output caches, trains Random Forest routing policies, and
simulates a parallel IDK scheduler.

The project has two main phases:

1. **Offline preparation:** stream images, run pretrained models, and save
   aligned `.npz` caches.
2. **Cached-output research:** train routing models and compare cascade or
   scheduling policies without repeating expensive neural-network inference.

## Visual legend

The diagrams use one consistent visual language:

| Color | Meaning |
| --- | --- |
| Blue | Dataset, feature, or scheduler-control input |
| Purple | Pretrained model, trained Random Forest, or model worker |
| Green | Reusable cache or worker queue |
| Orange | Processing logic, threshold decision, or routing rule |
| Gray | Final prediction or reported metric |
| Red | Canceled or wasted work |

## 1. Project overview

![Project overview](docs/architecture/project-overview.svg)

```mermaid
%%{init: {"theme": "neutral", "flowchart": {"curve": "basis", "htmlLabels": true}}}%%
flowchart TB
    subgraph source["Data source"]
        hf["ImageNet-V2 on Hugging Face<br/>matched-frequency · top-images · threshold-0.7"]:::data
    end

    subgraph preparation["Data and model-output preparation"]
        helper["dataset.py<br/>optional streaming helper"]:::logic
        research["dynamic_rf_IDK_Cascade.ipynb<br/>streaming · transforms · inference · cache creation"]:::notebook
        models["Pretrained classifiers<br/>ResNet-18 · ResNet-34 · ResNet-50 · ResNet-152"]:::model
    end

    subgraph storage["Reusable artifacts"]
        train_cache[("Training caches<br/>matched-frequency + top-images<br/>4 models per variant")]:::artifact
        test_cache[("Test caches<br/>threshold-0.7<br/>4 models")]:::artifact
        contract["Each .npz cache<br/>probabilities · labels · predictions<br/>times_ms · keys"]:::contract
    end

    subgraph experiments["Cached-output experiments"]
        skip["Original RF IDK cascade<br/>no-skip · threshold · RF-skip"]:::logic
        scheduler["dynamic_parallel_IDK_scheduler.ipynb<br/>heavy RF router + discrete-event scheduler"]:::notebook
        guide["dynamic_parallel_IDK_scheduler_study_guide.md<br/>line-by-line learning guide"]:::doc
    end

    subgraph results["Experiment outputs"]
        reports["Accuracy · routing counts · latency<br/>throughput · utilization · wasted work"]:::output
    end

    hf --> helper
    hf --> research
    helper -. reusable streaming functions .-> research
    models --> research
    research --> train_cache
    research --> test_cache
    contract --- train_cache
    contract --- test_cache
    train_cache --> skip
    test_cache --> skip
    train_cache --> scheduler
    test_cache --> scheduler
    skip --> reports
    scheduler --> reports
    guide -. explains .-> scheduler

    classDef data fill:#dbeafe,stroke:#2563eb,color:#172554,stroke-width:1.5px;
    classDef model fill:#ede9fe,stroke:#7c3aed,color:#2e1065,stroke-width:1.5px;
    classDef artifact fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:1.5px;
    classDef contract fill:#f0fdf4,stroke:#15803d,color:#14532d,stroke-dasharray:5 3;
    classDef notebook fill:#fef3c7,stroke:#d97706,color:#451a03,stroke-width:1.5px;
    classDef logic fill:#ffedd5,stroke:#ea580c,color:#431407,stroke-width:1.5px;
    classDef output fill:#f3f4f6,stroke:#4b5563,color:#111827,stroke-width:1.5px;
    classDef doc fill:#fce7f3,stroke:#db2777,color:#500724,stroke-width:1.5px;
```

### How to read it

- `dynamic_rf_IDK_Cascade.ipynb` is the data preparation and original cascade
  notebook. It streams data, loads pretrained models, creates caches, trains
  the RF skipper, and evaluates skip strategies.
- `dynamic_parallel_IDK_scheduler.ipynb` consumes the caches. It trains a
  second Random Forest for choosing a heavy model and runs scheduling
  simulations.
- `dataset.py` provides the streaming portion as a small reusable Python
  helper; the research notebook also contains its own streaming functions.
- Once caches exist, routing and scheduler experiments do not need raw images
  or live neural-network inference.

## 2. Offline cache and training pipeline

![Offline cache and training pipeline](docs/architecture/offline-training-pipeline.svg)

```mermaid
%%{init: {"theme": "neutral", "flowchart": {"curve": "basis", "htmlLabels": true}}}%%
flowchart TB
    subgraph datasets["ImageNet-V2 variants"]
        train["Training variants<br/>matched-frequency + top-images"]:::data
        test["Test variant<br/>threshold-0.7"]:::data
    end

    subgraph generation["Offline cache generation · dynamic_rf_IDK_Cascade.ipynb"]
        stream["stream_imagenet_v2_rows<br/>image · label · key"]:::logic
        loader["PyTorch streaming loader<br/>model-specific transforms"]:::logic
        zoo["Pretrained models<br/>ResNet-18 · 34 · 50 · 152"]:::model
        infer["cache_logits<br/>batched inference + timing"]:::logic
        cache[(".npz caches by variant and model<br/>probabilities · labels · predictions<br/>times_ms · keys")]:::artifact
    end

    train --> stream
    test --> stream
    stream --> loader
    zoo --> infer
    loader --> infer
    infer --> cache

    subgraph skip_path["Original RF skip experiment"]
        skip_train["Matched + Top<br/>ResNet-18 and ResNet-34 caches"]:::artifact
        skip_features["ResNet-18 features<br/>confidence · entropy · margin"]:::feature
        skip_labels["Target<br/>skip ResNet-34 or run ResNet-34"]:::decision
        skipper["RandomForestClassifier<br/>RF skipper"]:::model
        cascade["Threshold-0.7 cascade evaluation<br/>18 → optional 34 → 152"]:::logic
        skip_results["Accuracy · time/image · exits<br/>skip-decision metrics"]:::output
    end

    cache --> skip_train
    skip_train --> skip_features
    skip_train --> skip_labels
    skip_features --> skipper
    skip_labels --> skipper
    skipper --> cascade
    cache --> cascade
    cascade --> skip_results

    subgraph heavy_path["Heavy-router and scheduler experiment · dynamic_parallel_IDK_scheduler.ipynb"]
        heavy_train["Matched + Top<br/>all four model caches"]:::artifact
        double_idk["Keep rows where<br/>ResNet-18 and ResNet-34 are both IDK"]:::decision
        heavy_features["Six early-model features<br/>confidence · entropy · margin<br/>from ResNet-18 and ResNet-34"]:::feature
        route_labels["Route target<br/>ResNet-50 if it qualifies<br/>otherwise ResNet-152"]:::decision
        router["RandomForestClassifier<br/>heavy router"]:::model
        test_eval["Threshold-0.7 evaluation<br/>router + sequential baseline<br/>+ dynamic scheduler"]:::logic
        dynamic_results["Accuracy · latency · throughput<br/>utilization · cancellations · wasted work"]:::output
    end

    cache --> heavy_train
    heavy_train --> double_idk
    double_idk --> heavy_features
    double_idk --> route_labels
    heavy_features --> router
    route_labels --> router
    router --> test_eval
    cache --> test_eval
    test_eval --> dynamic_results

    subgraph legend["Legend"]
        l_data["Dataset"]:::data
        l_model["Trained or pretrained model"]:::model
        l_cache[("Cached arrays")]:::artifact
        l_feature["Derived features"]:::feature
        l_decision["Selection / target rule"]:::decision
        l_output["Experiment metrics"]:::output
    end

    classDef data fill:#dbeafe,stroke:#2563eb,color:#172554,stroke-width:1.5px;
    classDef model fill:#ede9fe,stroke:#7c3aed,color:#2e1065,stroke-width:1.5px;
    classDef artifact fill:#dcfce7,stroke:#16a34a,color:#14532d,stroke-width:1.5px;
    classDef logic fill:#fef3c7,stroke:#d97706,color:#451a03,stroke-width:1.5px;
    classDef feature fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e,stroke-width:1.5px;
    classDef decision fill:#ffedd5,stroke:#ea580c,color:#431407,stroke-width:1.5px;
    classDef output fill:#f3f4f6,stroke:#4b5563,color:#111827,stroke-width:1.5px;
```

### Cache contract

Every model cache must contain aligned arrays:

| Field | Meaning |
| --- | --- |
| `probabilities` | Shape `[num_images, 1000]`; model probability distribution |
| `labels` | Ground-truth ImageNet class for every image |
| `predictions` | Model top-1 class for every image |
| `times_ms` | Cached inference time per image in milliseconds |
| `keys` | Image identifiers used to verify ordering across model caches |

The cache validator in the scheduler notebook requires all four model caches to
contain the same sample count, labels, and keys.

### Training and test split

- **Router training:** matched-frequency plus top-images.
- **Reported test experiments:** threshold-0.7.
- **Original RF skipper:** learns whether ResNet-34 should run after
  ResNet-18 says IDK.
- **Heavy RF router:** learns whether a double-IDK image should go to
  ResNet-50 or ResNet-152.

## 3. Dynamic worker-reuse scheduler

![Dynamic worker-reuse scheduler](docs/architecture/dynamic-scheduler.svg)

```mermaid
%%{init: {"theme": "neutral", "sequence": {"useMaxWidth": true, "wrap": true, "diagramMarginX": 24, "actorMargin": 36}}}%%
sequenceDiagram
    autonumber
    participant S as Scheduler
    participant R18 as ResNet-18 worker
    participant Q34 as ResNet-34 queue
    participant R34 as ResNet-34 worker
    participant H as Completion min-heap
    participant RF as Heavy RF router
    participant QH as Selected heavy FIFO queue
    participant R50 as ResNet-50 worker
    participant R152 as ResNet-152 worker
    participant M as Metrics

    Note over S,M: Cached-output discrete-event simulation: models do not execute live or share hardware here
    Note over S,H: Inputs are aligned threshold-0.7 caches plus max_in_flight backpressure

    loop Until all images are finalized and all running events are drained
        Note over S: Admit only when ResNet-18 is free and unfinished images are below max_in_flight
        S->>R18: Start next image using cached times_ms
        S->>Q34: Speculatively enqueue the same image

        alt ResNet-34 worker is free and image is unfinished
            Q34->>R34: Dispatch queued image
        else Image was finalized before queue dispatch
            Q34-->>M: Increment canceled_jobs[ResNet-34]
        end

        R18-->>H: Push completion event
        R34-->>H: Push completion event if job started
        H-->>S: Pop every event at the earliest timestamp

        alt Completion belongs to an already-finalized image
            S-->>M: Increment wasted_completions[model]
        else ResNet-18 completed with confidence ≥ threshold
            S->>M: Finalize with ResNet-18 prediction
        else ResNet-34 completed with confidence ≥ threshold
            S->>M: Finalize with ResNet-34 prediction
        else Both early workers completed as IDK
            S->>RF: Predict from 18 + 34 confidence, entropy, and margin
            alt Router class 0
                RF-->>S: Route to ResNet-50
                S->>QH: Enqueue for ResNet-50
                QH->>R50: Start when worker is free
                R50-->>H: Push heavy completion event
            else Router class 1
                RF-->>S: Route to ResNet-152
                S->>QH: Enqueue for ResNet-152
                QH->>R152: Start when worker is free
                R152-->>H: Push heavy completion event
            end
            H-->>S: Pop selected heavy-model completion
            S->>M: Finalize with heavy-model prediction
        else Only one early IDK result is available
            Note over S,H: Keep the image unfinished and wait for the other early completion
        end
    end

    S->>M: Calculate accuracy, makespan, FPS, latency, exits, executions, utilization, routes, cancellations, and wasted completions
```

### Scheduler behavior

1. Admission starts ResNet-18 and immediately queues the same image for
   ResNet-34. This speculative overlap can improve throughput.
2. Every model has one independent simulated worker. Completion events are
   ordered by finish time in a min-heap.
3. A confident ResNet-18 or ResNet-34 result finalizes the image.
4. If both early models return IDK, the heavy router selects ResNet-50 or
   ResNet-152 from six early-model features.
5. `max_in_flight` limits admitted but unfinished images. A small value applies
   backpressure and reduces queueing latency.
6. Work still waiting in a queue can be canceled after another model finalizes
   the image. Work already running cannot be interrupted and is counted as a
   wasted completion.

> **Important:** this scheduler is a deterministic cached-output
> discrete-event simulation. It models independent workers using stored
> per-image inference times; it does not execute four neural networks
> concurrently on hardware. Queueing logic is modeled, while model contention,
> data transfer, router inference time, and orchestration overhead are not.

## 4. Reading the reported metrics

- **Accuracy:** fraction of final predictions equal to ground-truth labels.
- **Makespan:** simulated time from the first admission to the last processed
  completion event.
- **Throughput:** images divided by makespan in seconds.
- **Latency:** time from one image's admission until its final prediction.
- **Utilization:** one worker's total busy time divided by makespan.
- **Canceled job:** queued speculative work removed before it starts.
- **Wasted completion:** work that had already started and completed after the
  image was finalized elsewhere.

Higher `max_in_flight` values can keep the ResNet-152 bottleneck busy and
maximize throughput, but they also allow a longer queue to form. That is why
the unlimited run can have similar throughput and dramatically worse latency
than a bounded run.

## Diagram sources

The editable Mermaid sources are:

- [`project-overview.mmd`](docs/architecture/project-overview.mmd)
- [`offline-training-pipeline.mmd`](docs/architecture/offline-training-pipeline.mmd)
- [`dynamic-scheduler.mmd`](docs/architecture/dynamic-scheduler.mmd)

The SVG files are generated from those sources and can be used in documents or
presentations that do not render Mermaid directly.

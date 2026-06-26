# Dynamic Parallel IDK Scheduler Study Guide

This guide explains `dynamic_parallel_IDK_scheduler.ipynb` in cell order.
Line numbers refer to the source lines inside each notebook cell, not physical
lines in the `.ipynb` JSON file.

## 1. What the notebook is trying to do

The notebook does not run neural networks directly. It loads previously saved
outputs for four models:

- ResNet-18
- ResNet-34
- ResNet-50
- ResNet-152

Each cache contains predictions, class probabilities, labels, image keys, and
measured inference times. The notebook then:

1. Trains a random forest to choose ResNet-50 or ResNet-152 when both early
   models say "IDK."
2. Evaluates that router.
3. Simulates a sequential cascade.
4. Simulates four independent model workers processing images concurrently.
5. Compares throughput, latency, accuracy, utilization, and wasted work.

An early model says IDK when its maximum class probability is below
`THRESHOLD`.

## 2. End-to-end data flow

```text
cached .npz files
       |
       v
load_cache / load_caches / validate_caches
       |
       +--> confidence and probability features
       |        [confidence, entropy, top-two margin]
       |
       +--> train heavy router on double-IDK examples
       |
       +--> sequential baseline:
       |        ResNet-18 -> maybe ResNet-34 -> maybe ResNet-152
       |
       +--> dynamic simulation:
                ResNet-18 and ResNet-34 run speculatively
                -> if both IDK, RF selects ResNet-50 or ResNet-152
```

## Cell 1: Imports and constants

### Line 1

`# Cell 1: Imports and constants`

A comment naming the cell's purpose. It has no runtime effect.

### Line 2

`import heapq`

Imports Python's min-heap implementation. The scheduler stores future
completion events in a heap so the event with the earliest timestamp is always
removed first.

### Line 3

`import numpy as np`

Imports NumPy under its standard alias. NumPy is used for cached arrays,
feature calculations, masks, metrics, and assertions.

### Line 4

`from collections import deque, Counter`

- `deque` supplies efficient FIFO queues for waiting model jobs.
- `Counter` stores execution counts, busy times, cancellations, and routes.

### Line 5

`from sklearn.ensemble import RandomForestClassifier`

Imports the scikit-learn model used to route double-IDK images to a heavy
network.

### Lines 7-10

```python
RESNET18 = "resnet18"
RESNET34 = "resnet34"
RESNET50 = "resnet50"
RESNET152 = "resnet152"
```

Defines canonical string keys. These strings index cache dictionaries, worker
state, queues, and result dictionaries. Constants reduce spelling mistakes and
make later comparisons clearer.

### Lines 12-13

```python
ROUTE_RESNET50 = 0
ROUTE_RESNET152 = 1
```

Defines the two class labels learned by the random forest:

- Class `0` means send the image to ResNet-50.
- Class `1` means send the image to ResNet-152.

### Line 14

`THRESHOLD = 0.65`

An early or heavy model is treated as confident when its maximum predicted
probability is at least `0.65`. Despite the test artifact names containing
`threshold07`, this notebook's runtime threshold is `0.65`.

### Line 16

The saved notebook currently contains:

```python
MODELS = (RESNET18, RESNET34, RESNET50, RESNET152)HEAVY_MODELS = (RESNET50, RESNET152)
```

This is a syntax error because two assignments were accidentally joined. The
intended code is:

```python
MODELS = (RESNET18, RESNET34, RESNET50, RESNET152)
HEAVY_MODELS = (RESNET50, RESNET152)
```

`MODELS` represents all workers. `HEAVY_MODELS` represents only the models the
random forest can select. Stored outputs exist because an earlier notebook
kernel already had these names defined, but a clean "Run All" will fail here.

## Cell 2: Cache-loading helpers

### Lines 1-8

```python
REQUIRED_CACHE_FIELDS = (
    "probabilities",
    "labels",
    "predictions",
    "times_ms",
    "keys",
)
```

Defines the required schema of every model cache:

- `probabilities`: one class-probability vector per image.
- `labels`: ground-truth class IDs.
- `predictions`: the model's chosen class IDs.
- `times_ms`: inference time per image in milliseconds.
- `keys`: stable image identifiers used to verify alignment.

### Line 11

`def load_cache(path):`

Defines a function that loads one `.npz` cache file.

### Line 12

`with np.load(path, allow_pickle=False) as data:`

Opens the NumPy archive as a context manager. `allow_pickle=False` avoids
loading arbitrary pickled Python objects and expects ordinary NumPy arrays.

### Line 13

`return {name: data[name] for name in data.files}`

Copies every named array from the archive into a regular dictionary. Reading
the arrays before leaving the `with` block is necessary because the archive is
closed afterward.

### Line 16

`def load_caches(paths):`

Defines a function for loading all model caches for one dataset. `paths` is
expected to map model names to `.npz` paths.

### Line 17

`caches = {model: load_cache(path) for model, path in paths.items()}`

Loads each model's file and builds this nested structure:

```python
caches[model_name][field_name]
```

### Line 18

`validate_caches(caches)`

Checks the schema and verifies that all models refer to the same images in the
same order.

### Line 19

`return caches`

Returns the validated nested dictionary.

### Line 22

`def validate_caches(caches):`

Defines the central cache consistency checker.

### Line 23

`missing_models = [model for model in MODELS if model not in caches]`

Builds a list of required model names that are absent.

### Lines 24-25

```python
if missing_models:
    raise ValueError(f"Missing model caches: {missing_models}")
```

Stops immediately if any of the four model caches is missing.

### Line 27

`for model, cache in caches.items():`

Visits each model cache.

### Line 28

`missing = [name for name in REQUIRED_CACHE_FIELDS if name not in cache]`

Finds required arrays missing from the current model's cache.

### Lines 29-30

Raises a model-specific `ValueError` if the cache schema is incomplete.

### Line 32

`reference = caches[RESNET18]`

Uses ResNet-18 as the reference cache for labels, keys, and sample count.

### Line 33

`sample_count = len(reference["labels"])`

Gets the expected number of images.

### Lines 34-35

Rejects an empty dataset. Later means, percentiles, and throughput calculations
assume at least one sample.

### Line 37

Starts a second pass over every model cache.

### Line 38

`lengths = [len(cache[name]) for name in REQUIRED_CACHE_FIELDS]`

Collects the first-dimension length of each required array.

### Lines 39-40

Checks that every required array has exactly the ResNet-18 sample count.

### Lines 41-42

Uses exact array equality to ensure every model cache has the same ground-truth
labels in the same order.

### Lines 43-44

Uses exact array equality to ensure every cache has the same image keys in the
same order. This prevents combining predictions or timings from different
images.

### Line 46

`return sample_count`

Returns the validated sample count so callers do not need to calculate it
again.

## Cell 3: Feature helpers

### Line 1

Names the cell.

### Line 2

`def confidence(cache):`

Defines a helper that calculates one confidence score per cached image.

### Line 3

`return np.asarray(cache["probabilities"]).max(axis=1)`

Converts probabilities to a NumPy array and takes the largest class probability
in each row. For shape `[N, C]`, the result has shape `[N]`.

### Line 6

`def features_from_probs(probs):`

Defines the feature extractor used by the random forest.

### Line 7

Normalizes the input to a NumPy array.

### Lines 8-9

Requires a two-dimensional `[number of samples, number of classes]` array with
at least two classes. The top-two margin cannot be calculated with one class.

### Line 11

`conf = probs.max(axis=1)`

Calculates confidence: the probability of the model's most likely class.

### Line 12

`entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)`

Calculates Shannon entropy for each prediction distribution:

```text
entropy = -sum(p * log(p))
```

The small `1e-12` prevents `log(0)`. Lower entropy usually means the model is
more certain; higher entropy means probability mass is spread across classes.

### Line 13

`top_two = np.partition(probs, -2, axis=1)[:, -2:]`

Finds the two largest probabilities in each row without fully sorting all
classes. Their internal order is not guaranteed.

### Line 14

`margin = top_two.max(axis=1) - top_two.min(axis=1)`

Calculates the gap between the highest and second-highest probabilities. A
large margin generally indicates a clearer decision.

### Line 15

```python
return np.column_stack([conf, entropy, margin]).astype(np.float32)
```

Returns an `[N, 3]` feature matrix in the order:

```text
[confidence, entropy, margin]
```

The values are converted to 32-bit floats to reduce memory use.

## Cell 4: Simplified heavy-router training

### Line 2

```python
def _heavy_router_rows(caches, threshold=THRESHOLD, require_correct=False):
```

Defines an internal data-preparation function. It selects double-IDK examples
and creates random-forest features and labels. The leading underscore signals
that it is intended as a private helper.

### Line 3

Validates cache alignment before combining data from multiple models.

### Lines 5-6

Extracts three features each from ResNet-18 and ResNet-34 probabilities.

### Line 7

Reads ground-truth labels from the validated ResNet-18 cache.

### Lines 9-12

Creates a Boolean mask named `double_idk`. It is true only where both early
models have confidence below the threshold.

### Line 14

Marks images where ResNet-50's confidence reaches the threshold.

### Line 15

Marks images where ResNet-152's confidence reaches the threshold.

### Lines 16-18

When `require_correct=True`, confidence is not enough: the corresponding heavy
model must also predict the ground-truth label. With the default `False`,
"can50" and "can152" really mean "is confident," not "is correct."

### Line 20

```python
included = double_idk & (can50 | can152)
```

Keeps double-IDK examples for which at least one heavy model is confident (and,
optionally, correct).

### Line 21

Combines the selected ResNet-18 and ResNet-34 features. Each row has six
features:

```text
[conf18, entropy18, margin18, conf34, entropy34, margin34]
```

### Lines 22-26

Creates the target route:

- Choose ResNet-50 whenever `can50` is true.
- Otherwise choose ResNet-152.

This deliberately gives ResNet-50 priority when both heavy models qualify,
presumably because ResNet-50 is faster.

### Line 28

Counts all double-IDK examples, including ones unusable for router training.

### Line 29

Counts excluded examples where neither heavy model qualifies.

### Line 30

Returns features, route labels, the total double-IDK count, and exclusion count.

### Line 33

Defines training across a list of cache groups. Here the list contains the
matched-frequency and top-images ImageNet-V2 variants.

### Lines 34-37

Initializes lists for feature/label blocks and counters for reporting.

### Line 39

Loops over each training dataset's four-model cache collection.

### Lines 40-44

Calls `_heavy_router_rows` with the same threshold and correctness policy.

### Lines 45-46

Accumulates double-IDK and excluded counts across datasets.

### Lines 47-49

Appends only non-empty feature and target blocks. This avoids concatenating
empty datasets unnecessarily.

### Lines 51-52

Raises a clear error if no training dataset produced a usable route label.

### Lines 54-55

Vertically concatenates all selected rows into one training matrix and target
vector.

### Lines 56-57

Counts target labels for each heavy route.

### Lines 59-63

Prints dataset diagnostics. In the stored run:

- 5,559 training images were double-IDK.
- 1,741 were usable for router training.
- 144 were labeled for ResNet-50.
- 1,597 were labeled for ResNet-152.
- 3,818 were excluded.

### Lines 65-66

Checks two accounting invariants:

- Both route counts must add up to the training row count.
- Included plus excluded rows must equal all double-IDK rows.

Python assertions are useful for development, but can be disabled with
`python -O`; they are not a replacement for user-facing validation.

### Lines 68-74

Creates the random forest:

- `n_estimators=100`: use 100 decision trees.
- `max_depth=6`: limit tree complexity.
- `min_samples_leaf=20`: require at least 20 examples per leaf.
- `class_weight="balanced"`: upweight the rare ResNet-50 route.
- `random_state=42`: make training reproducible.

### Line 75

Fits the random forest on the six early-model features and route labels.

### Line 76

Returns the trained router.

## Cell 5: Heavy-router evaluation

### Line 2

Defines router evaluation on one dataset.

### Lines 3-7

Builds evaluation features and expected route labels using exactly the same
selection logic as training.

### Lines 9-11

If usable rows exist, predicts their routes and computes mean classification
accuracy.

### Lines 12-14

If no rows exist, creates an empty prediction vector and reports accuracy as
NaN instead of dividing by zero.

### Line 16

Creates a 2-by-2 integer confusion matrix initialized to zero.

### Lines 17-18

Efficiently increments `confusion[true_route, predicted_route]` once per
example. Rows are true classes and columns are predicted classes.

### Lines 20-23

Counts true route labels for ResNet-50 and ResNet-152.

### Lines 24-27

Counts routes predicted by the random forest.

### Lines 29-36

Prints evaluation diagnostics and the confusion matrix.

The stored result was:

```text
[[ 57  29]
 [175 589]]
```

Therefore:

- 57 true ResNet-50 routes were predicted correctly.
- 29 true ResNet-50 routes were sent to ResNet-152.
- 175 true ResNet-152 routes were incorrectly sent to ResNet-50.
- 589 true ResNet-152 routes were predicted correctly.

The reported `0.76` accuracy is conditional on the 850 evaluated rows. It does
not include the 1,810 double-IDK examples excluded because neither heavy model
met the confidence rule.

### Lines 38-46

Returns all evaluation metrics in a dictionary for later inspection or
reporting.

## Cell 6: Sequential cached-output baseline

### Line 2

Defines a non-concurrent reference cascade.

### Line 3

Validates caches and gets the number of samples.

### Line 4

Reads ground-truth labels.

### Lines 6-7

Precomputes ResNet-18 and ResNet-34 confidence vectors.

### Line 8

Allocates the final predicted class for every image.

### Line 9

Allocates the model name that supplied each final prediction.

### Line 10

Initializes each image's simulated latency to zero.

### Line 12

Processes images one at a time.

### Line 13

Adds the cached ResNet-18 inference time because every image starts there.

### Lines 14-17

If ResNet-18 is confident, use its prediction and model name, then `continue`
to the next image.

### Line 19

If ResNet-18 said IDK, add ResNet-34's inference time.

### Lines 20-23

If ResNet-34 is confident, use its prediction and skip the heavy model.

### Lines 25-27

If both early models say IDK, always run and use ResNet-152. This baseline does
not use the random-forest router or ResNet-50.

### Line 29

Treats the sum of all per-image sequential latencies as total makespan.

### Line 30

Calculates top-1 accuracy against ground truth.

### Lines 31-33

Counts how many final predictions came from each model. ResNet-50's count will
always be zero in this baseline.

### Lines 35-44

Builds the result dictionary:

- `samples`: number of images.
- `accuracy`: fraction correctly classified.
- `makespan_ms`: total serial runtime.
- `mean_latency_ms`: average per-image runtime.
- `median_latency_ms`: 50th percentile.
- `p95_latency_ms`: 95th percentile.
- `throughput_fps`: images per second.
- `final_model_counts`: exits by model.

### Lines 46-47

Checks that every image has one final model and accuracy is a valid fraction.

### Line 48

Returns the baseline result.

## Cell 7: RF skip baseline

### Lines 1-3

This cell contains comments only. It explicitly says the older random-forest
skip baseline was removed. The notebook compares:

- Sequential `18 -> 34 -> 152`.
- Dynamic speculative early workers plus a learned `50/152` route.

No code executes in this cell.

## Cell 8: Dynamic worker-reuse simulation

This is the core of the notebook. It is a discrete-event simulation: no model
actually runs here. Cached per-image timings determine when simulated jobs
finish.

### Line 2

Defines the simulation. `max_in_flight` optionally limits the number of admitted
images that have not yet received a final prediction.

### Line 3

Validates caches and gets the sample count.

### Lines 4-7

Validates `max_in_flight`:

- `None` means no backlog limit.
- Otherwise it must be a positive Python or NumPy integer.
- It is normalized to a Python `int`.

Booleans technically pass `isinstance(True, int)`, so `True` would act like
`1`; that edge case is not rejected.

### Lines 9-13

Loads labels, extracts early-model features, and takes feature column zero as
confidence.

### Lines 15-17

Combines all six early features and predicts all heavy routes once. Only
double-IDK images later use these predictions. Router computation time is not
included in simulated latency or makespan.

### Lines 19-22

Allocates per-image state:

- `start_times`: admission time.
- `end_times`: finalization time.
- `final_predictions`: chosen class, initially `-1`.
- `final_models`: chosen model, initially `None`.

### Lines 23-26

Creates Boolean arrays recording whether each early model has completed for
each image.

### Lines 27-30

Creates Boolean arrays recording whether each completed early result was IDK.

### Line 31

Tracks whether a heavy job has already been queued for an image, preventing
duplicate routes.

### Line 33

Creates one worker slot per model. `None` means idle; otherwise the value is the
index of the image currently running.

### Lines 34-38

Creates FIFO queues for ResNet-34, ResNet-50, and ResNet-152. ResNet-18 has no
queue because new images are admitted directly to its single worker.

### Line 39

Initializes the event heap. An event represents a model completion.

### Line 40

Initializes a monotonically increasing event number. It breaks ties when two
events have identical completion times.

### Line 41

`next_image` is the index of the next image not yet admitted.

### Line 42

`unfinished` counts admitted images that do not yet have final predictions.
This is the value constrained by `max_in_flight`.

### Line 43

`finalized` counts completed images.

### Line 44

Initializes simulated wall-clock time.

### Lines 46-50

Initializes counters for:

- jobs started per model;
- total busy milliseconds per model;
- queued jobs canceled before starting;
- running jobs that finish after another model finalized the image;
- heavy routes selected by the random forest.

### Line 52

Defines a nested helper to start one model job.

### Line 53

Allows the helper to increment `event_number` from the enclosing function.

### Line 54

Reads this model/image pair's cached inference duration.

### Line 55

Marks the model worker busy with this image.

### Lines 56-57

Records one execution and adds its full duration to model busy time.

### Line 58

Creates a unique event sequence number.

### Lines 59-62

Pushes this completion event into the min-heap:

```text
(finish time, tie breaker, model, image index)
```

### Line 64

Defines a helper that starts the next valid queued job for one model.

### Lines 65-66

Does nothing if the worker is already busy.

### Line 67

Keeps examining queued jobs until one starts or the queue becomes empty.

### Line 68

Removes the oldest queued image.

### Lines 69-71

If another model already finalized that image, count the queued job as canceled
and examine the next queue item.

### Lines 72-73

Otherwise start the job and return because this worker is now occupied.

### Line 75

Defines the admission helper for the pipeline's first worker.

### Line 76

Allows updates to `next_image` and `unfinished`.

### Lines 77-78

Cannot admit an image while ResNet-18 is busy or after all images are admitted.

### Lines 79-80

Stops admission when the unfinished-image limit has been reached.

### Lines 82-84

Selects the next image, advances the admission pointer, and increments the
number of unfinished images.

### Line 85

Records the image's latency start time.

### Line 86

Starts ResNet-18 immediately.

### Line 87

Speculatively queues the same image for ResNet-34 immediately, before knowing
whether ResNet-18 will be confident. This overlap is the key source of extra
throughput and wasted work.

### Line 89

Defines a helper that accepts one model's result as final.

### Line 90

Allows updates to `unfinished` and `finalized`.

### Lines 91-92

Makes finalization idempotent: a later result cannot overwrite an already
accepted result.

### Lines 93-95

Stores the final model, prediction, and completion time.

### Lines 96-97

Removes the image from the unfinished count and increments completed images.

### Lines 99-100

At time zero:

1. Admit image zero to ResNet-18.
2. Start its already-queued speculative ResNet-34 job if that worker is free.

### Line 102

Continues while some image is not finalized or a running completion event
still exists. Remaining events are processed even after all images are
finalized so wasted completions can be counted.

### Lines 103-104

If work remains but no future event exists, scheduler state is inconsistent,
so raise an error instead of looping forever.

### Line 106

Advances simulated time to the earliest completion timestamp.

### Line 107

Starts a list for all events completing at exactly this time.

### Lines 108-109

Pops every simultaneous completion. Processing them as a batch reduces
arbitrary behavior caused by event insertion order.

### Line 111

Creates a mapping from image index to models that completed that image now.

### Line 112

Iterates through the completion batch. The first two tuple fields are ignored.

### Lines 113-114

Checks that the event belongs to the job currently recorded on that worker.

### Line 115

Marks the worker idle.

### Lines 117-119

If this image was already finalized, the just-finished running job was wasted.
Count it and do not process its prediction.

### Line 121

Groups the model completion under its image.

### Lines 122-125

For early models, records completion and whether confidence was below the
threshold.

### Lines 127-128

Begins resolution only after all completions at this timestamp have updated
state.

### Line 129

Converts the model list to a set for membership tests.

### Lines 131-133

If ResNet-18 completed confidently, finalize with ResNet-18. It has first
priority among simultaneous results.

### Lines 134-136

Otherwise, if ResNet-34 completed confidently, finalize with ResNet-34.

### Lines 137-139

Otherwise accept a completed ResNet-50 heavy result.

### Lines 140-142

Otherwise accept a completed ResNet-152 heavy result.

The resolution priority for exact ties is therefore:

```text
ResNet-18 > ResNet-34 > ResNet-50 > ResNet-152
```

### Lines 144-149

Checks whether both early workers have completed and both said IDK.

### Line 150

Only route the image if both early models said IDK and no heavy job was already
queued.

### Line 151

Reads the random forest's precomputed route.

### Lines 152-155

Maps class `0` to ResNet-50 and class `1` to ResNet-152.

### Lines 156-157

Rejects any unexpected router class.

### Lines 159-161

Marks the image routed, records route statistics, and appends the image to the
selected heavy worker's queue.

### Lines 163-165

As soon as ResNet-18 becomes free, attempts to admit one new image, subject to
the unfinished-image limit.

### Lines 166-168

Attempts to start one waiting job on each now-idle downstream worker.

### Line 170

Calculates each image's end-to-end latency from admission until finalization.

### Line 171

Calculates final top-1 accuracy.

### Line 172

The last processed event time is the simulated makespan.

### Line 173

Converts makespan into images per second.

### Lines 175-177

Counts final exits by model.

### Lines 178-180

Converts model execution counters into a complete dictionary containing all
four models, including zero counts.

### Lines 181-183

Calculates each worker's utilization:

```text
worker busy time / total makespan
```

Utilizations should be interpreted independently. Their sum may exceed 1
because four workers can run simultaneously.

### Lines 184-186

Builds route counts for only ResNet-50 and ResNet-152.

### Lines 187-189

Builds queued-job cancellation counts for all workers.

### Lines 190-192

Builds wasted-running-completion counts for all workers.

### Lines 194-208

Packages summary metrics and scheduler diagnostics into the result dictionary.

### Lines 210-211

Checks final-model accounting and accuracy range.

### Lines 212-215

Recalculates throughput and checks that the stored value agrees.

### Lines 216-220

Recalculates every utilization and checks that the stored value agrees.

### Line 222

Returns the dynamic simulation result.

## Cell 9: Run the cached-output experiment

### Lines 2-15

Defines two training cache groups:

- `matched_*` files for the matched-frequency variant.
- `top_*` files for the top-images variant.

Each group has aligned outputs from all four models.

### Lines 17-22

Defines the four test cache paths for the threshold-0.7 ImageNet-V2 variant.
Again, this file naming does not override `THRESHOLD = 0.65`.

### Line 24

Loads and validates both training cache groups.

### Line 25

Trains the heavy random-forest router using the default threshold and
`require_correct=False`.

### Line 27

Loads and validates the test caches.

### Line 28

Evaluates the router on usable double-IDK test rows.

### Line 29

Runs the sequential `18 -> 34 -> 152` baseline.

### Line 31

Creates a dictionary to hold dynamic results keyed by in-flight limit.

### Line 32

Runs seven scheduler configurations:

```text
unlimited, 2, 3, 4, 8, 16, 32
```

### Line 33

Creates a readable label for logging.

### Line 34

Prints the configuration about to run.

### Lines 35-39

Runs the simulator and stores its result under the corresponding limit.

## Cell 10: Comparison table and dynamic-run details

### Lines 3-12

Creates ordered `(display name, result)` pairs for the sequential baseline and
all dynamic configurations.

The label `"Sequential Dynamic RF baseline"` is misleading: that baseline does
not use the RF router and is not dynamic.

### Lines 14-17

Prints a fixed-width table header. Alignment operators mean:

- `<34`: left-align in 34 characters.
- `>8`, `>16`, and so on: right-align in the given width.

### Line 18

Loops over each comparison row.

### Lines 19-26

Prints accuracy to three decimals and timing metrics to two decimals.

### Line 28

Loops through the dynamic results in insertion order.

### Line 29

Builds the readable limit label again.

### Lines 30-36

Prints detailed diagnostics for each dynamic run:

- final prediction source;
- executions;
- worker utilization;
- heavy routes;
- jobs canceled while queued;
- jobs that finished after becoming unnecessary.

### Interpreting the stored table

The saved 10,000-image output shows:

- Sequential throughput: about `9.12 FPS`.
- Dynamic throughput with `max_in_flight=3`: about `17.09 FPS`.
- Dynamic accuracy: about `0.762`, close to the sequential `0.763`.
- Unlimited throughput: about `17.19 FPS`, but mean latency grows to roughly
  `39,961 ms`.

Unlimited admission lets images accumulate behind the almost fully utilized
ResNet-152 worker. Throughput reaches its bottleneck limit, but individual
images can wait a very long time. A low `max_in_flight` applies backpressure and
trades a small amount of throughput for much lower latency.

For `max_in_flight=3`, stored utilization is approximately:

```text
ResNet-18:  29.5%
ResNet-34:  43.8%
ResNet-50:  12.3%
ResNet-152: 99.3%
```

ResNet-152 is the bottleneck.

## Cell 11: First-N comparison

### Line 1

Defines a helper to slice caches to the first `n` aligned images. The comment
says 2,000 images, but the actual experiment later uses 5,000.

### Line 2

Validates the original caches and gets their sample count.

### Line 3

Caps `n` at the available sample count. It does not reject zero or negative
values, so input validation could be stronger.

### Line 5

Creates the output cache dictionary.

### Line 7

Loops through all model caches.

### Line 8

Creates the current model's nested dictionary.

### Line 10

Loops through every key/value pair in that model's cache.

### Lines 11-12

Slices required per-image arrays to their first `n` entries.

### Lines 13-14

Copies any non-required metadata without slicing.

### Line 16

Validates that the newly sliced caches remain aligned.

### Line 17

Returns the sliced cache collection.

### Line 20

`n_bulk = 5000;`

Chooses 5,000 images. The semicolon is legal but unnecessary in Python.

### Line 21

`mif=3`

Chooses a maximum of three unfinished images for the dynamic run. PEP 8 style
would write `mif = 3`.

### Lines 24-27

Runs the sequential baseline on the first 5,000 test images.

### Lines 28-33

Runs the dynamic simulation on the same 5,000 images using the trained router
and `max_in_flight=3`.

### Lines 35-40

Prints sequential accuracy, total seconds, throughput, and mean latency.

### Lines 41-46

Prints the same dynamic metrics plus the in-flight limit.

The stored result was:

```text
Sequential:
  accuracy       0.7648
  total time   551.0122 s
  throughput     9.0742 FPS
  mean latency 110.2024 ms

Dynamic:
  accuracy       0.7632
  total time   290.8787 s
  throughput    17.1893 FPS
  mean latency 172.3475 ms
```

The dynamic design nearly doubles throughput, but mean per-image latency is
higher because images may wait in downstream queues.

## Cells 12 and 13

Both cells are empty. They define no values and perform no work.

## 3. Important modeling assumptions

The results are from a simulator, not a live parallel deployment:

1. Each model has exactly one independent worker.
2. Cached single-image inference times are treated as deterministic durations.
3. Simultaneous workers do not slow one another down through GPU, CPU, memory,
   or I/O contention.
4. Queueing, random-forest inference, data transfer, and orchestration overhead
   take zero simulated time.
5. A running speculative job cannot be interrupted; it becomes a wasted
   completion if another result wins first.
6. Router accuracy is measured only on double-IDK rows where at least one heavy
   model passes the route-label rule.

These assumptions make the notebook useful for studying scheduling policy, but
its FPS values should not be treated as measured production throughput.

## 4. Core concepts to remember

### IDK decision

```python
confidence < threshold
```

Confidence is the largest predicted class probability.

### Router features

Each early model contributes:

```text
confidence, entropy, top-two margin
```

Together, ResNet-18 and ResNet-34 provide six features.

### Router target

```text
use ResNet-50 if it qualifies;
otherwise use ResNet-152
```

### Sequential behavior

```text
18 confident? use 18
else 34 confident? use 34
else use 152
```

### Dynamic behavior

```text
start 18 and queue 34 speculatively
accept a confident early result
if both are IDK, route to 50 or 152
```

### Throughput versus latency

More images in flight keep the heavy worker busy and improve throughput, but
increase queueing delay. `max_in_flight` is the backpressure control that
balances those objectives.

## 5. Study questions

1. Why does the router need entropy and margin if confidence is already known?
2. Why is ResNet-50 given priority when both heavy models qualify?
3. Why can total worker utilization exceed 100% while each individual worker
   remains below 100%?
4. Why does unlimited admission preserve throughput but produce extreme mean
   and P95 latency?
5. What accuracy/cost tradeoff changes when `require_correct=True`?
6. Why are there both `canceled_jobs` and `wasted_completions`?
7. What real hardware effects are missing from the simulator?


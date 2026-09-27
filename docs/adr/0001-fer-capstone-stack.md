# ADR 0001: Stack and Scope for the Facial Emotion Recognition Capstone

Date: 2026-09-27
Status: Accepted

## Context

A single-night capstone project must classify facial expressions into 7
emotions using a CNN with Batch Normalization and Data Augmentation, and
explain predictions with Grad-CAM/saliency maps. Target hardware is a
MacBook Air M1 with 8GB RAM and no dedicated GPU. No novelty or SOTA
accuracy is required — a small, correct, explainable pipeline is the goal.

## Decisions

### 1. Framework: PyTorch (not TensorFlow/Keras)

PyTorch's explicit training loop and hook API make a from-scratch Grad-CAM
implementation straightforward and transparent for a report. TensorFlow
would need `tf.GradientTape` plumbing for the same result with less
pedagogical clarity. Torch also has first-class Apple Silicon (`mps`)
support.

### 2. Dataset: FER-2013 only, fetched from a Hugging Face mirror

CK+ requires a manual, non-automatable access request and adds a second
data pipeline — out of scope for a one-night build. FER-2013 alone already
satisfies the 7-emotion classification requirement.

The official distribution channel is Kaggle, which requires an account and
API token. Instead we fetch `Jeneral/fer-2013` from Hugging Face Hub — a
public, ungated, Apache-2.0-labeled re-upload of the same dataset as
pre-split PyTorch tensors — via plain `curl`, removing any manual
credential/download step for the user. Trade-off: it's a community mirror,
not the canonical Kaggle page, so provenance rests on the mirror
maintainer rather than Kaggle directly.

### 3. Environment: dedicated conda env pinned to Python 3.11

The machine's default Homebrew Python is 3.14, which is ahead of stable
PyTorch/torchvision wheel support at the time of writing. A pinned conda
env avoids fighting dependency resolution and keeps the project isolated
from the system Python.

### 4. Deliverable: single Jupyter notebook

A single notebook (`FER_Capstone.ipynb`) instead of modular `.py` files.
For a one-night, one-person deliverable, a linear notebook is faster to
write, easier to run end-to-end, and is the conventional submission format
for a coursework capstone. The trade-off (less reusable/testable code) is
acceptable given the scope.

### 5. Explainability: hand-rolled Grad-CAM + vanilla saliency, no extra library

Both techniques are implementable in ~30 lines each using PyTorch hooks and
`tensor.grad`. Avoids adding and version-pinning a third-party
explainability library (e.g. `pytorch-grad-cam`) for a capstone-scale need.

### 6. Verification strategy: run-and-inspect, no automated test suite

The deliverable is a data-science notebook, not a library. Correctness is
verified by executing each stage (shape/dtype assertions, a small-subset
training smoke test, then the full run) rather than by unit tests, per the
project's scope and the one-night timeline.

## Consequences

- Reproducing this project on another machine requires recreating the
  conda env from `environment.yml`; no `requirements.txt`/pip-only path is
  provided.
- If the Hugging Face mirror disappears or changes format, the data-loading
  cell will need updating — the notebook should fail loudly (shape/key
  assertions) rather than silently, to surface this quickly.
- Accuracy is expected to be modest (~55–65% test accuracy) given the
  simple architecture and FER-2013's known difficulty/class imbalance. This
  is an accepted trade-off, not a defect, given "no novelty" scope.

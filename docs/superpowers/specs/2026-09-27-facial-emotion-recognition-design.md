# Facial Emotion Recognition Capstone — Design

Date: 2026-09-27
Status: Approved

## Purpose

A one-night, self-contained capstone project for a Deep Learning and
Applications course. Goal is a *working* demonstration of the two rubric
requirements below, not a novel or state-of-the-art system:

1. A CNN with Batch Normalization and Data Augmentation, classifying facial
   expressions into 7 core emotions (angry, disgust, fear, happy, sad,
   surprise, neutral).
2. Model explainability via Grad-CAM and saliency maps on trained-model
   predictions.

## Constraints

- Hardware: MacBook Air, Apple M1, 8GB RAM, no dedicated GPU (Metal/MPS
  only). Training must comfortably fit in this envelope.
- Time budget: single overnight session. No hyperparameter search, no
  ensembling, no multi-dataset training.
- Host Python (3.14, Homebrew) is too new for stable PyTorch wheels — the
  project uses its own conda environment pinned to Python 3.11.
- Single deliverable: one Jupyter notebook, runnable top-to-bottom.

## Dataset

FER-2013 only (CK+ excluded — it requires a manual, non-automatable license
request, which doesn't fit the one-night/no-extra-steps constraint).

Source: `Jeneral/fer-2013` on Hugging Face Hub (Apache-2.0 mirror of the
original Kaggle FER-2013 challenge set), fetched directly via `curl` — no
account or API token required. Already downloaded to `data/train.pt` (28,709
training images) and `data/test.pt` (7,178 test images), each a serialized
PyTorch tensor pair of (pixels, label).

Note: this is a community re-upload, not the official Kaggle page. The
Apache-2.0 label is the mirror maintainer's own; treat the underlying image
data under FER-2013's original academic-use expectations if redistributed
outside this coursework context.

## Architecture

**Framework**: PyTorch + torchvision.

**Pipeline (single notebook, `FER_Capstone.ipynb`)**:

1. Setup — imports, device selection (`mps` if available, else `cpu`), seed.
2. Data loading — `torch.load` the two `.pt` files; wrap in a custom
   `Dataset` that reshapes flat pixel vectors to 1×48×48 and normalizes to
   [0, 1].
3. Data augmentation — applied only to the training split, via torchvision
   transforms: random horizontal flip, small random rotation (±10°), small
   random translation. Validation/test data is not augmented.
4. Train/val split — carve a validation set out of the training tensor
   (e.g. 90/10 stratified-ish split by simple random sample), keep the
   provided test tensor untouched as the held-out test set.
5. Model — a small CNN:
   `[Conv(32)-BN-ReLU-MaxPool] -> [Conv(64)-BN-ReLU-MaxPool] ->
   [Conv(128)-BN-ReLU-MaxPool] -> Dropout -> FC(128) -> Dropout -> FC(7)`
6. Training loop — Adam optimizer, CrossEntropyLoss, ~15–25 epochs (small
   enough to finish in well under an hour on this hardware given the small
   image size and dataset), track train/val loss and accuracy per epoch,
   plot the curves.
7. Evaluation — test-set accuracy, confusion matrix, per-class precision/
   recall via `sklearn.metrics.classification_report`.
8. Explainability — manual Grad-CAM (forward/backward hook on the last conv
   block) and a vanilla input-gradient saliency map, rendered as heatmap
   overlays on one correctly-classified sample per emotion class (7 images
   total, or up to 14 if also showing one misclassified example per class
   time permitting).
9. Save the trained model checkpoint (`checkpoints/model.pt`, gitignored).

## Testing / Verification Approach

No separate test suite — this is a notebook-based data science deliverable.
Verification happens by execution, per the user's instruction to test after
every implementation step:

- After the data-loading cell: assert tensor shapes, print label
  distribution, visually display a handful of sample images.
- After first defining the model: run a single forward pass on a dummy
  batch and assert the output shape is `(batch, 7)`.
- Before committing to a full run: train for 1–2 epochs on a small subset
  (e.g. 1,000 samples) to confirm the loop runs end-to-end without shape/
  dtype errors and loss decreases.
- Full run: confirm loss decreases and val accuracy is in a plausible range
  for a simple CNN on FER-2013 (roughly 55–65%; this is a known hard,
  imbalanced dataset, so this is not a bug if val accuracy lands there).
- Grad-CAM/saliency cell: confirm heatmaps are non-blank, non-NaN, and
  visibly concentrate on face regions (eyes/mouth) rather than background.

## Out of Scope

- CK+ dataset, multi-dataset training or transfer learning.
- Hyperparameter tuning, architecture search, ensembling.
- Web/app deployment, real-time webcam inference.
- Automated test suite / CI.

## Environment

Conda environment (`environment.yml`), Python 3.11:
`pytorch`, `torchvision`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`,
`seaborn`, `jupyter`, `ipykernel`.

# Facial Emotion Recognition Capstone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one runnable Jupyter notebook that trains a small CNN (with
BatchNorm + data augmentation) to classify FER-2013 faces into 7 emotions,
and explains its predictions with Grad-CAM and saliency maps.

**Architecture:** A jupytext "percent format" script (`fer_capstone.py`) is
the editable source of truth — it runs as a plain Python script for fast,
automated verification after every step — and is paired via jupytext to
the actual deliverable notebook (`FER_Capstone.ipynb`), which is kept in
sync and is what gets executed end-to-end and submitted.

**Tech Stack:** Python 3.11 (conda env), PyTorch + torchvision, NumPy,
Pandas, Pillow, scikit-learn, matplotlib, seaborn, jupytext, Jupyter.

**Spec:** `docs/superpowers/specs/2026-09-27-facial-emotion-recognition-design.md`
**ADR:** `docs/adr/0001-fer-capstone-stack.md`

## Global Constraints

- Conda env pinned to Python 3.11 (host Python 3.14 is incompatible with
  stable PyTorch wheels) — env name `fer-capstone`, defined in
  `environment.yml`.
- Data already downloaded and verified: `data/train.pt` (28,709 records)
  and `data/test.pt` (7,178 records), each a pickled `list[dict]` with keys
  `img_bytes` (JPEG-encoded 48×48 grayscale bytes) and `labels` (one of the
  7 emotion strings, verified via direct inspection: `angry`, `disgust`,
  `fear`, `happy`, `neutral`, `sad`, `surprise`).
- Class index mapping is fixed and must be used everywhere: alphabetical
  order `["angry", "disgust", "fear", "happy", "neutral", "sad",
  "surprise"]` → indices `0..6`.
- No CK+, no hyperparameter search, no ensembling, no deployment (per ADR).
- Batch size and epoch count must stay modest — target hardware is 8GB
  RAM, no dedicated GPU (Apple M1, `mps` backend only).
- Test accuracy in the ~55–65% range is an acceptable, expected outcome
  for this simple architecture on FER-2013 — not a bug to chase past.
- Every task must end in something actually run (script execution), not
  just written code — per the project's explicit "test after every
  implementation" requirement.

## Review Focus

- **Severe class imbalance** (`disgust` is ~13x rarer than `happy`) could
  let the model collapse to predicting majority classes while accuracy
  still looks plausible — the evaluation task must report per-class
  precision/recall/F1, not just overall accuracy.
- **Corrupted or reshaped pickle records** (a record missing a key, or an
  image that isn't 48×48/mode `L`) would otherwise surface as a cryptic
  error deep in training — the data-loading task asserts shape/dtype/key
  invariants immediately after decoding, and fails with a clear message.
- **`mps` backend unavailable or erroring on an op** on a given machine —
  device selection must fall back to CPU automatically rather than
  crashing the whole notebook.
- **Degenerate Grad-CAM/saliency output** (all-zero or NaN heatmap, e.g.
  from a dead-ReLU activation on a particular sample) would render a
  blank/misleading overlay — the explainability task asserts the heatmap
  has nonzero variance before rendering it.
- **Fresh clone / missing data files** — if `data/train.pt` or
  `data/test.pt` isn't present (e.g. `.gitignore` excludes `data/`, so a
  fresh clone won't have it), the notebook should raise a clear
  `FileNotFoundError`-based message pointing at the ADR's download
  instructions, not fail deep inside `pickle.load`.

---

## File Structure

- `environment.yml` — conda environment definition.
- `fer_capstone.py` — jupytext percent-format source script; all pipeline
  code lives here during development.
- `FER_Capstone.ipynb` — generated/paired notebook; the actual deliverable.
- `data/train.pt`, `data/test.pt` — already present (gitignored).
- `checkpoints/model.pt` — saved trained weights (gitignored).
- `README.md` — setup + run instructions.

---

### Task 1: Environment Setup

**Files:**
- Create: `environment.yml`
- Create: `README.md`

**Interfaces:**
- Produces: a working `fer-capstone` conda env with `python`, `torch`,
  `torchvision` importable, used by every subsequent task.

- [ ] **Step 1: Write `environment.yml`**

```yaml
name: fer-capstone
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.11
  - pytorch
  - torchvision
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - pillow
  - jupyter
  - ipykernel
  - jupytext
```

- [ ] **Step 2: Create the environment**

Run: `conda env create -f environment.yml`
Expected: env `fer-capstone` created without dependency conflicts.

- [ ] **Step 3: Verify the environment works**

Run:
```bash
conda run -n fer-capstone python3 -c "
import torch, torchvision, PIL, sklearn, matplotlib, seaborn, pandas, numpy
print('torch', torch.__version__)
print('mps available:', torch.backends.mps.is_available())
"
```
Expected: prints a torch version string and `mps available: True` (or
`False` — either is fine, just must not error).

- [ ] **Step 4: Write `README.md`**

```markdown
# Facial Emotion Recognition Capstone

CNN (BatchNorm + data augmentation) classifying FER-2013 faces into 7
emotions, with Grad-CAM/saliency-map explainability.

## Setup

    conda env create -f environment.yml
    conda activate fer-capstone

## Run

Open and run `FER_Capstone.ipynb` top to bottom (Jupyter). Or regenerate
it from source after editing `fer_capstone.py`:

    jupytext --to notebook fer_capstone.py

## Data

`data/train.pt` / `data/test.pt` are the FER-2013 dataset (28,709 train +
7,178 test images), fetched from the `Jeneral/fer-2013` Hugging Face
mirror. See `docs/adr/0001-fer-capstone-stack.md` for provenance and how
to re-fetch if missing:

    curl -sL "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/train.pt" -o data/train.pt
    curl -sL "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/test.pt" -o data/test.pt
```

- [ ] **Step 5: Commit**

```bash
git add environment.yml README.md
git commit -m "Add conda environment and setup instructions"
```

---

### Task 2: Data Loading

**Files:**
- Create: `fer_capstone.py` (start the file; jupytext percent format)

**Interfaces:**
- Produces: `EMOTIONS` (list[str], len 7), `LABEL_TO_IDX` (dict[str,int]),
  `load_split(path: str) -> tuple[np.ndarray uint8 (N,48,48), np.ndarray
  int64 (N,)]`, used by Task 3 onward.

- [ ] **Step 1: Write the data-loading cell**

```python
# %%
import pickle
import io
import os
import numpy as np
from PIL import Image

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL_TO_IDX = {name: i for i, name in enumerate(EMOTIONS)}


def load_split(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Download it first — see README.md 'Data' section "
            f"or docs/adr/0001-fer-capstone-stack.md."
        )
    with open(path, "rb") as f:
        records = pickle.load(f)

    images = np.zeros((len(records), 48, 48), dtype=np.uint8)
    labels = np.zeros(len(records), dtype=np.int64)
    for i, rec in enumerate(records):
        assert "img_bytes" in rec and "labels" in rec, f"record {i} missing expected keys"
        img = Image.open(io.BytesIO(rec["img_bytes"])).convert("L")
        assert img.size == (48, 48), f"record {i} has unexpected size {img.size}"
        images[i] = np.array(img, dtype=np.uint8)
        assert rec["labels"] in LABEL_TO_IDX, f"record {i} has unknown label {rec['labels']!r}"
        labels[i] = LABEL_TO_IDX[rec["labels"]]
    return images, labels


# %%
train_images, train_labels = load_split("data/train.pt")
test_images, test_labels = load_split("data/test.pt")

print("train:", train_images.shape, train_labels.shape)
print("test:", test_images.shape, test_labels.shape)
assert train_images.shape == (28709, 48, 48)
assert test_images.shape == (7178, 48, 48)
assert train_images.dtype == np.uint8
assert set(np.unique(train_labels)) == set(range(7))

import collections
print("train label counts:", collections.Counter(train_labels.tolist()))
print("test label counts:", collections.Counter(test_labels.tolist()))
```

- [ ] **Step 2: Run it and verify**

Run: `conda run -n fer-capstone python3 fer_capstone.py`
Expected: prints shapes `(28709, 48, 48) (28709,)` and `(7178, 48, 48)
(7178,)`, no assertion errors, and label counts showing `disgust` as the
rare class (~547 in train, ~111 in test).

- [ ] **Step 3: Add a visual sanity check (saved, not just asserted)**

```python
# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 7, figsize=(14, 2))
for emotion_idx, ax in enumerate(axes):
    sample_idx = np.where(train_labels == emotion_idx)[0][0]
    ax.imshow(train_images[sample_idx], cmap="gray")
    ax.set_title(EMOTIONS[emotion_idx], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("checkpoints_sample_grid.png")
print("saved checkpoints_sample_grid.png")
```

- [ ] **Step 4: Run and open the saved image to confirm faces look correct**

Run: `conda run -n fer-capstone python3 fer_capstone.py`
Then inspect `checkpoints_sample_grid.png` — each of the 7 tiles should
show a distinguishable, non-corrupted grayscale face.

- [ ] **Step 5: Commit**

```bash
git add fer_capstone.py
git commit -m "Add and verify FER-2013 data loading"
```

---

### Task 3: Dataset Class, Augmentation, Train/Val Split

**Files:**
- Modify: `fer_capstone.py` (append)

**Interfaces:**
- Consumes: `load_split`, `train_images`, `train_labels`, `test_images`,
  `test_labels` from Task 2.
- Produces: `FERDataset` (torch Dataset), `train_ds`, `val_ds`, `test_ds`
  instances, used by Task 4/5's `DataLoader`s.

- [ ] **Step 1: Write the Dataset class and transforms**

```python
# %%
import torch
from torch.utils.data import Dataset
from torchvision import transforms

torch.manual_seed(42)
np.random.seed(42)


class FERDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float().div(255.0).unsqueeze(0)  # (1, 48, 48)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
])

# %%
rng = np.random.default_rng(42)
n = len(train_labels)
perm = rng.permutation(n)
val_size = int(0.1 * n)
val_idx, train_idx = perm[:val_size], perm[val_size:]

train_ds = FERDataset(train_images[train_idx], train_labels[train_idx], transform=train_transform)
val_ds = FERDataset(train_images[val_idx], train_labels[val_idx], transform=None)
test_ds = FERDataset(test_images, test_labels, transform=None)

print("train_ds:", len(train_ds), "val_ds:", len(val_ds), "test_ds:", len(test_ds))
assert len(train_ds) + len(val_ds) == n
assert len(test_ds) == 7178

sample_img, sample_label = train_ds[0]
assert sample_img.shape == (1, 48, 48), sample_img.shape
assert sample_img.dtype == torch.float32
assert 0.0 <= sample_img.min() and sample_img.max() <= 1.0
print("sample OK:", sample_img.shape, sample_img.dtype, sample_label)
```

- [ ] **Step 2: Run and verify**

Run: `conda run -n fer-capstone python3 fer_capstone.py`
Expected: prints dataset sizes (`train_ds: 25839 val_ds: 2870 test_ds:
7178` — exact split may vary slightly but should sum to 28709/7178), no
assertion errors.

- [ ] **Step 3: Commit**

```bash
git add fer_capstone.py
git commit -m "Add FERDataset, augmentation transforms, and train/val split"
```

---

### Task 4: Model Definition

**Files:**
- Modify: `fer_capstone.py` (append)

**Interfaces:**
- Produces: `EmotionCNN(nn.Module)` with `forward(x: (B,1,48,48)) ->
  (B,7)`, and `.block3` as the named layer Task 8's Grad-CAM hooks into.

- [ ] **Step 1: Write the model**

```python
# %%
import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )  # -> (32, 24, 24)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )  # -> (64, 12, 12)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )  # -> (128, 12, 12) -- Grad-CAM hooks this block's output
        self.pool3 = nn.MaxPool2d(2)  # -> (128, 6, 6)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# %%
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
try:
    _probe = torch.zeros(2, 1, 48, 48, device=device)
    EmotionCNN().to(device)(_probe)
except RuntimeError as e:
    print(f"mps backend errored ({e}); falling back to cpu")
    device = torch.device("cpu")
print("using device:", device)

model = EmotionCNN().to(device)
dummy = torch.randn(4, 1, 48, 48, device=device)
out = model(dummy)
print("output shape:", out.shape)
assert out.shape == (4, 7)
```

- [ ] **Step 2: Run and verify**

Run: `conda run -n fer-capstone python3 fer_capstone.py`
Expected: prints `using device: mps` or `using device: cpu`, then `output
shape: torch.Size([4, 7])`, no errors.

- [ ] **Step 3: Commit**

```bash
git add fer_capstone.py
git commit -m "Add EmotionCNN model with BatchNorm, verified forward pass"
```

---

### Task 5: Training Loop + Small-Subset Smoke Test

**Files:**
- Modify: `fer_capstone.py` (append)

**Interfaces:**
- Consumes: `EmotionCNN`, `train_ds`, `val_ds`, `device` from Tasks 3-4.
- Produces: `train_one_epoch(model, loader, optimizer, criterion, device)
  -> (avg_loss, accuracy)`, `evaluate(model, loader, criterion, device) ->
  (avg_loss, accuracy)`, used by Task 6's full run and Task 7's evaluation.

- [ ] **Step 1: Write the training/eval functions**

```python
# %%
from torch.utils.data import DataLoader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total
```

- [ ] **Step 2: Smoke-test on a small subset (fast, catches bugs early)**

```python
# %%
smoke_train = torch.utils.data.Subset(train_ds, range(500))
smoke_val = torch.utils.data.Subset(val_ds, range(100))
smoke_train_loader = DataLoader(smoke_train, batch_size=32, shuffle=True)
smoke_val_loader = DataLoader(smoke_val, batch_size=32)

smoke_model = EmotionCNN().to(device)
smoke_optimizer = torch.optim.Adam(smoke_model.parameters(), lr=1e-3)
smoke_criterion = nn.CrossEntropyLoss()

losses = []
for epoch in range(2):
    loss, acc = train_one_epoch(smoke_model, smoke_train_loader, smoke_optimizer, smoke_criterion, device)
    val_loss, val_acc = evaluate(smoke_model, smoke_val_loader, smoke_criterion, device)
    losses.append(loss)
    print(f"[smoke] epoch {epoch}: train_loss={loss:.3f} train_acc={acc:.3f} val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

assert losses[-1] < losses[0], "loss did not decrease on smoke test — check the training loop"
print("smoke test passed: loss decreased")
```

- [ ] **Step 3: Run and verify**

Run: `conda run -n fer-capstone python3 fer_capstone.py`
Expected: two `[smoke]` lines printed, second epoch's train_loss lower
than the first's, `smoke test passed` printed, no errors. (Val accuracy on
500 samples over 2 epochs will be low/noisy — that's expected, this step
only checks the loop runs and loss trends down.)

- [ ] **Step 4: Commit**

```bash
git add fer_capstone.py
git commit -m "Add training/eval loop, verified with small-subset smoke test"
```

---

### Task 6: Full Training Run

**Files:**
- Modify: `fer_capstone.py` (append)

**Interfaces:**
- Consumes: `EmotionCNN`, `train_ds`, `val_ds`, `train_one_epoch`,
  `evaluate`, `device` from Tasks 3-5.
- Produces: trained `model` (in-memory) and `checkpoints/model.pt` on
  disk, consumed by Task 7 (evaluation) and Task 8 (explainability).

- [ ] **Step 1: Write the full training run with curve plotting and checkpoint save**

```python
# %%
import os

os.makedirs("checkpoints", exist_ok=True)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128)

model = EmotionCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 20
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"epoch {epoch+1}/{EPOCHS}: train_loss={train_loss:.3f} train_acc={train_acc:.3f} "
          f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

torch.save(model.state_dict(), "checkpoints/model.pt")
print("saved checkpoints/model.pt")

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(history["train_loss"], label="train")
axes[0].plot(history["val_loss"], label="val")
axes[0].set_title("Loss")
axes[0].legend()
axes[1].plot(history["train_acc"], label="train")
axes[1].plot(history["val_acc"], label="val")
axes[1].set_title("Accuracy")
axes[1].legend()
plt.tight_layout()
plt.savefig("checkpoints/training_curves.png")
print("saved checkpoints/training_curves.png")

assert history["val_acc"][-1] > history["val_acc"][0], "val accuracy did not improve over training"
```

- [ ] **Step 2: Run the full training and verify**

Run: `conda run -n fer-capstone python3 fer_capstone.py`
Expected: 20 epoch lines printed, final val_acc noticeably higher than
the first epoch's (roughly 45-65% range is plausible), `checkpoints/
model.pt` and `checkpoints/training_curves.png` written, no assertion
errors. This step will take the longest (likely several minutes on this
hardware) — that's expected.

- [ ] **Step 3: Commit**

```bash
git add fer_capstone.py
git commit -m "Add full training run with checkpointing and loss/accuracy curves"
```

---

### Task 7: Evaluation on Test Set

**Files:**
- Modify: `fer_capstone.py` (append)

**Interfaces:**
- Consumes: trained `model`, `test_ds`, `EMOTIONS` from Tasks 3, 4, 6.
- Produces: printed classification report + saved confusion matrix image,
  consumed only by the final notebook read-through (no downstream code
  dependency).

- [ ] **Step 1: Write the evaluation cell**

```python
# %%
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

test_loader = DataLoader(test_ds, batch_size=128)

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
print(f"test accuracy: {test_acc:.3f}")
assert 0.0 < test_acc < 1.0

print(classification_report(all_labels, all_preds, target_names=EMOTIONS, zero_division=0))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=EMOTIONS, yticklabels=EMOTIONS, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix (test acc={test_acc:.3f})")
plt.tight_layout()
plt.savefig("checkpoints/confusion_matrix.png")
print("saved checkpoints/confusion_matrix.png")
```

- [ ] **Step 2: Run and verify**

Run: `conda run -n fer-capstone python3 fer_capstone.py`
Expected: prints test accuracy (plausible range ~0.45-0.65) and a full
per-class precision/recall/F1 table (7 rows), `checkpoints/
confusion_matrix.png` written. Check the report: `disgust`'s row should
not show `0.00` recall across the board (that would indicate the model
collapsed and ignores the rare class entirely) — if it does, that's a
real finding to note, not a script bug.

- [ ] **Step 3: Commit**

```bash
git add fer_capstone.py
git commit -m "Add test-set evaluation: accuracy, classification report, confusion matrix"
```

---

### Task 8: Explainability — Grad-CAM and Saliency Maps

**Files:**
- Modify: `fer_capstone.py` (append)

**Interfaces:**
- Consumes: trained `model`, `model.block3`, `test_ds`, `EMOTIONS`,
  `device` from Tasks 3, 4, 6.
- Produces: `GradCAM` class and `compute_saliency` function (self-
  contained, no downstream consumers — this is the final visualization
  task).

- [ ] **Step 1: Write Grad-CAM and saliency implementations**

```python
# %%
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy()


def compute_saliency(model, input_tensor, class_idx):
    input_tensor = input_tensor.clone().requires_grad_(True)
    model.zero_grad()
    output = model(input_tensor)
    score = output[0, class_idx]
    score.backward()
    saliency = input_tensor.grad.abs()[0, 0]
    saliency = saliency - saliency.min()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency.detach().cpu().numpy()
```

- [ ] **Step 2: Generate and save Grad-CAM + saliency overlays for one sample per class**

```python
# %%
from PIL import Image as PILImage


def resize_map(arr_2d, size=(48, 48)):
    img = PILImage.fromarray((arr_2d * 255).astype(np.uint8))
    img = img.resize(size, PILImage.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


gradcam = GradCAM(model, model.block3)

fig, axes = plt.subplots(3, 7, figsize=(16, 7))
for col, emotion_idx in enumerate(range(7)):
    sample_idx = int(np.where(test_ds.labels == emotion_idx)[0][0])
    img_tensor, label = test_ds[sample_idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    cam = gradcam.generate(input_tensor, class_idx=label)
    cam_resized = resize_map(cam)
    assert cam_resized.std() > 1e-6, f"Grad-CAM heatmap is degenerate for {EMOTIONS[emotion_idx]}"

    saliency = compute_saliency(model, input_tensor, class_idx=label)
    assert saliency.std() > 1e-6, f"saliency map is degenerate for {EMOTIONS[emotion_idx]}"

    face = img_tensor[0].numpy()
    axes[0, col].imshow(face, cmap="gray")
    axes[0, col].set_title(EMOTIONS[emotion_idx], fontsize=9)
    axes[0, col].axis("off")
    axes[1, col].imshow(face, cmap="gray")
    axes[1, col].imshow(cam_resized, cmap="jet", alpha=0.5)
    axes[1, col].axis("off")
    axes[2, col].imshow(saliency, cmap="hot")
    axes[2, col].axis("off")

axes[0, 0].set_ylabel("input", fontsize=9)
axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)
axes[2, 0].set_ylabel("saliency", fontsize=9)
plt.tight_layout()
plt.savefig("checkpoints/explainability.png")
print("saved checkpoints/explainability.png")
```

- [ ] **Step 3: Run and verify**

Run: `conda run -n fer-capstone python3 fer_capstone.py`
Expected: no assertion errors (no degenerate heatmaps), `checkpoints/
explainability.png` saved — open it and confirm Grad-CAM/saliency
highlights concentrate on eyes/mouth/brow regions rather than image
corners or background.

- [ ] **Step 4: Commit**

```bash
git add fer_capstone.py
git commit -m "Add Grad-CAM and saliency map explainability visualizations"
```

---

### Task 9: Assemble Final Notebook

**Files:**
- Create: `FER_Capstone.ipynb` (generated from `fer_capstone.py`)
- Modify: `fer_capstone.py` (add markdown headers via jupytext `# %% [markdown]` cells)

**Interfaces:**
- Consumes: the complete `fer_capstone.py` from Tasks 2-8.
- Produces: the final deliverable notebook — no downstream tasks.

- [ ] **Step 1: Add markdown section headers to `fer_capstone.py`**

Insert a `# %% [markdown]` cell before each task's code, e.g.:

```python
# %% [markdown]
# ## 1. Data Loading
#
# Loads FER-2013 from the pickled `.pt` files (JPEG-encoded 48x48
# grayscale images + string emotion labels) and verifies shapes/labels.
```

Add one such header before each of: Data Loading, Dataset & Augmentation,
Model, Training/Eval Functions, Full Training Run, Evaluation,
Explainability.

- [ ] **Step 2: Generate the notebook**

Run: `conda run -n fer-capstone jupytext --to notebook fer_capstone.py`
Expected: creates `FER_Capstone.ipynb` with no errors.

- [ ] **Step 3: Execute the notebook end-to-end fresh (the real integration test)**

Run:
```bash
conda run -n fer-capstone jupyter nbconvert --to notebook --execute --inplace FER_Capstone.ipynb
```
Expected: completes with no cell errors. This re-runs everything
(including the full 20-epoch training) inside the notebook itself — this
is the definitive check that the deliverable actually works standalone,
independent of the dev script.

- [ ] **Step 4: Spot-check the executed notebook**

Open `FER_Capstone.ipynb` and confirm: data shape prints are present, the
training curve plot renders inline, the classification report table is
present, the confusion matrix and explainability images render inline,
and the last cell ran without a traceback.

- [ ] **Step 5: Commit**

```bash
git add fer_capstone.py FER_Capstone.ipynb
git commit -m "Generate and verify final FER_Capstone.ipynb end-to-end"
```

---

## Self-Review Notes

- **Spec coverage:** CNN+BatchNorm (Task 4), data augmentation (Task 3),
  7-emotion classification (Tasks 2-7), Grad-CAM + saliency (Task 8),
  single notebook deliverable (Task 9) — all spec sections have an owning
  task.
- **Placeholder scan:** none found; every step has runnable code and a
  concrete expected-output description.
- **Type consistency:** `EmotionCNN.forward` returns `(B, 7)` logits used
  identically in Tasks 5, 6, 7, 8; `EMOTIONS`/`LABEL_TO_IDX` from Task 2
  are reused unchanged everywhere; `test_ds.labels` (Task 3) is what Task
  8 indexes into.
- **Review Focus:** all five items above are each pinned to a concrete
  assertion in their owning task (class imbalance → Task 7's
  classification report; corrupted records → Task 2's asserts; `mps`
  fallback → Task 4's try/except; degenerate heatmaps → Task 8's std()
  asserts; missing data files → Task 2's `FileNotFoundError`).

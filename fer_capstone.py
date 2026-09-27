# %% [markdown]
# ## 1. Data Loading
#
# Loads FER-2013 from the pickled `.pt` files (JPEG-encoded 48x48
# grayscale images + string emotion labels) and verifies shapes/labels.

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

# %% [markdown]
# ## 2. Dataset Class, Augmentation, Train/Val Split
#
# Wraps the numpy arrays in a `torch.utils.data.Dataset`, applies data
# augmentation (flip/rotation/translation) only to the training split, and
# carves a validation set out of the training data.

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

# %% [markdown]
# ## 3. Model Definition
#
# A small CNN: three Conv-BatchNorm-ReLU blocks with max-pooling, then a
# dropout + fully-connected classifier head over the 7 emotion classes.
# `block3` is kept as a named submodule so Grad-CAM (Task 8) can hook it.

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

# %% [markdown]
# ## 4. Training / Evaluation Functions
#
# Standard train/eval epoch loops, smoke-tested on a small subset before
# committing to the full run in the next section.

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

# %% [markdown]
# ## 5. Full Training Run
#
# Trains the real model for 20 epochs, saves a checkpoint, and plots
# loss/accuracy curves.

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

# %% [markdown]
# ## 6. Evaluation on Test Set
#
# Test accuracy, per-class precision/recall/F1, and a confusion matrix —
# important given FER-2013's class imbalance (rare `disgust` class).

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

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

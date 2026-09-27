# Facial Emotion Recognition Capstone

CNN (BatchNorm + data augmentation) classifying FER-2013 faces into 7
emotions, with Grad-CAM/saliency-map explainability.

## Setup

    conda env create -f environment.yml
    conda activate fer-capstone

## Run

Open and run `FER_Capstone.ipynb` top to bottom (Jupyter). Or regenerate
it from source after editing `fer_capstone.py`:

    jupytext --to notebook -o FER_Capstone.ipynb fer_capstone.py

## Data

`data/train.pt` / `data/test.pt` are the FER-2013 dataset (28,709 train +
7,178 test images), fetched from the `Jeneral/fer-2013` Hugging Face
mirror. See `docs/adr/0001-fer-capstone-stack.md` for provenance and how
to re-fetch if missing:

    curl -sL "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/train.pt" -o data/train.pt
    curl -sL "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main/test.pt" -o data/test.pt

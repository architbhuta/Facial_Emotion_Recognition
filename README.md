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
mirror. See `docs/adr/0001-fer-capstone-stack.md` for provenance. Re-fetch
if missing (pinned to the revision this project was built against, with
checksums so a corrupted/failed download is caught immediately instead of
surfacing later as a cryptic unpickling error):

    mkdir -p data
    curl -fsSL "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/3a46cbfae3f5b348449335f300666a0ae330f121/train.pt" -o data/train.pt
    curl -fsSL "https://huggingface.co/datasets/Jeneral/fer-2013/resolve/3a46cbfae3f5b348449335f300666a0ae330f121/test.pt" -o data/test.pt
    shasum -a 256 -c <<'EOF'
    22b4a2cb7a0dad26ec86cf2f12c3c8f5aa4a5dca0e59400f07d583d0ee212d1e  data/train.pt
    ae0b8cac6df796dc0d263387aa0258b930bf5a31f069967d0f58899d2422233f  data/test.pt
    EOF

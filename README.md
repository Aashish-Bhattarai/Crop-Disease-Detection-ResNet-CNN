# Crop Disease Detection — ResNet18 transfer learning (PyTorch)

Image classifier for pepper, potato and tomato leaf disease, trained on
PlantVillage and served behind a FastAPI endpoint. Training, evaluation,
CLI prediction and the serving app are all in this repo.

## What it actually is

- **ResNet18** with ImageNet weights, **backbone frozen** — every parameter has
  `requires_grad = False` and only the final `fc` layer is replaced
  (`nn.Linear(512, 15)`) and trained (`src/model.py`).
- Trained with **Adam on `model.fc.parameters()` only**, `lr=1e-3`, 5 epochs,
  batch size 32, images resized to 224×224 (`src/config.py`, `src/train.py`).
- Framework is **PyTorch + torchvision**, not TensorFlow/Keras.

## Dataset

**PlantVillage — 20,639 images across 15 classes** (counted on disk), loaded
with `torchvision.datasets.ImageFolder`:

| Crop | Classes |
|---|---|
| Pepper (bell) | Bacterial spot, healthy |
| Potato | Early blight, Late blight, healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites (two-spotted), Target Spot, Yellow Leaf Curl Virus, Mosaic virus, healthy |

Split is **80/20 stratified, `random_state=42`** (`src/data.py`). Training
applies a random horizontal flip; validation does not. The dataset itself is
not committed — download PlantVillage into `data/PlantVillage/` as one
subfolder per class.

## Results

Measured on the 20% held-out split (4,128 images) by `src/eval_holdout.py`.
Full output is committed under [`evaluation/`](evaluation/).

| Metric | Value |
|---|---|
| Accuracy | **0.9276** |
| Macro F1 | **0.9185** |
| Weighted F1 | **0.9277** |
| Images evaluated | 4,128 |

Per-class F1 ranges from **0.7371** (Tomato Early blight) to **0.9933**
(Pepper bell healthy); the weakest classes are Tomato Early blight (0.7371),
Target Spot (0.8433) and Potato healthy (0.8667, n=30 — the smallest class).
See [`evaluation/classification_report.txt`](evaluation/classification_report.txt)
and [`evaluation/metrics.json`](evaluation/metrics.json).

### Read these caveats before believing that number

- **Possible train/test overlap.** `src/eval_holdout.py` reconstructs the split
  with the same seed and code as `src/train.py`, so the evaluated images are
  held out *provided the committed checkpoint was trained by this code at this
  seed*. That provenance is not independently verified for the existing
  checkpoint, so treat 0.9276 as an upper bound. Retrain with `src/train.py`
  before quoting it as a clean test result.
- **This is a lab number and lab numbers do not survive the field.** The same
  checkpoint was evaluated against real field photography in a separate
  project ([FieldTrust](https://github.com/Aashish-Bhattarai/fieldtrust)):
  accuracy fell to **21.2%** on PlantDoc and **4.9%** on the pathologist-labelled
  Tanzania potato dataset, with early-blight recall of 0% across 300 real
  images. PlantVillage is uniform-background lab imagery; a 0.93 score on it
  says nothing about a photo taken in a real field.
- No hyperparameter search, no early stopping, no cross-validation — 5 epochs
  of a fixed configuration.

## Layout

```
src/
  config.py         paths, image size 224, batch 32, 5 epochs, lr 1e-3, seed 42
  data.py           ImageFolder + 80/20 stratified split, transforms
  model.py          ResNet18, frozen backbone, new fc head
  train.py          trains fc only, saves {"model": state_dict, "class_to_idx": ...}
  evaluate.py       interactive report + confusion matrix
  eval_holdout.py   non-interactive eval, writes evaluation/ artifacts
  predict.py        load_model() / predict(), plus a CLI
app/
  main.py           FastAPI service
Dockerfile          python:3.10-slim, uvicorn on :8000
```

## Running it

```bash
pip install -r requirements.txt

python -m src.train           # trains, writes models/crop_disease_model.pth
python -m src.eval_holdout    # writes evaluation/metrics.json + report + matrix
python -m src.predict --image path/to/leaf.jpg
```

Serving:

```bash
uvicorn app.main:app --reload        # or: docker build -t crop-disease . && docker run -p 8000:8000 crop-disease
curl -F "file=@leaf.jpg" http://localhost:8000/predict
# {"prediction": "Potato___Early_blight"}
```

`POST /predict` takes a multipart image upload and returns the predicted class
label. The model is loaded once at startup.

The checkpoint (`models/*.pth`, 43 MB) is gitignored — train it locally or
supply your own.

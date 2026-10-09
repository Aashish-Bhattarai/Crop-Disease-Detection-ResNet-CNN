# Crop Disease Detection — ResNet18 (PyTorch) — and an honest look at where it breaks

**0.93 accuracy in the lab. 21% on real field data. Both numbers are in this repo on purpose.**

A ResNet18 transfer-learning classifier for pepper, potato and tomato leaf
diseases (PlantVillage, 20,639 images, 15 classes), served behind a FastAPI
endpoint — plus the evaluation work that shows exactly where it stops working:
the same checkpoint scores **21.2% on PlantDoc field imagery** and
**4.9% on Tanzania field samples**. Lab accuracy is an upper bound, and most
public plant-disease projects quietly leave that part out.

## Headline results

| Benchmark | Accuracy | What it means |
|---|---|---|
| PlantVillage held-out split (4,128 images) | **0.9276** (macro F1 0.9185) | laboratory conditions |
| PlantDoc (real field photos) | **0.2120** | distribution shift is brutal |
| Tanzania field samples | **0.0490** | different crops, lighting, backgrounds |

Full per-class metrics, confusion matrix and eval scripts are committed under
[`evaluation/`](evaluation/). The follow-up question — *why* it collapses and
how to detect that automatically — is the evaluation-harness work this project
feeds into.

## What it is

- **ResNet18**, ImageNet weights, **backbone frozen** — only the final `fc`
  layer is replaced and trained (`src/model.py`).
- Adam on `model.fc.parameters()` only, `lr=1e-3`, 5 epochs, batch 32,
  224×224 (`src/config.py`, `src/train.py`).
- PyTorch + torchvision. 80/20 stratified split, `random_state=42`;
  train-time horizontal flip only.

## Dataset

PlantVillage — 20,639 images, 15 classes across pepper (2), potato (3) and
tomato (10). Not committed (~382 MB): download PlantVillage into
`data/PlantVillage/` as one subfolder per class.

## Structure

```
src/                data, model, train, eval_holdout, predict (CLI)
app/main.py         FastAPI service
evaluation/         committed holdout metrics, report, confusion matrix
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

The checkpoint (`models/*.pth`, 43 MB) is gitignored — train it locally or
supply your own.

## License

MIT

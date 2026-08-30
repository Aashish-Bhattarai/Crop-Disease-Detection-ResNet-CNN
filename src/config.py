from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "PlantVillage"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

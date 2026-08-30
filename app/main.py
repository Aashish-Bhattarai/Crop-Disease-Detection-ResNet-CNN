from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil

from src.predict import load_model, predict

# Constants
IMAGE_SIZE = 224

# Load model once at startup
model, class_to_idx = load_model()

# Initialize FastAPI
app = FastAPI(title="Crop Disease Detection API")

# Upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image and returns the predicted crop disease class.
    """
    image_path = UPLOAD_DIR / file.filename

    # Save uploaded file
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run prediction
    prediction = predict(model, class_to_idx, str(image_path), IMAGE_SIZE)

    return {"prediction": prediction}

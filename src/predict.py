import torch
from PIL import Image
from torchvision import transforms
from src.model import build_model
from src.config import MODEL_DIR, DEVICE, IMAGE_SIZE
from pathlib import Path
import argparse

def load_model(model_path: Path = MODEL_DIR / "crop_disease_model.pth"):
    """
    Load the PyTorch model and class mapping.
    """
    ckpt = torch.load(model_path, map_location=DEVICE)
    model = build_model(len(ckpt["class_to_idx"]))
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()
    idx_to_class = {v: k for k, v in ckpt["class_to_idx"].items()}
    return model, idx_to_class


def predict(model, idx_to_class, image_path: str, image_size: int = IMAGE_SIZE):
    """
    Predict the class of an image using the given model.
    """
    device = next(model.parameters()).device  # get model device

    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred_idx = output.argmax(1).item()

    return idx_to_class[pred_idx]


# Showing prediction from command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict crop disease from an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
    else:
        model, idx_to_class = load_model()
        result = predict(model, idx_to_class, str(image_path))
        print(f"Predicted class: {result}")

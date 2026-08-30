import torch
import torch.optim as optim
import torch.nn as nn

from src.data import get_dataloaders
from src.model import build_model
from src.config import DEVICE, NUM_EPOCHS, LEARNING_RATE, MODEL_DIR

def train():
    train_loader, _, class_to_idx = get_dataloaders()
    model = build_model(len(class_to_idx)).to(DEVICE)

    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        correct = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == y).sum().item()

        print(f"Epoch {epoch+1}: acc={correct/len(train_loader.dataset):.3f}")

    torch.save({
        "model": model.state_dict(),
        "class_to_idx": class_to_idx
    }, MODEL_DIR / "crop_disease_model.pth")

if __name__ == "__main__":
    train()

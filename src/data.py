from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import torch

from src.config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, RANDOM_SEED

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_dataloaders():
    train_ds = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    targets = train_ds.targets

    train_idx, val_idx = train_test_split(
        range(len(targets)),
        test_size=0.2,
        stratify=targets,
        random_state=RANDOM_SEED
    )

    train_set = Subset(train_ds, train_idx)
    val_set = Subset(val_ds, val_idx)

    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False),
        train_ds.class_to_idx
    )

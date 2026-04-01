from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score

# ----------------------------
# Paths
# ----------------------------
TRAIN_CSV = Path("outputs/train_brands.csv")
TEST_CSV = Path("outputs/test_brands.csv")
MODEL_DIR = Path("outputs/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Settings
# ----------------------------
BATCH_SIZE = 32
IMAGE_SIZE = 224
EPOCHS = 3
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # safer on Mac
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BrandDataset(Dataset):
    def __init__(self, csv_file: Path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["full_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def main():
    print("Using device:", DEVICE)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = BrandDataset(TRAIN_CSV, transform=train_transform)
    test_dataset = BrandDataset(TEST_CSV, transform=test_transform)

    num_classes = int(train_dataset.df["label"].nunique())
    print("Number of classes:", num_classes)
    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, DEVICE
        )

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.4f}")

    model_path = MODEL_DIR / "resnet18_brand_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
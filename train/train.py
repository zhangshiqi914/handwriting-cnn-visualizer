import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from tqdm import tqdm

from train.model import CNNForMNIST


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total

        progress_bar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "acc": f"{acc:.4f}",
        })

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def save_checkpoint(model, best_acc, checkpoint_path):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    torch.save(
        {
            "task": "digit",
            "dataset_name": "MNIST",
            "model_name": "CNNForMNIST",
            "model_state_dict": model.state_dict(),
            "num_classes": 10,
            "class_names": [str(i) for i in range(10)],
            "best_acc": best_acc,
            "image_size": 28,
            "input_channels": 1,
        },
        checkpoint_path,
    )

    print(f"Saved best model to: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Dataset root directory.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoint/MNIST/best.pth",
        help="Path to save the best model checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Use 0 on Windows to avoid DataLoader multiprocessing issues.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Task: MNIST digit recognition")
    print("Model: CNNForMNIST")
    print("Num classes: 10")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10),
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.MNIST(
        root=args.data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = CNNForMNIST(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
        )

        test_loss, test_acc = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, "
            f"test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                model=model,
                best_acc=best_acc,
                checkpoint_path=args.checkpoint,
            )

    print("Training finished.")
    print(f"Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
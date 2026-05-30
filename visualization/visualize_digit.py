import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from train.model import CNNForMNIST


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    num_classes = checkpoint["num_classes"]

    model = CNNForMNIST(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


def show_probability_bar(probabilities, class_names):
    plt.figure(figsize=(8, 4))

    x = list(range(len(class_names)))
    y = probabilities.cpu().numpy()

    plt.bar(x, y)
    plt.xticks(x, class_names)
    plt.ylim(0, 1)
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")

    plt.tight_layout()
    plt.show()


def show_feature_maps(feature_tensor, layer_name, max_channels=8):
    """
    feature_tensor shape:
        [1, C, H, W]
    """

    feature_tensor = feature_tensor.squeeze(0).cpu()

    num_channels = min(max_channels, feature_tensor.shape[0])

    plt.figure(figsize=(12, 3))

    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)
        plt.imshow(feature_tensor[i], cmap="gray")
        plt.axis("off")
        plt.title(f"{layer_name}-{i}")

    plt.tight_layout()
    plt.show()


def show_input_image(image_tensor):
    """
    image_tensor shape:
        [1, 28, 28]

    MNIST normalization:
        mean = 0.1307
        std = 0.3081
    """

    image = image_tensor.squeeze(0).cpu()
    image = image * 0.3081 + 0.1307

    plt.figure(figsize=(3, 3))
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title("Input Image")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoint/MNIST/best.pth",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of MNIST test image.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    test_dataset = datasets.MNIST(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform,
    )

    image, label = test_dataset[args.index]

    model, checkpoint = load_model(args.checkpoint, device)

    class_names = checkpoint.get("class_names", [str(i) for i in range(10)])

    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, feature_maps = model(input_tensor, return_features=True)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
        pred_id = probabilities.argmax().item()
        pred_name = class_names[pred_id]

    print(f"True label: {label}")
    print(f"Predicted: {pred_name}")
    print(f"Confidence: {probabilities[pred_id].item():.4f}")

    show_input_image(image)
    show_probability_bar(probabilities, class_names)

    show_feature_maps(feature_maps["conv1"], "Conv1", max_channels=8)
    show_feature_maps(feature_maps["conv2"], "Conv2", max_channels=8)


if __name__ == "__main__":
    main()
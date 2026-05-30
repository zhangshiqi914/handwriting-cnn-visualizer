import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

from train.model import CNNForMNIST


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_PATH = ROOT_DIR / "frontend" / "index.html"
CHECKPOINT_PATH = ROOT_DIR / "checkpoint" / "MNIST" / "best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI()


class PredictRequest(BaseModel):
    image: str


def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model = CNNForMNIST(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    class_names = checkpoint.get("class_names", [str(i) for i in range(10)])

    return model, class_names, checkpoint


MODEL, CLASS_NAMES, CHECKPOINT = load_model()


def decode_canvas_image(image_base64: str) -> Image.Image:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    return image


def image_array_to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8), mode="L")

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def preprocess_image(image: Image.Image):
    """
    Frontend canvas:
        280 x 280

    Model input:
        28 x 28

    Processing:
        1. Detect white stroke region
        2. Crop it
        3. Resize while keeping aspect ratio
        4. Center it on a 28x28 black canvas
        5. Normalize with MNIST mean/std
    """

    image = image.convert("L")
    arr = np.array(image)

    coords = np.argwhere(arr > 20)

    if coords.size == 0:
        arr_28_uint8 = np.zeros((28, 28), dtype=np.uint8)
    else:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        padding = 20

        y_min = max(y_min - padding, 0)
        x_min = max(x_min - padding, 0)
        y_max = min(y_max + padding, arr.shape[0] - 1)
        x_max = min(x_max + padding, arr.shape[1] - 1)

        cropped = arr[y_min:y_max + 1, x_min:x_max + 1]
        cropped_img = Image.fromarray(cropped)

        w, h = cropped_img.size

        if w > h:
            new_w = 20
            new_h = max(1, int(h * 20 / w))
        else:
            new_h = 20
            new_w = max(1, int(w * 20 / h))

        cropped_img = cropped_img.resize(
            (new_w, new_h),
            Image.Resampling.LANCZOS,
        )

        canvas_28 = Image.new("L", (28, 28), color=0)

        left = (28 - new_w) // 2
        top = (28 - new_h) // 2

        canvas_28.paste(cropped_img, (left, top))

        arr_28_uint8 = np.array(canvas_28).astype(np.uint8)

    input_base64 = image_array_to_base64(arr_28_uint8)

    arr_28 = arr_28_uint8.astype(np.float32) / 255.0
    arr_28 = (arr_28 - 0.1307) / 0.3081

    tensor = torch.tensor(arr_28, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    return tensor.to(DEVICE), input_base64


def tensor_to_base64_image(feature_2d: torch.Tensor) -> str:
    """
    Convert one feature map channel to a base64 PNG image.

    The original spatial size is preserved:
        conv1: 28 x 28
        conv2: 14 x 14

    Values are normalized to 0~255 only for browser display.
    """

    feature = feature_2d.detach().cpu().numpy()

    min_val = feature.min()
    max_val = feature.max()

    if max_val - min_val < 1e-6:
        feature = np.zeros_like(feature)
    else:
        feature = (feature - min_val) / (max_val - min_val)

    feature = (feature * 255).astype(np.uint8)

    return image_array_to_base64(feature)


def encode_feature_maps(feature_maps, max_channels: int = 8):
    result = {}

    for layer_name, tensor in feature_maps.items():
        tensor = tensor.squeeze(0)

        channels = []
        num_channels = min(max_channels, tensor.shape[0])

        for i in range(num_channels):
            channels.append(tensor_to_base64_image(tensor[i]))

        result[layer_name] = channels

    return result


@app.get("/")
def index():
    return FileResponse(FRONTEND_PATH)


@app.post("/predict")
def predict(request: PredictRequest):
    image = decode_canvas_image(request.image)
    input_tensor, processed_input = preprocess_image(image)

    with torch.no_grad():
        logits, feature_maps = MODEL(input_tensor, return_features=True)
        probabilities = F.softmax(logits, dim=1).squeeze(0)

    pred_id = int(torch.argmax(probabilities).item())
    pred_name = CLASS_NAMES[pred_id]
    confidence = float(probabilities[pred_id].item())

    return {
        "task": "digit",
        "dataset_name": CHECKPOINT.get("dataset_name", "MNIST"),
        "model_name": CHECKPOINT.get("model_name", "CNNForMNIST"),
        "prediction": pred_name,
        "confidence": confidence,
        "processed_input": processed_input,
        "probabilities": [
            {
                "class_name": CLASS_NAMES[i],
                "probability": float(probabilities[i].item()),
            }
            for i in range(len(CLASS_NAMES))
        ],
        "feature_maps": encode_feature_maps(feature_maps, max_channels=8),
        "device": str(DEVICE),
    }
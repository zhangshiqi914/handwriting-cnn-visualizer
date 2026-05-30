# Handwriting CNN Visualizer

A real-time CNN visualization system for handwritten digit recognition.

This project uses a convolutional neural network trained on the MNIST dataset. Users can draw a digit on a browser canvas, and the system will perform real-time inference while visualizing the predicted digit, confidence score, probability distribution, processed 28×28 model input, and intermediate convolutional feature maps.

## Features

* Handwritten digit recognition based on MNIST
* Real-time browser canvas input
* PyTorch CNN inference
* Probability bar visualization
* Processed 28×28 model input display
* Conv1 and Conv2 feature map visualization
* FastAPI backend
* HTML/CSS/JavaScript frontend
* GPU or CPU inference support

## Demo Interface

The web interface contains:

* A drawing canvas for handwritten digit input
* Real-time predicted digit
* Confidence score
* Probability distribution over digits 0-9
* Processed 28×28 model input
* Feature maps from convolutional layers

## Environment

Recommended environment:

```text
Python 3.10
PyTorch 2.5.1
torchvision 0.20.1
CUDA 11.8
```

Create a conda environment:

```bash
conda create -n handwriting-cnn python=3.10 -y
conda activate handwriting-cnn
```

Install PyTorch with CUDA 11.8:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url
```

Install other dependencies:

```bash
pip install -r requirements.txt
```

A minimal `requirements.txt` can be:

```text
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
numpy
pillow
fastapi
uvicorn
pydantic
matplotlib
tqdm
```

## Train the Model

Train the MNIST digit recognition model:

```bash
python -m train.train --epochs 5 --batch-size 128
```

The best model checkpoint will be saved to:

```text
checkpoint/MNIST/best.pth
```

The training script automatically downloads MNIST to:

```text
data/
```

## Run the Web App

Start the FastAPI backend:

```bash
python -m uvicorn backend.main:app --reload
```

Open the web page in your browser:

```text
http://127.0.0.1:8000
```

Then draw a digit on the canvas. The system will display:

* predicted digit
* confidence score
* probability distribution
* processed 28×28 model input
* Conv1 feature maps
* Conv2 feature maps

## Offline Visualization

You can also visualize predictions and feature maps using a sample from the MNIST test set:

```bash
python -m visualization.visualize_digit --index 0
```

Use another test sample:

```bash
python -m visualization.visualize_digit --index 10
```

This script displays:

* original MNIST input image
* prediction probability bar chart
* Conv1 feature maps
* Conv2 feature maps

## Checkpoint

The project expects the trained MNIST checkpoint at:

```text
checkpoint/MNIST/best.pth
```

If this file does not exist, train the model first:

```bash
python -m train.train --epochs 5 --batch-size 128
```

## Image Preprocessing

The browser canvas is larger than the CNN input size. The backend preprocesses the drawn image by:

1. detecting the handwritten stroke region
2. cropping the digit
3. resizing it while keeping the aspect ratio
4. centering it on a 28×28 black canvas
5. normalizing it with MNIST mean and standard deviation

This helps reduce the distribution gap between browser handwriting and MNIST images.

## Display Note: Prediction vs. Probability Bar

The web interface uses two different display strategies:

* `Prediction` and `Confidence` are based on the latest raw inference result returned by the backend.
* The probability bars are visually smoothed on the frontend to make real-time updates less abrupt.

Because of this smoothing, the confidence value and the probability bar may not match perfectly at every instant while the user is drawing. For example, the backend may return a confidence of `99.9%` for digit `7`, while the probability bar for `7` is still gradually moving toward that value.

This is an intentional UI design choice. The raw prediction is used for the final predicted class and confidence score, while the probability bars are smoothed only to improve visual continuity during real-time interaction.

## Future Work

* Add EMNIST letter recognition
* Support uppercase and lowercase letter classification
* Add model selection in the web interface
* Add more CNN architectures for comparison
* Add confusion matrix and evaluation reports
* Improve preprocessing for browser-drawn digits

## License

This project is for learning and research purposes.

# Handwritten Digit Recognizer (神经网络)


A neural network implementation built from scratch using NumPy to recognize handwritten digits trained on the MNIST dataset. This project demonstrates core deep learning concepts including forward propagation, backpropagation, activation functions, and optimization techniques.

<p align="center">
  <img src="https://github.com/phantomvoyager4/HandwrittenDigitsRecognizer/blob/main/demo.gif" width="250" />
</p>

## Table of Contents

[Overview](#overview)
[Architecture](#architecture)
[Technologies & Dependencies](#technologies--dependencies)
[Project Structure](#project-structure)
[Core Components](#core-components)
[How It Works](#how-it-works)
[App System](#app-system)
[Usage](#usage)
[Learning outcomes](#learning-outcomes)
[Future enchacements](#future-enchancements)
[Dataset](#dataset)
[References](#references)

## Overview

**Key Performance Metrics:**
- Accuracy: ~95%
- Training Epochs: 1001
- Dataset: MNIST (60,000 training images, 10,000 test images)
- Input Size: 784 pixels (28×28 images)
- Output Classes: 10 (digits 0-9)

## Architecture

The neural network consists of:
- **Input Layer**: 784 neurons (flattened 28×28 pixel images)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one per digit)

## Technologies & Dependencies

- **Python 3.8+**
- **NumPy 1.24.3**: Numerical computing and matrix operations
- **idx2numpy 1.2.3**: For loading and converting MNIST IDX binary format
- **Pillow 10.0.0**: Image processing for UI input

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/digit-recognizer.git
cd digit-recognizer
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
digit-recognizer/
├── HDR.py              # Neural network implementation
├── train.py            # Training script
├── App.py              # Interactive GUI application
├── README.md           # This file
└── dataset/            # MNIST dataset files
    └── (binary IDX format files)
```

## Core Components

### HDR.py - Neural Network Implementation

#### `data_handling(pathimages, pathlabels)`
Loads MNIST data from IDX binary format and normalizes pixel values to [0, 1] range.

#### `Layer` Class
Implements a fully connected neural network layer with:
- **Forward Propagation**: `fpropagation(input)` - Computes weighted sum and bias
- **Backward Propagation**: `backward(backwardpass)` - Calculates gradients for weights, biases, and inputs
- Weights initialized with small random values for stability

#### `Activation` Class
Implements ReLU (Rectified Linear Unit) activation function:
- **Forward**: `max(0, x)` - Non-linear activation
- **Backward**: Gradient passing for backpropagation

#### `Softmax` Class
Converts raw network output to probability distribution:
- Numerically stable implementation with max subtraction
- Output sums to 1 across all classes

#### `Loss` Class
Computes categorical cross-entropy loss:
- Formula: $-\ln(y_{pred})$ for the correct class
- Clipping predictions to prevent log(0) errors

#### `Backpropagation` Class
Combines Softmax and Loss for efficient gradient calculation:
- Integrates forward pass through activation and loss
- Computes backward gradients for optimization

#### `Optimizer` Class
Implements basic Stochastic Gradient Descent (SGD):
- Adjusts weights and biases using learning rate
- Simple parameter update rule: $\theta = \theta - \alpha \cdot \nabla\theta$

### train.py - Training Script

Trains the neural network with configurable parameters (learning rate 0.5, 400 epochs). Includes periodic loss and accuracy logging.

## How It Works

**Forward Propagation**: Input (784 features) → Layer 1 (128 neurons, ReLU) → Layer 2 (64 neurons, ReLU) → Output (10 classes, Softmax) → Loss

**Backward Propagation**: Computes gradients through Softmax/Loss, propagates back through both hidden layers, and updates all weights and biases using SGD.

### Key Mathematics

**ReLU**: $f(x) = \max(0, x)$ <br> **Softmax**: $\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ <br> **Cross-Entropy Loss**: $L = -\sum_i y_i \ln(\hat{y}_i)$

## App System

### App.py - Interactive GUI Application

An interactive Tkinter-based GUI that loads a trained model and allows real-time digit prediction from user drawings.

**Key Features:**
- **Model Loading**: `transfer_network()` loads pretrained weights from saved `.npz` files
- **Network Pipeline**: Implements the full neural network using loaded layers and activations
- **Drawing Canvas**: 300×300 pixel drawing area with black ink on white background
- **Image Processing**: Converts hand-drawn images to normalized 28×28 pixel format matching MNIST specifications
  - Grayscales and inverts the image
  - Centers the digit using bounding box
  - Scales to 20×20 and pads to 28×28
  - Normalizes pixel values to [0, 1]
- **Real-time Prediction**: Processes drawn image through network and displays predicted digit
- **Controls**: "Predict" button triggers recognition, "Clear" button resets canvas

**Workflow:**
1. App initializes with pretrained model
2. User draws digit on canvas
3. Clicking "Predict" sends image through preprocessing pipeline
4. Network pipeline forwards the normalized image through all layers
5. Output layer produces 10 class probabilities via Softmax
6. `argmax` selects the highest probability digit
7. Result displayed with confidence percentage
8. Optional: Save drawn image with prediction metadata

**Network Pipeline Components:**
```
Input (784) → Layer1 (128) → ReLU → Layer2 (64) → ReLU → Output (10) → Softmax → Prediction
```

**Key Features:**
- Checkboxes for optional image saving
- Real-time prediction display
- Confidence percentage shown with prediction
- Preprocessing: grayscale, invert, center, resize, normalize

## Usage

### Running the GUI Application

```bash
python source/App.py
```

Launch the interactive digit recognizer:
1. Click "Select model" and load a pretrained model from `models_data_storage/`
2. Draw a digit (0-9) on the white canvas
3. Check "Save image?" if you want to save the drawing
4. Click "Predict" to see the AI's prediction with confidence
5. Click "Clear" to reset and draw again

**Example:** Draw a "7" → Prediction: 7 (99.45%)

### Training the Model

```bash
python train.py
```

Train a new model from scratch (results logged every 100 epochs).

**Example Output:**
```
Epoch: 0, Loss: 2.297, Accuracy: 0.107
Epoch: 100, Loss: 0.218, Accuracy: 0.935
Epoch: 200, Loss: 0.143, Accuracy: 0.956
Epoch: 300, Loss: 0.115, Accuracy: 0.963
```

Trained models are saved to `models_data_storage/` as NumPy `.npz` files.

## Performance Metrics

**Available Models:**
- **Model 1**: 95.99% accuracy (0.5 learning rate)
- **Model 2**: 97.04% accuracy (optimized training)

**Training Configuration:**
- **Epochs**: 1001
- **Learning Rate**: 0.5
- **Optimizer**: Stochastic Gradient Descent
- **Batch Size**: Full batch
- **Dataset**: MNIST (60,000 training images)
- **Validation**: 10,000 test images

## Learning Outcomes

This project demonstrates:
- **Neural Network Architecture**: Building networks from scratch without frameworks
- **Forward Propagation**: How data flows through layers and activations
- **Backpropagation**: Computing gradients and updating weights
- **Activation Functions**: ReLU for hidden layers, Softmax for output
- **Loss Functions**: Cross-entropy loss for multi-class classification
- **Image Processing**: Preprocessing, normalization, and transformation
- **End-to-End ML Systems**: Data → Model → Prediction → Visualization


## Dataset

The MNIST dataset contains 70,000 handwritten digit images (28×28 pixels):
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Format**: Binary IDX format (loaded via idx2numpy)
- **Pixel Values**: 0-255 (normalized to [0, 1] for training)

Source: [MNIST Database by Yann LeCun](http://yann.lecun.com/exdb/mnist/)


## References

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen
- [ReLU Activation Function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [Softmax and Cross-Entropy](https://en.wikipedia.org/wiki/Softmax_function)
- [Backpropagation Algorithm](https://en.wikipedia.org/wiki/Backpropagation)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [idx2numpy Library](https://github.com/ivanyu/idx2numpy)


**Note**: This project was built to understand and dive into neural network mechanics. For production applications, use established frameworks like PyTorch, TensorFlow, or scikit-learn.



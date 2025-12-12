# Handwritten Digit Recognizer (神经网络)

**📄 License: Free to Use** - This project is freely available for educational and personal use.

A neural network implementation built from scratch using NumPy to recognize handwritten digits from the MNIST dataset. This project demonstrates core deep learning concepts including forward propagation, backpropagation, activation functions, and optimization techniques.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technologies & Dependencies](#technologies--dependencies)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [How It Works](#how-it-works)
- [App System](#app-system)
- [Usage](#usage)
- [Dataset](#dataset)
- [References](#references)
- [License](#license)

## Overview

This is an educational implementation of a neural network designed to classify handwritten digits (0-9) with approximately **93% accuracy** after 400 training epochs. The entire model is implemented from first principles using only NumPy and idx2numpy, without relying on deep learning frameworks like TensorFlow or PyTorch.

**Key Performance Metrics:**
- Accuracy: ~93%
- Training Epochs: 400
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

- **Python 3.x**
- **NumPy**: Numerical computing and matrix operations
- **idx2numpy**: For loading and converting MNIST IDX binary format

Install dependencies:
```bash
pip install numpy idx2numpy
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

**ReLU**: $f(x) = \max(0, x)$ | **Softmax**: $\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | **Cross-Entropy Loss**: $L = -\sum_i y_i \ln(\hat{y}_i)$

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
1. App initializes with pretrained model (default: model_1_95.28_0.5.npz)
2. User draws digit on canvas
3. Clicking "Predict" sends image through preprocessing pipeline
4. Network pipeline forwards the normalized image through all layers
5. Output layer produces 10 class probabilities via Softmax
6. `argmax` selects the highest probability digit
7. Result displayed on GUI

**Network Pipeline Components:**
```
Input (784) → Layer1 (128) → ReLU → Layer2 (64) → ReLU → Output (10) → Softmax → Prediction
```

## Usage

### Running the GUI Application

```bash
python App.py
```

Launch the interactive digit recognizer. Draw a digit and click "Predict" to see the AI's prediction.

### Training the Model

```bash
python train.py
```

Train a new model (results logged every 100 epochs).

**Example Output:**
```
Epoch: 0, Loss: 2.297, Accuracy: 0.107
Epoch: 100, Loss: 0.218, Accuracy: 0.935
Epoch: 200, Loss: 0.143, Accuracy: 0.956
Epoch: 300, Loss: 0.115, Accuracy: 0.963
```

## Learning Outcomes

This project demonstrates:
- Neural network fundamentals from scratch
- Forward and backward propagation mechanics
- Activation functions (ReLU, Softmax) and loss functions
- Matrix operations with NumPy
- Image preprocessing and normalization
- End-to-end ML system integration

## Future Enhancements

- Confidence scores with predictions
- Test set evaluation script
- Weight visualization
- Hyperparameter tuning utilities
- Batch normalization and regularization (L1/L2)
- Additional model saving/loading options

## Dataset

The MNIST dataset contains 70,000 handwritten digit images (28×28 pixels, 784 features when flattened):
- 60,000 training images
- 10,000 test images
- Pixel values: 0-255 (normalized to [0, 1])

Source: [MNIST Database](http://yann.lecun.com/exdb/mnist/)

## References

- [ReLU Activation Function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [Softmax & Cross-Entropy](https://en.wikipedia.org/wiki/Softmax_function)
- [Backpropagation Algorithm](https://en.wikipedia.org/wiki/Backpropagation)
- [idx2numpy](https://github.com/ivanyu/idx2numpy)

## License

Free to use for educational and personal projects. This implementation demonstrates neural network principles using pure Python and NumPy.



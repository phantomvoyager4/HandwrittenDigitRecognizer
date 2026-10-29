import numpy as np


def read_idx(path):
    #IDX format: 2 zero bytes, a data type byte, the number of dimensions,
    #then one big-endian uint32 per dimension, then the raw uint8 values
    with open(path, 'rb') as file:
        data = file.read()
    n_dimensions = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], 'big') for i in range(n_dimensions))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * n_dimensions).reshape(shape)


def data_handling(pathimages, pathlabels):
    images = read_idx(pathimages).reshape((-1, 784)) / 255.0
    labels = read_idx(pathlabels).astype(np.int64)
    return images, labels


def train_validation_split(images, labels, validation_size, seed):
    #the validation set is cut from the training set, so the test set is used only once, at the very end
    order = np.random.default_rng(seed).permutation(len(labels))
    validation, train = order[:validation_size], order[validation_size:]
    return images[train], labels[train], images[validation], labels[validation]

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, input):
        self.input = input 
        self.output = np.dot(input, self.weights) + self.biases  
        return self.output
    def backward(self, dvalues):
        self.dweights = np.dot(self.input.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)
        return self.dinputs

class Activation_RELU:
    def forward(self, input):
        self.input = input
        self.output= np.maximum(0, input)
        return self.output
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.input <= 0] = 0
        return self.dinputs

class Softmax:
    def forward (self, input):
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        sum = np.sum(exp, axis=1, keepdims=True)
        self.output = exp/sum
        return self.output
    
class Loss:
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 0.0000001, 0.99999999)
        predicted = y_pred[range(y_pred.shape[0]), y_true]
        loss = -np.log(predicted)
        mean_loss = np.mean(loss)
        return mean_loss

class Backpropagation:
    def __init__(self):
        self.activation = Softmax()
        self.loss = Loss()
    def forward(self, input, labels):
        self.result = self.activation.forward(input)
        self.lossf = self.loss.forward(self.result, labels)
        return self.lossf
    def backward(self, labels):
        probabilities_normalized = self.result.copy()
        probabilities_normalized[range(probabilities_normalized.shape[0]), labels] -= 1
        dvalues = probabilities_normalized / probabilities_normalized.shape[0]
        return dvalues

class Optimizer:
    #plain gradient descent update; whether it is SGD or full-batch depends on what train.py feeds it
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
    def adjust_parameters(self, layer):
        layer.weights -= (self.learning_rate * layer.dweights)
        layer.biases -= (self.learning_rate * layer.dbiases)
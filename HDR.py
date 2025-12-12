import numpy as np
import idx2numpy


def data_handling(pathimages, pathlabels):
    images = idx2numpy.convert_from_file(pathimages).reshape((60000, 784))
    images = images / 255.0
    labels = idx2numpy.convert_from_file(pathlabels)
    return images, labels

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
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

class Activation:
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
    def __init__(self, input):
        self.learning_rate = input
    def adjust_parameters(self, layer):
        layer.weights -= (self.learning_rate * layer.dweights)
        layer.biases -= (self.learning_rate * layer.dbiases)

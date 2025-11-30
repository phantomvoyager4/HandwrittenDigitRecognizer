import numpy as np
import idx2numpy

def data_handling(pathimages, pathlabels):
    images = idx2numpy.convert_from_file(pathimages).reshape((60000, 784))
    images = images / 255.0
    labels = idx2numpy.convert_from_file(pathlabels)
    map = np.identity(10)
    labels_encoded = map[labels]
    print(images.shape, labels_encoded.shape) #(60000, 784) (60000, 10)
    return images, labels_encoded



pathimagess = 'dataset/train-images.idx3-ubyte'
pathlabelss = 'dataset/train-labels.idx1-ubyte'
data_handling(pathimagess, pathlabelss)

#NEURONS LOGIC
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))
    
    def fpropagation(self, input):
        self.output = np.dot(input, self.weights) + self.biases  
        return self.output

#ACTIVATE NEURONS USING RELU
class Activation:
    def forward(self, input):
        self.output= np.maximum(0, input)
        return self.output

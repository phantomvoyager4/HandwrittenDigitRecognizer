import numpy as np
from HDR import data_handling, Layer, Activation, Optimizer, Backpropagation
import os #for model versioning

pathimagess = 'dataset/train-images.idx3-ubyte'
pathlabelss = 'dataset/train-labels.idx1-ubyte'
images, labels = data_handling(pathimagess, pathlabelss)

hidden_layer1 = Layer(n_inputs=784, n_neurons=128)
activation1 = Activation()
activation2 = Activation()
hidden_layer2 = Layer(n_inputs=128, n_neurons=64)
output_layer = Layer(n_inputs=64, n_neurons=10)
loss_activation = Backpropagation()
optimization = Optimizer(0.5)
network = [hidden_layer1, activation1, hidden_layer2, activation2, output_layer]
trainable_layers = [hidden_layer1, hidden_layer2, output_layer]

for epoch in range (401):
    current_input = images
    for layer in network:
        iteration = layer.forward(current_input)
        current_input = iteration

    loss = loss_activation.forward(current_input, labels)

    highestchangebyrow = np.argmax(current_input, axis=1)
    comparison = np.equal(highestchangebyrow, labels)
    accuracy = np.mean(comparison)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")
        
    dvalues = loss_activation.backward(labels)
    for layer in reversed(network):
        iterate = layer.backward(dvalues) 
        dvalues = iterate

    for layer in trainable_layers:
        optimization.adjust_parameters(layer)
    
models_folder_path = "models_data_storage"
model_version_number = 1

while True:
    model_path = f"{models_folder_path}/model_{model_version_number}"
    if os.path.exists(model_path):
        model_version_number += 1
    else:
        break

os.makedirs(model_path)
modelaccuracy = round(accuracy * 100, 2)
model_lr = optimization.learning_rate

#model pathing: model_[version number]_[model accuracy]_[model learning rate]
fullpath = f"{model_path}/model_{model_version_number}_{modelaccuracy}_{model_lr}.npz"
savedata = {}
for index,layer in enumerate(trainable_layers):
    wbid = index + 1
    savedata.update({f"w{wbid}" : layer.weights, f"b{wbid}" : layer.biases})
np.savez(fullpath, **savedata)
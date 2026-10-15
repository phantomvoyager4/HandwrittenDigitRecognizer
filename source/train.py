import numpy as np
from model import data_handling, Layer, Activation_RELU, Optimizer, Backpropagation
from plot import AccuracyPlot
import os #for model versioning

pathimagess = 'dataset/train-images.idx3-ubyte'
pathlabelss = 'dataset/train-labels.idx1-ubyte'
test_images_path = 'dataset/t10k-images.idx3-ubyte'
test_labels_path = 'dataset/t10k-labels.idx1-ubyte'
testimages, testlabels = data_handling(test_images_path, test_labels_path)
images, labels = data_handling(pathimagess, pathlabelss)

hidden_layer1 = Layer(n_inputs=784, n_neurons=128)

activation1 = Activation_RELU()
activation2 = Activation_RELU()
 

hidden_layer2 = Layer(n_inputs=128, n_neurons=64)
output_layer = Layer(n_inputs=64, n_neurons=10)
loss_activation = Backpropagation()
optimization = Optimizer(0.5)
network = [hidden_layer1, activation1, hidden_layer2, activation2, output_layer]
trainable_layers = [hidden_layer1, hidden_layer2, output_layer]

epochs = 1000
accuracy_plot = AccuracyPlot(epochs)

print('Starting training')
for epoch in range (epochs):
    current_input = images
    for layer in network:
        iteration = layer.forward(current_input)
        current_input = iteration

    loss = loss_activation.forward(current_input, labels)

    highest_change_by_row = np.argmax(current_input, axis=1)
    comparison = np.equal(highest_change_by_row, labels)
    accuracy = np.mean(comparison)
    accuracy_plot.add_train_accuracy(epoch, accuracy)
        
    dvalues = loss_activation.backward(labels)
    for layer in reversed(network):
        iterate = layer.backward(dvalues) 
        dvalues = iterate

    for layer in trainable_layers:
        optimization.adjust_parameters(layer)

    if epoch % 100 == 0 or epoch == epochs - 1:
        training_current_input = testimages
        for layer in network:
            training_current_input = layer.forward(training_current_input)
        predicted = np.argmax(training_current_input, axis=1)
        comparison_test = np.equal(predicted, testlabels)
        accuracy_test = np.mean(comparison_test)
        accuracy_plot.add_test_accuracy(epoch, accuracy_test)
        print(f"Epoch: {epoch}, Loss: {loss:.3f},Accuracy:{accuracy:.3f} Test_Accuracy: {accuracy_test:.3f}")
    accuracy_plot.refresh()
    
    
models_folder_path = "models_data_storage"
model_version_number = 1

while True:
    model_path = f"{models_folder_path}/model_{model_version_number}"
    if os.path.exists(model_path):
        model_version_number += 1
    else:
        break

os.makedirs(model_path)
modelaccuracy = round(accuracy_test * 100, 2)
model_lr = optimization.learning_rate

#model pathing: model_[version number]_[model accuracy]_[model learning rate]
fullpath = f"{model_path}/model_{model_version_number}_{modelaccuracy}_{model_lr}.npz"
savedata = {}
for index,layer in enumerate(trainable_layers):
    wbid = index + 1
    savedata.update({f"w{wbid}" : layer.weights, f"b{wbid}" : layer.biases})
np.savez(fullpath, **savedata)

accuracy_plot.save(f"{model_path}/model_{model_version_number}_accuracy.png")
accuracy_plot.show()

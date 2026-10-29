import argparse
import os #for model versioning
import time

import numpy as np

from model import data_handling, train_validation_split, Layer, Activation_RELU, Optimizer, Backpropagation


parser = argparse.ArgumentParser(description='Train the digit recognizer on MNIST')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--learning-rate', type=float, default=0.1)
parser.add_argument('--batch-size', type=int, default=64, help='0 trains on the whole training set at once (full-batch gradient descent)')
parser.add_argument('--validation-size', type=int, default=10000, help='images cut from the training set to choose the best epoch')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--no-plot', action='store_true', help='no live plot window; the plot is still saved with the model')
args = parser.parse_args()

if args.no_plot:
    import matplotlib
    matplotlib.use('Agg')
from plot import AccuracyPlot
from raport import create_raport, save_raport

np.random.seed(args.seed) #weight initialisation in Layer uses the global generator
rng = np.random.default_rng(args.seed) #order of the training images in every epoch

pathimagess = 'dataset/train-images.idx3-ubyte'
pathlabelss = 'dataset/train-labels.idx1-ubyte'
test_images_path = 'dataset/t10k-images.idx3-ubyte'
test_labels_path = 'dataset/t10k-labels.idx1-ubyte'
testimages, testlabels = data_handling(test_images_path, test_labels_path)
all_images, all_labels = data_handling(pathimagess, pathlabelss)
images, labels, valimages, vallabels = train_validation_split(all_images, all_labels, args.validation_size, args.seed)

hidden_layer1 = Layer(n_inputs=784, n_neurons=128)
activation1 = Activation_RELU()
hidden_layer2 = Layer(n_inputs=128, n_neurons=64)
activation2 = Activation_RELU()
output_layer = Layer(n_inputs=64, n_neurons=10)
loss_activation = Backpropagation()
optimization = Optimizer(learning_rate=args.learning_rate)
network = [hidden_layer1, activation1, hidden_layer2, activation2, output_layer]
trainable_layers = [hidden_layer1, hidden_layer2, output_layer]


def forward(inputs):
    for layer in network:
        inputs = layer.forward(inputs)
    return inputs


def predict(inputs):
    return np.argmax(forward(inputs), axis=1)


def accuracy(inputs, targets):
    return np.mean(predict(inputs) == targets)


def snapshot():
    return [(layer.weights.copy(), layer.biases.copy()) for layer in trainable_layers]


epochs = args.epochs
batch_size = args.batch_size if args.batch_size > 0 else len(labels)
optimizer_name = 'Full-batch gradient descent' if batch_size >= len(labels) else f'Mini-batch SGD (batch size {batch_size})'
log_every = max(1, epochs // 10)
accuracy_plot = AccuracyPlot(epochs, live=not args.no_plot)

best_val_accuracy, best_epoch, best_weights = -1.0, None, None

print(f'Starting training: {optimizer_name}, learning rate {args.learning_rate}, {epochs} epochs')
print(f'Training set {len(labels)}, validation set {len(vallabels)}, test set {len(testlabels)} (used once, after training)')
start_time = time.time()
for epoch in range(epochs):
    order = rng.permutation(len(labels))
    batch_losses = []
    for start in range(0, len(labels), batch_size):
        batch = order[start:start + batch_size]
        output = forward(images[batch])
        batch_losses.append(loss_activation.forward(output, labels[batch]) * len(batch))

        dvalues = loss_activation.backward(labels[batch])
        for layer in reversed(network):
            dvalues = layer.backward(dvalues)
        for layer in trainable_layers:
            optimization.adjust_parameters(layer)

    loss = sum(batch_losses) / len(labels)
    train_accuracy = accuracy(images, labels)
    val_accuracy = accuracy(valimages, vallabels)
    accuracy_plot.add_train_accuracy(epoch, train_accuracy)
    accuracy_plot.add_validation_accuracy(epoch, val_accuracy)

    #the saved model is the one from the epoch with the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy, best_epoch, best_weights = val_accuracy, epoch, snapshot()

    if epoch % log_every == 0 or epoch == epochs - 1:
        print(f"Epoch: {epoch}, Loss: {loss:.3f}, Accuracy: {train_accuracy:.4f}, Validation_Accuracy: {val_accuracy:.4f}")
    accuracy_plot.refresh()
training_time = time.time() - start_time

for layer, (weights, biases) in zip(trainable_layers, best_weights):
    layer.weights, layer.biases = weights, biases
test_predictions = predict(testimages)
test_accuracy = np.mean(test_predictions == testlabels)
print(f'Best epoch: {best_epoch} (validation accuracy {best_val_accuracy:.4f}), test accuracy: {test_accuracy:.4f}')

models_folder_path = "models_data_storage"
model_version_number = 1

while True:
    model_path = f"{models_folder_path}/model_{model_version_number}"
    if os.path.exists(model_path):
        model_version_number += 1
    else:
        break

os.makedirs(model_path)
modelaccuracy = round(test_accuracy * 100, 2)
model_lr = optimization.learning_rate

#model pathing: model_[version number]_[test accuracy]_[model learning rate]
fullpath = f"{model_path}/model_{model_version_number}_{modelaccuracy}_{model_lr}.npz"
savedata = {}
for index, layer in enumerate(trainable_layers):
    wbid = index + 1
    savedata.update({f"w{wbid}": layer.weights, f"b{wbid}": layer.biases})
np.savez(fullpath, **savedata)

accuracy_plot.save(f"{model_path}/model_{model_version_number}_accuracy.png")
#per-epoch history, so plots can be redrawn later without training again
with open(f"{model_path}/model_{model_version_number}_history.csv", 'w') as history:
    history.write("epoch,train_accuracy,validation_accuracy\n")
    for epoch, train_acc, val_acc in zip(accuracy_plot.train_epochs, accuracy_plot.train_accuracy, accuracy_plot.validation_accuracy):
        history.write(f"{epoch},{train_acc:.6f},{val_acc:.6f}\n")
raport = create_raport(
    model_name=f"model_{model_version_number}",
    network=network,
    learning_rate=model_lr,
    optimizer_name=optimizer_name,
    seed=args.seed,
    training_time=training_time,
    live_plot=not args.no_plot,
    train_epochs=accuracy_plot.train_epochs,
    train_accuracy=accuracy_plot.train_accuracy,
    validation_epochs=accuracy_plot.validation_epochs,
    validation_accuracy=accuracy_plot.validation_accuracy,
    best_epoch=best_epoch,
    test_predictions=test_predictions,
    test_labels=testlabels,
)
print(raport)
save_raport(raport, f"{model_path}/model_{model_version_number}_raport.txt")
accuracy_plot.show()

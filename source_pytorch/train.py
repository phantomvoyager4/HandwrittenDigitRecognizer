import torch 
import torch.nn
import torch.optim
from dataclasses import dataclass, asdict
import copy
from pathlib import Path

from model import NN_Model, ModelArgs
from device import device
from data import data_loading

device_selection = device()

# Directory logic
MODELS_DIRECTORY_PATH = Path(__file__).resolve().parent.parent / "models_data_storage"
model_version_number = 1
while True:
    MODEL_SAVING_PATH = MODELS_DIRECTORY_PATH / f"model_pytorch_{model_version_number}.pt"
    if Path.exists(MODEL_SAVING_PATH):
        model_version_number += 1
    else:
        break

@dataclass
class Hyperparams:
    """
    Hyperparameters adjusted for MLP training.
    """
    epochs: int = 30
    learning_rate:float = 0.5
    batch_size: int = 64
    validation_size: int = 10000
    seed: int = 42

def accuracy(logits_grid: torch.Tensor, labels: torch.Tensor) -> float:
    predicted_digits_idx = torch.argmax(logits_grid, dim=1)
    predicted_digits_validated = predicted_digits_idx == labels
    predicted_digits_validated = predicted_digits_validated.float()
    mean_predicted = torch.mean(predicted_digits_validated)
    return mean_predicted.item()

hp = Hyperparams()
torch.manual_seed(hp.seed)

train_images, train_labels, validation_images, validation_labels, test_images, test_labels = data_loading(hp.validation_size, hp.seed)

args = ModelArgs()
model = NN_Model(args)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters() ,lr=hp.learning_rate)

# Evaluation measures
train_loss_values = []
batch_loss_values = []
validation_loss_values = []
epoch_count = []
mean_predicted_values = []
best_accuracy = -1
best_epoch = None
best_weights = None

# Training loop
for epoch in range(hp.epochs):
    model.train()
    order = torch.randperm(len(train_images))

    batches_train_loss = 0
    iterat = 0 
    for start in range(0, len(train_labels), hp.batch_size):
        
        batch_idx = order[start:start + hp.batch_size]

        if len(batch_idx) < hp.batch_size:
            continue

        predictions = model(train_images[batch_idx])

        batch_loss = loss_fn(predictions, train_labels[batch_idx])
        batches_train_loss += batch_loss.item()
        iterat += 1

        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()

    batches_train_loss_avg = batches_train_loss / iterat
    model.eval()
    with torch.inference_mode():
        
        validation_predictions = model(validation_images)
        validation_loss = loss_fn(validation_predictions, validation_labels)

        mean_predicted = accuracy(validation_predictions, validation_labels)
        if mean_predicted > best_accuracy:
            best_accuracy = mean_predicted
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

        train_predictions = model(train_images)
        train_loss = loss_fn(train_predictions, train_labels)
    
    if epoch % 10 == 0 or epoch == hp.epochs - 1:
        epoch_count.append(epoch)
        batch_loss_values.append(batches_train_loss_avg)
        train_loss_values.append(train_loss.item())
        validation_loss_values.append(validation_loss.item())
        mean_predicted_values.append(mean_predicted)
        print(f"Epoch: {epoch}, Fraction of correctly predictied digits: {mean_predicted_values[-1]:.3f} Batch avg loss: {batch_loss_values[-1]:.3f}, Train epoch loss: {train_loss_values[-1]:.3f}, Validation loss: {validation_loss_values[-1]:.3f}")

model.load_state_dict(best_weights)
model.eval()
with torch.inference_mode(): 
    predictions_test = model(test_images)
    test_accuracy = accuracy(predictions_test, test_labels)

results = {"Best_Epoch": best_epoch,
           "Best_Accuracy": best_accuracy,
           "Test_Accuracy": test_accuracy}
model_save = {"weights": model.state_dict(),
                "model_config": asdict(args),
                "training_config": asdict(hp),
                "results": results}
torch.save(model_save, MODEL_SAVING_PATH)

print(f"Best epoch: {best_epoch}, Best accuracy {best_accuracy:.3f}, Test accuracy: {test_accuracy:.3f} \n save path: {MODEL_SAVING_PATH}")

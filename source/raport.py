import numpy as np
from model import Layer

DROP_THRESHOLD = 0.05 #smallest fall in training accuracy counted as a drop
CONVERGENCE_TARGETS = (0.90, 0.95)


def create_raport(model_name, network, learning_rate, optimizer_name, seed, training_time, live_plot,
                  train_epochs, train_accuracy, validation_epochs, validation_accuracy, best_epoch,
                  test_predictions, test_labels):
    sections = [
        [f"Training report: {model_name}"],
        config_section(network, learning_rate, optimizer_name, seed, len(train_epochs), training_time, live_plot),
        accuracy_section(train_epochs, train_accuracy, validation_epochs, validation_accuracy,
                         best_epoch, test_predictions, test_labels),
        convergence_section(train_epochs, train_accuracy),
        stability_section(train_epochs, train_accuracy),
        per_digit_section(test_predictions, test_labels),
    ]
    return "\n\n".join("\n".join(section) for section in sections)


def save_raport(raport, path):
    with open(path, 'w') as file:
        file.write(raport + "\n")


def config_section(network, learning_rate, optimizer_name, seed, epochs, training_time, live_plot):
    architecture = str(network[0].weights.shape[0])
    for layer in network:
        if isinstance(layer, Layer):
            architecture += f" -> {layer.weights.shape[1]}"
        else:
            architecture += f" ({type(layer).__name__})"
    architecture += " (Softmax)"
    minutes, seconds = divmod(round(training_time), 60)
    plot_note = ", including live plot updates" if live_plot else ""
    return [
        "Configuration",
        f"  Architecture:   {architecture}",
        f"  Optimizer:      {optimizer_name}",
        f"  Learning rate:  {learning_rate}",
        f"  Epochs:         {epochs}",
        f"  Seed:           {seed}",
        f"  Training time:  {minutes}m {seconds}s ({training_time / epochs:.2f} s per epoch{plot_note})",
    ]


def accuracy_section(train_epochs, train_accuracy, validation_epochs, validation_accuracy,
                     best_epoch, test_predictions, test_labels):
    #the saved model comes from the epoch with the best validation accuracy;
    #the test set was not looked at until that choice was made
    best_train = train_accuracy[train_epochs.index(best_epoch)]
    best_validation = validation_accuracy[validation_epochs.index(best_epoch)]
    test = float(np.mean(test_predictions == test_labels))

    #bias = error on the training set, variance = how much worse the model does on unseen data
    bias = 1 - best_train
    variance = (1 - best_validation) - (1 - best_train)

    return [
        f"Selected model (epoch {best_epoch}, best validation accuracy of {len(validation_epochs)} epochs)",
        f"  Training set:    {best_train:.2%}",
        f"  Validation set:  {best_validation:.2%}",
        f"  Test set:        {test:.2%}  (measured once, on the selected model)",
        "",
        "Bias and variance",
        f"  Bias (training error):                         {bias:.2%}",
        f"  Variance (validation error - training error):  {variance:.2%}",
        "",
        "Last epoch",
        f"  Training set:    {train_accuracy[-1]:.2%}",
        f"  Validation set:  {validation_accuracy[-1]:.2%}",
    ]


def convergence_section(train_epochs, train_accuracy):
    lines = ["Convergence (training set)"]
    for target in CONVERGENCE_TARGETS:
        reached = next((epoch for epoch, accuracy in zip(train_epochs, train_accuracy) if accuracy >= target), None)
        result = f"epoch {reached}" if reached is not None else "not reached"
        lines.append(f"  First reached {target:.0%}:  {result}")
    return lines


def find_drops(train_accuracy):
    #a drop is a continuous fall in accuracy, returned as (peak index, lowest point index)
    drops = []
    start = 0
    for index in range(1, len(train_accuracy) + 1):
        if index == len(train_accuracy) or train_accuracy[index] >= train_accuracy[index - 1]:
            if train_accuracy[start] - train_accuracy[index - 1] >= DROP_THRESHOLD:
                drops.append((start, index - 1))
            start = index
    return drops


def stability_section(train_epochs, train_accuracy):
    lines = [f"Training stability (falls in training accuracy of {DROP_THRESHOLD * 100:.0f} percentage points or more)"]
    drops = find_drops(train_accuracy)
    if not drops:
        lines.append("  No drops - training accuracy rose steadily")
        return lines
    drops.sort(key=lambda drop: train_accuracy[drop[0]] - train_accuracy[drop[1]], reverse=True)
    lines.append(f"  Number of drops:  {len(drops)}")
    lines.append("  Largest drops:")
    for peak, lowest in drops[:3]:
        lines.append(f"    epoch {train_epochs[peak]} -> {train_epochs[lowest]}:  "
                     f"{train_accuracy[peak]:.2%} -> {train_accuracy[lowest]:.2%}")
    return lines


def per_digit_section(predictions, labels):
    lines = [
        "Per-digit results (test set)",
        "  Digit  Samples  Accuracy  Precision        F1",
    ]
    recalls, precisions, f1_scores = [], [], []
    for digit in range(10):
        samples = np.sum(labels == digit)
        predicted_as_digit = np.sum(predictions == digit)
        correct = np.sum((predictions == digit) & (labels == digit))
        recall = correct / samples if samples else 0.0
        precision = correct / predicted_as_digit if predicted_as_digit else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1_scores.append(f1)
        lines.append(f"  {digit:>5}  {samples:>7}  {recall:>8.2%}  {precision:>9.2%}  {f1:>8.2%}")
    lines.append(f"  Macro average   {np.mean(recalls):>8.2%}  {np.mean(precisions):>9.2%}  {np.mean(f1_scores):>8.2%}")
    weakest = int(np.argmin(recalls))
    lines += [
        "",
        f"  Weakest digit: {weakest} ({recalls[weakest]:.2%} accuracy)",
        "  Accuracy = recall: share of a digit's images the model classified correctly",
        "  Precision: share of predictions of a digit that were correct",
        "  F1: balance of precision and recall",
    ]
    return lines

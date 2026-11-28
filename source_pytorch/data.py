from pathlib import Path

import torch
import numpy as np

#the dataset folder sits in the repo root, one level above this file, so paths work from any working directory
DATASET_DIR = Path(__file__).resolve().parent.parent / 'dataset'

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
    images = torch.from_numpy(images)
    images = images.to(torch.float32)
    labels = torch.from_numpy(labels)
    return images, labels

def train_validation_split(images, labels, validation_size, seed):
    #the validation set is cut from the training set, so the test set is used only once, at the very end
    order = np.random.default_rng(seed).permutation(len(labels))
    validation, train = order[:validation_size], order[validation_size:]
    return images[train], labels[train], images[validation], labels[validation]

def data_loading(validation_size, seed):
    train_images_path = DATASET_DIR / 'train-images.idx3-ubyte'
    train_labels_path = DATASET_DIR / 'train-labels.idx1-ubyte'
    test_images_path = DATASET_DIR / 't10k-images.idx3-ubyte'
    test_labels_path = DATASET_DIR / 't10k-labels.idx1-ubyte'

    train_images, train_labels = data_handling(train_images_path, train_labels_path)
    train_images, train_labels, validation_images, validation_labels = train_validation_split(train_images, train_labels, validation_size, seed)

    test_images, test_labels = data_handling(test_images_path, test_labels_path)
    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels
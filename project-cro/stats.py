# -*- coding: utf-8 -*-  

from __future__ import print_function
from keras.utils import to_categorical, plot_model
from keras.models import model_from_json
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from cro_mapper import map_int_to_letter
import pydot
import graphviz
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "./datasets/"
MODEL_PATH = "./model.json"
WEIGHTS_PATH = "./model.h5"

def load_dataset(x_path, y_path, num_classes):
    x, y = np.load(x_path), np.load(y_path)
    x = x.astype('float32')
    x /= 255.0
    x = x.reshape(*x.shape, 1)
    y = to_categorical(y, num_classes)
    return x, y

def load_model(model_path, weights_path):
    model_json_file = open(model_path, 'r')
    loaded_model_json = model_json_file.read()
    model_json_file.close()

    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    return model

def categorical_to_class(input):
    return np.argmax(input, axis=1)

def print_global_stats(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Macro precision:", precision_score(y_true, y_pred, average='macro'))
    print("Macro recall:", recall_score(y_true, y_pred, average='macro'))
    print("Macro f1:", f1_score(y_true, y_pred, average='macro'))

def plot_wrong_classification(x, y_true, y_pred, filename):
    samples = []
    fig_size = (10, 10)
    for i in range(0, x.shape[0]):
        if y_true[i] != y_pred[i]:
            samples.append((y_true[i], y_pred[i], i))

    def plot_sample(x, axis):
        img = x.reshape(x.shape[0], x.shape[1])
        axis.imshow(img, cmap='gray')

    fig = plt.figure(figsize=fig_size)

    for i in range(len(samples)):
        y_t, y_p, index = samples[i]
        ax = fig.add_subplot(*fig_size, i + 1, xticks=[], yticks=[])
        title = map_int_to_letter(y_t) + " -> " + map_int_to_letter(y_p) 
        ax.title.set_text(title)
        ax.title.set_fontsize(10)
        plot_sample(x[index], ax)

    fig.tight_layout()
    plt.savefig(filename)

def save_confusion_matrix_csv(confusion_matrix, filename):
    num_classes = confusion_matrix.shape[0]
    header = [map_int_to_letter(letter_int) for letter_int in range(num_classes)]
    header = ",".join(header)
    np.savetxt(filename, confusion_matrix.astype(int), fmt='%i', delimiter=",")

def main():
    model = load_model(MODEL_PATH, WEIGHTS_PATH)
    num_classes = model.layers[-1].output_shape[1]
    x, y_true_oh = load_dataset(DATASET_PATH + "x_train.dat", 
                             DATASET_PATH + "y_test.dat", 
                             num_classes)
    print(x.shape)
    x, y_true_oh = load_dataset(DATASET_PATH + "x_validation.dat", 
                             DATASET_PATH + "y_test.dat", 
                             num_classes)
    print(x.shape)
    x, y_true_oh = load_dataset(DATASET_PATH + "x_test.dat", 
                             DATASET_PATH + "y_test.dat", 
                             num_classes)
    print(x.shape)
    y_pred_oh = model.predict(x)

    y_pred = categorical_to_class(y_pred_oh)
    y_true = categorical_to_class(y_true_oh)
    print_global_stats(y_true, y_pred)
    plot_wrong_classification(x, y_true, y_pred, "wrong_classification.pdf")
    test_confusion = confusion_matrix(y_true, y_pred)
    save_confusion_matrix_csv(test_confusion, "confusion_matrix.csv")

if __name__ == "__main__":
    main()
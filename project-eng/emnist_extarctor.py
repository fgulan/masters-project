#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function
from os import path
import os
from scipy.io import loadmat
from scipy.misc import imsave
import numpy as np
import pickle
import sys
from sklearn.model_selection import train_test_split

## This part is borrowed from https://github.com/Coopss/EMNIST
def load_data(mat_file_path, width=28, height=28, max_=None, verbose=True):
    ''' Load data in from .mat file as specified by the paper.
        Arguments:
            mat_file_path: path to the .mat, should be in sample/
        Optional Arguments:
            width: specified width
            height: specified height
            max_: the max number of samples to load
            verbose: enable verbose printing
        Returns:
            A tuple of training and test data, and the mapping for class code to ascii value,
            in the following format:
                - ((training_images, training_labels), (testing_images, testing_labels), mapping)
    '''
    # Local functions
    def rotate(img):
        # Used to rotate images (for some reason they are transposed on read-in)
        flipped = np.fliplr(img)
        return np.rot90(flipped)

    def display(img, threshold=0.5):
        # Debugging only
        render = ''
        for row in img:
            for col in row:
                if col > threshold:
                    render += '@'
                else:
                    render += '.'
            render += '\n'
        return render

    # Load convoluted list structure form loadmat
    mat = loadmat(mat_file_path)

    # Load char mapping
    mapping = {kv[0]:kv[1:][0] for kv in mat['dataset'][0][0][2]}
    pickle.dump(mapping, open('mapping.pickle', 'wb' ))
    print(mapping)
    # Load training data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][0][0][0][0])
    training_images = mat['dataset'][0][0][0][0][0][0][:max_].reshape(max_, height, width, 1)
    training_labels = mat['dataset'][0][0][0][0][0][1][:max_]

    # Load testing data
    if max_ == None:
        max_ = len(mat['dataset'][0][0][1][0][0][0])
    else:
        max_ = int(max_ / 6)
    testing_images = mat['dataset'][0][0][1][0][0][0][:max_].reshape(max_, height, width, 1)
    testing_labels = mat['dataset'][0][0][1][0][0][1][:max_]

    # Reshape training data to be valid
    if verbose == True: _len = len(training_images)
    for i in range(len(training_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        training_images[i] = rotate(training_images[i])
    if verbose == True: print('')

    # Reshape testing data to be valid
    if verbose == True: _len = len(testing_images)
    for i in range(len(testing_images)):
        if verbose == True: print('%d/%d (%.2lf%%)' % (i + 1, _len, ((i + 1)/_len) * 100), end='\r')
        testing_images[i] = rotate(testing_images[i])
    if verbose == True: print('')

    # Convert type to float32
    training_images = training_images.astype('float32')
    testing_images = testing_images.astype('float32')

    # Normalize to prevent issues with model
    training_images /= 255
    testing_images /= 255
    
    train_shape = training_images.shape
    test_shape = testing_images.shape

    training_images = training_images.reshape(train_shape[0], train_shape[1], train_shape[2])
    testing_images = testing_images.reshape(test_shape[0], test_shape[1], test_shape[2])
    training_labels = training_labels.reshape(train_shape[0])
    testing_labels = testing_labels.reshape(test_shape[0])

    nb_classes = len(mapping)

    return ((training_images, training_labels), (testing_images, testing_labels), mapping, nb_classes)

def load_and_dump_dataset(dataset_path):
    ((X, y), (x_test, y_test), mapping, nb_classes) = load_data("emnist-letters.mat")
    print("Input shape:", X.shape, "Output shape:", y.shape)
    x_train, x_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)
    import pdb; pdb.set_trace()
    x_train.dump("datasets/x_train.dat")
    x_test.dump("datasets/x_test.dat")
    x_validation.dump("datasets/x_validation.dat")
    y_train.dump("datasets/y_train.dat")
    y_test.dump("datasets/y_test.dat")
    y_validation.dump("datasets/y_validation.dat")

    print("Sets saved...")
    print('Train samples:', x_train.shape[0])
    print('Validation samples:', x_validation.shape[0])
    print('Test samples:', x_test.shape[0])

def main():
    dataset_path = sys.argv[1]
    load_and_dump_dataset(dataset_path)

if __name__ == "__main__":
    main()
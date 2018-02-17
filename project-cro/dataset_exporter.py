import sys
import numpy as np
from sklearn.model_selection import train_test_split
from clcd import load_dataset

def load_and_dump_dataset(dataset_path):
    X, y = load_dataset(dataset_path)
    print("Input shape:", X.shape, "Output shape:", y.shape)
    x_train, x_validation_test, y_train, y_validation_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(x_validation_test, y_validation_test, test_size=0.35, random_state=42)

    x_train.dump("x_train.dat")
    x_test.dump("x_test.dat")
    x_validation.dump("x_validation.dat")
    y_train.dump("y_train.dat")
    y_test.dump("y_test.dat")
    y_validation.dump("y_validation.dat")

    print("Sets saved...")
    print('Train samples:', x_train.shape[0])
    print('Validation samples:', x_validation.shape[0])
    print('Test samples:', x_test.shape[0])

def main():
    dataset_path = sys.argv[1]
    load_and_dump_dataset(dataset_path)

if __name__ == "__main__":
    main()
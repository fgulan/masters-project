from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import clcd
import numpy as np
import h5py
from keras.models import model_from_json
import os
from scipy import misc
import cro_mapper

LOWER_PATH = "/Users/filipgulan/Diplomski/Keras/data/lower/"
UPPER_PATH = "/Users/filipgulan/Diplomski/Keras/data/upper/"

img_rows, img_cols = 30, 30

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights-improvement-645-0.94.hdf5")
print("Loaded model from disk")

image = misc.imread("/Users/filipgulan/Desktop/test.png", mode='F')
print("Image loaded...")
X = []
X.append(image)
X = np.asarray(X)
X = X.reshape(X.shape[0], img_rows, img_cols, 1)
X = X.astype('float32')
X /= 255

prediction = loaded_model.predict(X)[0]

print("Predictions:")
prediction_dict = dict(enumerate(prediction))
prediction_sorted = sorted(prediction_dict.items(), key=lambda x:x[1], reverse=True)

counter = 0
for key, value in prediction_sorted:
    print(cro_mapper.map_int_to_letter(key), value)
    counter += 1
    if counter == 5:
        break
    
index = np.argmax(prediction)
print("Recognized letter:", cro_mapper.map_int_to_letter(index))
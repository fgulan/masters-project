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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
# from matplotlib import pyplot as plt
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 32

# input image dimensions
img_rows, img_cols = 30, 30

x_test = np.load("x_test.dat")
x_validation = np.load("x_validation.dat")
y_test = np.load("y_test.dat")
y_validation = np.load("y_validation.dat")

x_test = x_test.astype('float32')
x_validation = x_validation.astype('float32')

x_validation /= 255
x_test /= 255
print(x_validation.shape[0], 'validation samples')
print(x_test.shape[0], 'test samples')
y_test = keras.utils.to_categorical(y_test, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)

json_file = open('5000_two_hidden/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("10_000_twohidden/weights-improvement-6852-0.95267.hdf5")
print("Loaded model from disk")

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_validation = x_validation.reshape(x_validation.shape[0], img_rows, img_cols, 1)

output = model.predict(x_test)

y_true = []
y_pred = []
for i in range(0, x_test.shape[0]):
    y_pred.append(np.argmax(output[i]))
    y_true.append(np.argmax(y_test[i]))

y_pred = np.asarray(y_pred)
y_true = np.asarray(y_true)
print("Micro precision:", precision_score(y_true, y_pred, average='micro'))
print("Macro precision:", precision_score(y_true, y_pred, average='macro'))
print("Accuracy:", accuracy_score(y_true, y_pred))

print("Micro recall:", recall_score(y_true, y_pred, average='micro'))
print("Macro recall:", recall_score(y_true, y_pred, average='macro'))

# exit()

# samples = []
# for i in range(0, x_test.shape[0]):
#     if y_true[i] != y_pred[i]:
#         samples.append((y_true[i], y_pred[i], i))

# def plot_sample(x, axis):
#     img = x.reshape(30,30)
#     axis.imshow(img, cmap='gray')

# fig = plt.figure(figsize=(10, 6))

# print()

# for i in range(len(samples)):
#     y_t, y_p, index = samples[i]
#     ax = fig.add_subplot(10, 6, i + 1, xticks=[], yticks=[])
#     title = cro_mapper.map_int_to_letter(y_t) + " -> " + cro_mapper.map_int_to_letter(y_p) 
#     ax.title.set_text(title)
#     ax.title.set_fontsize(10)
#     plot_sample(x_test[index], ax)

# fig.tight_layout()
# # fig.subplots_adjust(top=0.88)
# plt.show()

def accuracy_c(confusion, averaging="macro"):
    return confusion.trace()/confusion.sum()

def precision_c(confusion, averaging="macro"):
    size = confusion.shape[0]
    tw_ms = [twoway_confusion_matrix(confusion, i) for i in range(size)]
    if averaging == "macro":        
        precision=0.0
        for tw in tw_ms:
            if tw[0, 0] != 0:  
                precision += tw[0, 0]/(tw[0, 0] + tw[0, 1])
        return precision/size
    else:
        conf = np.zeros((2,2)).astype(float)
        for tw in tw_ms:
            conf += tw
        conf /= size
        return conf[0, 0]/(conf[0, 0] + conf[0, 1])

def recall_c(confusion, averaging="macro"):
    size = confusion.shape[0]
    tw_ms = [twoway_confusion_matrix(confusion, i) for i in range(size)]
    if averaging == "macro":        
        precision=0.0
        for tw in tw_ms:
            if tw[0, 0] != 0:  
                precision += tw[0, 0]/(tw[0, 0] + tw[1, 0])
        return precision/size
    else:
        conf = np.zeros((2,2)).astype(float)
        for tw in tw_ms:
            conf += tw
        conf /= size
        return conf[0, 0]/(conf[0, 0] + conf[1, 0])

def twoway_confusion_matrix(cm, i):
    tp = cm[i, i]
    fn = np.sum(cm[i,:]) - tp
    fp = np.sum(cm[:,i]) - tp
    tn = np.sum(cm) - fp - fn - tp
    return np.matrix([[tp, fp], [fn, tn]]).astype(float)

test_confusion = confusion_matrix(y_true, y_pred)
for i in range(test_confusion.shape[0]):
    print("Matrica zabune za klasu", cro_mapper.map_int_to_letter(i))
    tw = twoway_confusion_matrix(test_confusion, i)
    print("Preciznost", precision_c(tw, averaging="macro"))
    print("Odziv", recall_c(tw, averaging="macro"))
    print("Tocnost", accuracy_c(tw, averaging="macro"))
    print(tw)
np.set_printoptions(threshold=np.inf)

# with open("confmatrix_10000.txt", 'w') as f:
#     f.write(np.array2string(confusion_matrix(y_true, y_pred), separator=', '))
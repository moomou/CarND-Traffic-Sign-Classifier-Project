# Load pickled data
import pickle

import numpy as np
import sklearn
from sklearn import preprocessing as skproc

from data import load_pickles

datas = load_pickles()
X_train, y_train = datas['X_train'], datas['y_train']
X_valid, y_valid = datas['X_valid'], datas['y_valid']
X_test, y_test = datas['X_test'], datas['y_test']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:4]

# TODO: How many unique classes/labels there are in the dataset.
le = skproc.LabelEncoder()
le.fit(y_train)
n_classes = len(le.classes_)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    # this normalizes the data
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

datagen.fit(X_train.astype('float32'))

# model def
import os
import keras
from model import get_model

m = get_model(image_shape, n_classes)

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

# fits the model on batches with real-time data augmentation:
epochs = 240

monitor = 'val_acc'
callbacks = [
    keras.callbacks.ModelCheckpoint(
        os.path.join('./ckpts', 'chkpt.{epoch:05d}.{%s:.5f}.hdf5' % monitor),
        mode='min',
        monitor=monitor,
        save_weights_only=True, )
]
m.fit_generator(
    datagen.flow(X_train, y_train, batch_size=32),
    samples_per_epoch=len(X_train),
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    nb_epoch=epochs)

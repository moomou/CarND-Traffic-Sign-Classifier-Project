# Load pickled data
import pickle
import math

import numpy as np
import sklearn
from sklearn import preprocessing as skproc

from data import load_pickles

datas = load_pickles()
X_train, y_train = datas['X_train'], datas['y_train']
X_valid, y_valid = datas['X_valid'], datas['y_valid']
X_test, y_test = datas['X_test'], datas['y_test']

n_classes = datas['n_classes']
image_shape = X_train.shape[1:4]

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    shear_range=math.pi,
    rotation_range=25,
    width_shift_range=0.5,
    height_shift_range=0.5,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True)

datagen.fit(X_train.astype('float32'))

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
epochs = 360

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
    validation_data=(X_test, y_test),
    steps_per_epoch=len(X_train) // 32,
    callbacks=callbacks,
    epochs=epochs)

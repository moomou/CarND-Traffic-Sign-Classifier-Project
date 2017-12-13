# Load pickled data
import pickle
import math

import numpy as np
import sklearn
from sklearn import preprocessing as skproc

from data import (
    load_pickles,
    generator,
    pregen, )

BATCH = 128

datas = load_pickles()
X_train, y_train = datas['X_train'], datas['y_train']
X_valid, y_valid = datas['X_valid'], datas['y_valid']

train_gen = generator(X_train, y_train, batch_size=BATCH)
valid_gen = generator(X_valid, y_valid, batch_size=BATCH)

n_classes = datas['n_classes']
image_shape = X_train.shape[1:4]

import os
import keras
from model import get_model

m = get_model(image_shape, n_classes)

epochs = 360

monitor = 'val_acc'
callbacks = [
    keras.callbacks.TensorBoard(
        './log_dir', histogram_freq=2, write_grads=True),
    keras.callbacks.ModelCheckpoint(
        os.path.join('./ckpts', 'chkpt.{epoch:05d}.{%s:.5f}.hdf5' % monitor),
        mode='min',
        monitor=monitor,
        save_weights_only=True, )
]

validation_data = pregen(
    valid_gen, steps=len(X_valid) // BATCH, batch_size=BATCH)

m.fit_generator(
    train_gen,
    steps_per_epoch=100,
    validation_data=validation_data,
    callbacks=callbacks,
    epochs=epochs)

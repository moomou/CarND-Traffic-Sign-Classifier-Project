# model def
from keras.models import Sequential
from keras.layers import *
from keras import optimizers


def get_model(image_shape=(32, 32, 3), n_classes=43, load_weight=False):
    fsize_pow = 3
    ksize = 3
    blk = 5

    # build a classifier model to put on top of the convolutional model
    m = Sequential()

    for i in range(blk):
        fsize = 2**(fsize_pow + i)
        if i == 0:
            m.add(Conv2D(fsize, ksize, input_shape=image_shape))
        else:
            m.add(Conv2D(fsize, ksize))

        # m.add(BatchNormalization())
        m.add(Activation('relu'))
        if i % 2 == 0 and i:
            m.add(AveragePooling2D(2, strides=2))

    m.add(Flatten())
    m.add(Dropout(0.5))
    m.add(Dense(2**8, activation='relu'))
    m.add(Dense(n_classes, activation='softmax'))

    m.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizers.adam(),
        metrics=['accuracy'])

    m.summary()

    if load_weight:
        m.load_weight('weights.h5')

    return m

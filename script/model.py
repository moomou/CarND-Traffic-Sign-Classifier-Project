# model def
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers


def get_model(image_shape=(32, 32, 3), n_classes=43):
    fsize_pow = 5
    ksize = 3
    blk = 3

    # build a classifier model to put on top of the convolutional model
    x_out = x = Input(shape=image_shape, name='x')
    x_out = Conv2D(3, 1)(x_out)

    for i in range(blk):
        fsize = 2**(fsize_pow + i)
        x_out = Conv2D(fsize, ksize, padding='same', activation='relu')(x_out)
        x_out = Conv2D(fsize, ksize, padding='same', activation='relu')(x_out)
        x_out = MaxPooling2D(2)(x_out)

    x_out = Flatten()(x_out)
    x_out = Activation('relu')(x_out)
    x_out = Dropout(0.5)(x_out)
    x_out = Dense(2**10, activation='relu')(x_out)
    x_out = Dropout(0.5)(x_out)
    x_out = Dense(n_classes, activation='softmax')(x_out)

    m = Model(inputs=x, outputs=x_out)
    m.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizers.adam(),
        metrics=['accuracy'], )

    m.summary()
    return m

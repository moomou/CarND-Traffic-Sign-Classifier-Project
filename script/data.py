import pickle
import random

import numpy as np
from sklearn.utils import shuffle as sk_shuffle
from skimage import exposure
from sklearn import preprocessing as skproc
from scipy.ndimage import interpolation
from tqdm import trange


def rgb2gray(rgb):
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return gray.reshape(rgb.shape[:-1] + (1, ))


def transform_images(data, new_coords):
    data = rgb2gray(data)
    new_images = np.zeros(data.shape)

    for i in range(new_images.shape[0]):
        coords = new_coords[i]
        slice_x = slice(coords[0], coords[2])
        slice_y = slice(coords[1], coords[3])

        new_images[i, slice_x, slice_y, :] = data[i][slice_x, slice_y, :]
        new_images[i] = (new_images[i] - 128) / 128
        new_images[i] = exposure.equalize_hist(new_images[i])

    return new_images


def resize_coords(dataset):
    size_change = 32. / dataset['sizes']
    new_coords = dataset['coords'] * np.tile(size_change, 2)
    return new_coords.astype('uint8')


def load_pickles():
    training_file = './data/train.p'
    validation_file = './data/valid.p'
    testing_file = './data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    X_train = transform_images(X_train, resize_coords(train))
    X_valid = transform_images(X_valid, resize_coords(valid))
    X_test = transform_images(X_test, resize_coords(test))

    n_train = X_train.shape[0]
    n_validation = X_valid.shape[0]
    n_test = X_test.shape[0]
    image_shape = X_train.shape[1:4]

    le = skproc.LabelEncoder()
    le.fit(y_train)
    n_classes = len(le.classes_)

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_validation)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        n_classes=n_classes, )


def augment(X):
    augmented = []

    for i in range(X.shape[0]):
        im = np.squeeze(X[i])
        im = interpolation.rotate(im, random.randint(-10, 10), reshape=False)

        augmented.append(im.reshape((32, 32, 1)))

    return np.array(augmented)


def generator(X, y, batch_size=32, aug=True, shuffle=True):
    while True:
        if shuffle:
            X, y = sk_shuffle(X, y)

        steps = X.shape[0] // batch_size

        start = 0
        for i in range(steps):
            batch_X = np.copy(X[start:start + batch_size])
            batch_y = np.copy(y[start:start + batch_size])

            start += batch_size

            if aug:
                batch_X = augment(batch_X)

            yield batch_X, batch_y


def pregen(gen, steps, batch_size=32):
    X = np.zeros((batch_size * steps, 32, 32, 1))
    y = np.zeros((batch_size * steps, 1))

    start = 0
    for i in trange(steps):
        _X, _y = next(gen)
        X[start:start + batch_size] = _X
        y[start:start + batch_size] = _y.reshape((-1, 1))
        start += batch_size

    return X, y

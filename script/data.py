import os

EXTRA_DATA_ROOT = '../extra_data'


def load_extra_files():
    pass


def get_top_k(pred, k=5, captions=None):
    pass


def load_pickles():
    # Load pickled data
    import pickle
    import numpy as np
    import sklearn
    from sklearn import preprocessing as skproc
    # TODO: Fill this in based on where you saved the training and testing data

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

    return dict(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test, )

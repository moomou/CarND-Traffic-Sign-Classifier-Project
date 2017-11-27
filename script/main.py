import pickle

import numpy as np
import sklearn
from sklearn import preprocessing as skproc

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

le = skproc.LabelEncoder()
le.fit(y_train)

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.
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
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)

# model def
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import optimizers

model = InceptionV3(include_top=False, weights='imagenet')
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(le.classes_, activation='softmax'))

top_model.load_weights('...')

for layer in model.layers:
    layer.trainable = False

# add the model on top of the convolutional base
model.add(top_model)

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    metrics=['accuracy'])

epochs = 20
model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) / 32,
    epochs=epochs)

model.predict_on_batch()

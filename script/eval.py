import os

import fire
import numpy as np

import keras.backend as K
from skimage import exposure
from model import get_model
from data import load_pickles, generator, rgb2gray


def _get_m():
    m = get_model()
    m.load_weights(os.environ.pop('WEIGHT'))
    return m


def _extra_img():
    import os
    from scipy import misc
    EXTRA_IMG_DIR = './extra_data'
    os.listdir(EXTRA_IMG_DIR)

    test_imgs = []
    test_y = []
    for img in os.listdir(EXTRA_IMG_DIR):
        test_y.append(int(img[:-4]))
        img = misc.imread(os.path.join(EXTRA_IMG_DIR, img), mode='RGB')
        img = misc.imresize(img, (32, 32, 3))
        img = rgb2gray(np.array([img]))[0]
        img = (img - 128) / 128
        img = exposure.equalize_hist(img)

        test_imgs.append(img)

    X = np.array(test_imgs)
    y = np.array(test_y)
    print('Shape::', X.shape)
    print('Shape::', y.shape)

    return X, y


def val():
    datas = load_pickles()
    X_test, y_test = datas['X_test'], datas['y_test']

    test_gen = generator(
        X_test, y_test, batch_size=30, aug=False, shuffle=False)

    m = _get_m()

    res = m.predict_generator(test_gen, steps=(X_test.shape[0] // 30))
    res = np.argmax(res, axis=1)
    print(np.mean(res == y_test))


def extra():
    X, y = _extra_img()

    m = _get_m()
    res = m.predict_on_batch(X)
    top_k = np.argsort(-1. * res, axis=1)
    ans = np.argmax(res, axis=1)
    print('*' * 10)
    print('Top 5 class::', top_k[:, :5])

    for i in range(5):
        print('Top 5 prob::', res[i, top_k[:, :5][i]])

    print('Overall::', np.mean(ans == y))
    print('Ans::', ans)
    print('Oracle::', y)


def viz(name):
    m = _get_m()

    layer = m.get_layer(name)

    X, y = _extra_img()
    f = K.function([m.get_input(train=False)], [layer.get_output(train=False)])

    return f([X])


if __name__ == '__main__':
    import fire
    fire.Fire()

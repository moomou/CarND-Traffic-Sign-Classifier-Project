import os

import fire
import numpy as np

from model import get_model
from data import load_pickles


def _get_m():
    m = get_model()
    m.load_weights(os.environ.pop('WEIGHT'))
    return m


def val():
    datas = load_pickles()
    X_valid, y_valid = datas['X_valid'], datas['y_valid']

    m = _get_m()

    res = m.predict_on_batch(X_valid)
    res = np.argmax(res, axis=1)
    print(np.mean(res == y_valid))


def extra():
    import os
    from scipy import misc
    EXTRA_IMG_DIR = './extra_data'
    os.listdir(EXTRA_IMG_DIR)

    test_imgs = []
    test_y = []
    for img in os.listdir(EXTRA_IMG_DIR):
        test_y.append(int(img[:-4]))
        img = misc.imread(os.path.join(EXTRA_IMG_DIR, img))
        img = misc.imresize(img, (32, 32, 3))
        test_imgs.append(img[:, :, :3])

    X = np.array(test_imgs)
    y = np.array(test_y)
    print('Shape::', X.shape)
    print('Shape::', y.shape)

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


if __name__ == '__main__':
    import fire
    fire.Fire()

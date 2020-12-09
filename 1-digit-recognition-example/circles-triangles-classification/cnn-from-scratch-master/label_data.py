import numpy as np
import imageio as im
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.utils.np_utils import to_categorical
from matplotlib import pyplot as plt


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def make_labels(directory, data=None, y_hat=None, label=0):
    if y_hat is None:
        y_hat = []
    if data is None:
        data = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            img = im.imread(directory + file)
            grey = rgb2gray(img)
            data.append(grey)
        y_hat = [label] * len(data)
    return np.array(data), np.array(y_hat)


def make_labels_by_file():
    circles, y_circles = [], []
    circles, y_circles = make_labels('../dataset/shapes/circles/', data=circles, y_hat=y_circles, label=0)

    squares, y_squares = [], []
    squares, y_squares = make_labels('../dataset/shapes/squares/', data=squares, y_hat=y_squares, label=1)

    triangles, y_triangles = [], []
    triangles, y_triangles = make_labels('../dataset/shapes/triangles/', data=triangles, y_hat=y_triangles, label=2)

    X, y = np.vstack((circles, squares, triangles)), np.hstack((y_circles, y_squares, y_triangles)).reshape(-1, 1)

    print(X.shape, y.shape)
    return X, y


X, y = make_labels_by_file()

oh = OneHotEncoder()
oh.fit(y)
y_hot = oh.transform(y)
y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_hot, test_size=.2, random_state=42)
x_train, x_test, y_cat_train, y_cat_test = train_test_split(X, y_cat, test_size=.2, random_state=42)

# print(X_test[0][0])

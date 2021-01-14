import tensorflow as tf
import numpy as np
from convGA import Conv3x3
from maxpoolGA import MaxPool2FF
from softmaxGA import Softmax
import random

# The mnist package takes care of handling the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
# We only use the first 1k testing examples (out of 10k total) in the interest of time.
# Feel free to change this if you want.
(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
train_images = train_X[:1000]
train_labels = train_Y[:1000]
test_images = test_X[:1000]
test_labels = test_Y[:1000]

conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
pool = MaxPool2FF()  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)  # 13x13x8 -> 10


def forward(image, label):
    """
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    """
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.

    # num_filters   hard coded as   =  3
    filter_values = []
    for i in range(20):  # population
        # firstly, generate 20 different filters
        filter_values.append(np.random.randn(8, 3, 3) / 9)

    out = []
    loss = 100
    acc = 0

    for generation in range(100):  # generation size = 100
        for j, filter_value in enumerate(filter_values):  # population size
            out = conv.forward((image / 255) - 0.5, filter_value)
            out = pool.forward(out)
            out = softmax.forward(out)
            # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
            new_loss = -np.log(out[label])
            if new_loss < loss:
                loss = new_loss
                acc = 1 if np.argmax(out) == label else 0
                # else:
                # filter_values[j] = np.random.randn(8, 3, 3) / 9

        # mutation
        for k, filter_value in enumerate(filter_values):
            mutation_probability = random.uniform(0, 1)
            # if larger than 0.5 then mutate
            if mutation_probability > 0.5:
                # random number of elements to change
                # because it is 3x3 filter,
                #  8 x (3x3) = 72
                # so, we don't want to change to many element
                number_of_elements = random.randint(1, 20)  # TODO: optimize the param

                # the elements that have been already changed
                has_changed_list = []
                for h in range(number_of_elements):
                    row = random.randint(0, 2)
                    col = random.randint(0, 2)
                    # filter_size = 8 x (3x3),
                    # so randomly change one filter
                    the_number = random.randint(0, 7)
                    key_value_pair = the_number + row + col

                    if key_value_pair not in has_changed_list:
                        element = filter_value[the_number, row, col]
                        # TODO: find a better way of mutating the filter weight
                        filter_value[the_number, row, col] = mutation(element)
                        has_changed_list.append(key_value_pair)

    return out, loss, acc


def mutation(element):
    """
    :param element:
    :return:
    """

    # define a random param between -1.5 to 1.5
    v = 3.0 * random.random()
    epsilon = 1.5 - v
    if v >= 1.5:
        epsilon = v - 3.0

    #     operator are:  +, -, x, ÷
    #               or  0 or 1
    probability_operation = random.uniform(0, 1)
    if probability_operation < 0.2:
        element = element + epsilon
    elif 0.2 <= probability_operation < 0.4:
        element = element - epsilon
    elif 0.4 <= probability_operation < 0.6:
        element = element * epsilon
    elif 0.6 <= probability_operation < 0.8:
        element = element * epsilon
    elif 0.8 <= probability_operation < 0.9:
        element = 0
    else:
        element = 1

    return element


print('MNIST CNN initialized!')

loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
    # Do a forward pass.
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    # Print stats every 100 steps.
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
        )
        loss = 0
        num_correct = 0

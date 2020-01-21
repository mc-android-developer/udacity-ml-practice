#!/usr/bin/python3

import math

import matplotlib.pyplot as mp
import numpy as np
import pandas as pd

import l3s10_perceptron_algorithm.perceptron as pr
import utils.viz_helper as vh

np.random.seed(44)


# Activation (sigmoid) function
def sigmoid(x):
    return 1 / (1 + pow(math.e, (-1 * x)))


# Output (prediction) formula
def output_formula(features, weights, bias):
    c = sum(features * weights) + bias
    s = sigmoid(c)
    return s


def calc_error(labels, pred):
    errors = labels.copy()
    for i in range(len(labels)):
        errors[i] = error_formula(labels[i], pred[i])
    mean_error = np.mean(errors)
    return mean_error


# Error (log-loss) formula
# Calculates error for a single point with provided label and prediction
def error_formula(y, yp):
    if y == yp:
        return 0
    if yp == 0 or yp == 1:
        raise Exception('Prediction must not be 0 or 1')

    yn = (1 - y)
    ypn = (1 - yp)
    return -1 * (y * math.log(yp) + yn * math.log(ypn))


# Gradient descent step
def update_weights(x, y, yp, w, b, lr):
    # yp = output_formula(x, w, b)
    d = lr * (y - yp)
    dx = d * x
    ww = w + dx
    bb = b + d
    return ww, bb


def plot_points(X, y):
    admitted = X[np.argwhere(y == 1)]
    rejected = X[np.argwhere(y == 0)]
    mp.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='blue', edgecolor='k')
    mp.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='red', edgecolor='k')


def display(m, b, color='g--'):
    mp.xlim(-0.05, 1.05)
    mp.ylim(-0.05, 1.05)
    x = np.arange(-10, 10, 0.1)
    mp.plot(x, m * x + b, color)


def train(data, labels, epochs, lrate, graph_lines=False):
    viz = vh.Visualizer()
    viz.add_point_data(data, labels)

    ep_errors = []
    n_records, n_features = data.shape
    last_loss = None
    weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)
    bias = 0

    p = pr.Linear2XPerceptron(weights[0], weights[1], bias)
    p.min_x = -0.1
    p.max_x = 1.1
    viz.add_func(p, vh.Color.GREY2)
    viz.show()

    for e in range(epochs):

        # Printing out the log-loss error on the training set
        sh = labels.shape
        err = np.zeros(sh)
        for i in range(len(labels)):
            pred = output_formula(data[i], weights, bias)
            err[i] = error_formula(labels[i], pred)
        loss = np.mean(err)
        ep_errors.append(loss)

        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e, "==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = pred > 0.5
            accuracy = np.mean(predictions == labels)
            print("Accuracy: ", accuracy)

            # viz.show()

        # if graph_lines and e % (epochs / 100) == 0:
        # display(-weights[0] / weights[1], -bias / weights[1])

        # del_w = np.zeros(weights.shape)
        for d, l in zip(data, labels):
            pred1 = output_formula(d, weights, bias)
            dif1 = l - pred1
            error1 = error_formula(l, pred1)
            # lrate = lrate / ya[0] #?????
            # lrate = lrate * error
            nweights, nbias = update_weights(d, l, pred1, weights, bias, lrate)
            pred2 = output_formula(d, nweights, nbias)
            dif2 = l - pred
            error2 = error_formula(l, pred2)

            weights = nweights
            bias = nbias
            p.config(weights[0], weights[1], bias)

    # Plotting the solution boundary
    mp.title("Solution boundary")
    display(-weights[0] / weights[1], -bias / weights[1], 'black')

    # Plotting the data
    plot_points(data, labels)
    mp.show()

    # Plotting the error
    mp.title("Error Plot")
    mp.xlabel('Number of epochs')
    mp.ylabel('Error')
    mp.plot(ep_errors)
    mp.show()


def main():
    data = pd.read_csv('data.csv', header=None)
    X = np.array(data[[0, 1]])
    y = np.array(data[2])

    # Test data
    # X = np.array([[0.78051, -0.063669], [0.28774, 0.29139], [0.40714, 0.17878], [0.2923, 0.4217]])
    # y = np.array([1, 0, 0, 1])

    epochs = 1000
    learnrate = 0.1
    train(X, y, epochs, learnrate, True)


if __name__ == '__main__':
    main()

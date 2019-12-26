#!/usr/bin/python3

import math

import numpy as np

import utils.data_helper as dh
import utils.viz_helper as vh


def sigmoid_func(x):
    return 1 / (1 + pow(math.e, (-1 * x)))


def softmax_func(x):
    exp = np.exp(x[:, 0])
    sum = exp.sum()
    for i in range(0, len(x)):
        x[i, 1] = exp[i] * 1.0 / sum
    return x


def softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i * 1.0 / sumExpL)
    return result


def main():
    size = 200
    input_data = dh.generate_random_floats(X=size, Y=2, min=0, max=1)

    val = -10.
    step = 0.1
    for i in range(0, size):
        input_data[i][0] = val
        input_data[i][1] = sigmoid_func(val)
        val += step

    print('Sigmoid func result:')
    print(input_data)
    print()

    viz = vh.Visualizer()
    viz.add_point_group(input_data, vh.Color.RED)
    viz.show()

    # res = softmax([5.0, 6.0, 7.0])
    # print(res)
    # res2 = softmax_func(np.array([[5.0, 0.0], [6.0, 0.0], [7.0, 0.0]]))
    # print(res2)

    input_data = softmax_func(input_data)

    print('Softmax func result:')
    print(input_data)
    print()

    viz.add_point_group(input_data, vh.Color.GREEN)
    viz.show()


if __name__ == '__main__':
    main()

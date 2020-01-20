#!/usr/bin/python3

import math

import numpy as np

import utils.viz_helper as vh


class VizSigmoid(vh.VizFunc):
    def f(self, x):
        return sigmoid_func(x)

    def limits(self):
        return -10, 10


# Sigmoid function takes only 1 parameter
def sigmoid_func(x):
    return 1 / (1 + pow(math.e, (-1 * x)))


def main():
    size = 200
    data = np.zeros([size, 2])

    val = -10.
    step = 0.1
    for i in range(0, size):
        data[i][0] = val
        data[i][1] = sigmoid_func(val)
        val += step

    print('Sigmoid func result:')
    print(data)
    print()

    viz = vh.Visualizer()
    svf = VizSigmoid()
    viz.add_func(svf, vh.Color.RED)
    viz.show()


if __name__ == '__main__':
    main()

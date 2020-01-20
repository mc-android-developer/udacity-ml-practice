#!/usr/bin/python3

import numpy as np

import utils.data_helper as dh


# Softmax function takes only many parameters X - is array
def softmax_func(x):
    res = x.copy()
    exp = np.exp(x)
    sum = exp.sum()
    for i in range(0, len(x)):
        res[i] = exp[i] / sum
        res[i] = round(res[i], 2)
    return res


def main():
    size = 10
    inputs = 3
    data = dh.generate_random_floats(X=size, Y=inputs, min=-3, max=3)

    res = softmax_func([5, 6, 7])
    # print(res)

    print('Softmax func result:')
    for i in data:
        res = softmax_func(i)
        print(np.array2string(i) + ' -> ' + np.array2string(res))


if __name__ == '__main__':
    main()

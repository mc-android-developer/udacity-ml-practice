#!/usr/bin/python3

import data_helper as dh
import numpy as np
import viz_helper as vh


class Linear2XPerceptron:
    def __init__(self, w1=1, w2=1, b=0):
        self.config(w1, w2, b)

    def config(self, w1, w2, b):
        self.W = np.array([w1, w2])
        self.C = np.array([w1, w2, b])
        return self

    def config_vec(self, C):
        s = C.shape
        if s[0] != 3 or len(s) > 1:
            raise Exception('Bad shape ' + str(s) + ' Input data size must be 1x3')
        self.config(C[0], C[1], C[2])
        return self

    def calc_once(self, x1, x2):
        return self.C[0] * x1 + self.C[1] * x2 + self.C[2]

    def calc_var(self, x1):
        return np.sum((self.C[0] * x1 + self.C[2]) / (-1 * self.C[1]))

    def calc(self, X):
        s = X.shape
        if s[0] != 2:
            raise Exception('Bad shape ' + str(s) + ' Input data size must be Nx2')

        return np.asscalar(np.dot(self.W, X) + self.C[2])


def adjust_perceptron(prc, input, lrate=0.1):
    s = input.shape
    if s[0] < 3 or len(s) != 1:
        raise Exception('Bad shape ' + str(s) + ' Input data size must be 1x3')

    print('Adjusting perceptron:')

    print('Perceptron config ' + np.array2string(prc.C))
    print('Learning rate ' + str(lrate))

    n = input
    n[2] = 1  # bias is const
    print('Input ' + np.array2string(n))

    m = n * lrate
    print('Correction vector ' + np.array2string(m))

    r = prc.calc(input[0:2])
    if r > 0:
        m *= -1

    t = prc.C + m
    print('Adjusted perceptron config ' + np.array2string(t))
    prc.config_vec(t)

    print()
    return prc


def single_point_tuning():
    # input_data = dh.generate_random_input(X=1, Y=2, min=-5, max=5)
    input_data = np.array([[4, 5]])

    viz = vh.Visualizer()
    viz.add_point_group(input_data, 'red')

    # prc_config = dh.generate_random_input(X=1, Y=3, min=-5, max=5)
    prc_config = np.array([[3, 4, -10]])
    p = Linear2XPerceptron()
    p.config_vec(prc_config[0])  # start with any random config
    viz.add_perceptron(p, 'black')
    viz.show()

    y = p.calc_var(input_data[0])
    while y != input_data[0, 1]:
        i = np.array([input_data[0, 0], input_data[0, 1], 1])
        adjust_perceptron(p, i, 0.01)
        y = p.calc_var(input_data[0])
        viz.show()


def main():
    print('Hello Perceptron!')

    single_point_tuning()
    exit()

    size = 20
    input_data = dh.generate_random_input(X=size, Y=3, min=-5, max=5)
    dh.label_input_data(input_data)

    p = Linear2XPerceptron()
    p.config(1, 2, 3)  # start with any random config

    viz = vh.Visualizer()
    viz.add_point_data(input_data)
    viz.add_perceptron(p, 'brown')
    viz.show()

    convergence_cnt = 0
    while convergence_cnt < 20:
        for i in input_data:
            print('-----------------------------------------------------------------------')
            print('convergence_cnt: ' + str(convergence_cnt))

            adj_cnt = 0
            res = p.calc(i[:-1])
            score = 0 if res > 0 else 1

            if score == i[2]:
                convergence_cnt += 1
                continue

            while score != i[2]:
                convergence_cnt = 0
                print('For input ' + np.array2string(i) + ' perceptron result is ' + str(p.calc(i[:-1])) + ' and score is ' + str(score))

                adjust_perceptron(p, i, 0.1)
                res = p.calc(i[:-1])
                score = 0 if res > 0 else 1

                adj_cnt += 1
                if adj_cnt > 100:
                    raise Exception('Allowed adjustments number treshold exceeded')

            if convergence_cnt == 0:
                break

    print()
    print()
    print('Perceptron automatic adjustment completed')
    print('Result config ' + np.array2string(p.C))

    viz.show()


if __name__ == '__main__':
    main()

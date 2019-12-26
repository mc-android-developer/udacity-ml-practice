#!/usr/bin/python3

import numpy as np

import utils.data_helper as dh
import utils.viz_helper as vh


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
        if s[0] != 2 or len(s) != 1:
            raise Exception('Bad shape ' + str(s) + ' Input data size must be 1x2')

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
    # Parameters from lesson used
    # https://classroom.udacity.com/nanodegrees/nd188-bert/parts/a58738e5-e865-4f64-82e9-cbe7a41b272e/modules/67b445a1-38bc-4128-9d8b-58129e849573/lessons/b4ca7aaa-b346-43b1-ae7d-20d27b2eab65/concepts/8ea20904-0215-4e44-afa9-bb5a720bd028
    input_data = np.array([[4, 5]])

    viz = vh.Visualizer()
    viz.add_point_group(input_data, vh.Color.RED)

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


def n_points_tuning(size, learn_rate=0.1):
    input_data = dh.generate_random_input(X=size, Y=2, min=-5, max=5)
    # input_data = np.array([[-1.04768331, 2.97856097], [-3.95843463, 2.03727394]])
    data_labels = dh.label_input_data(input_data)

    p = Linear2XPerceptron()
    p.config(1, 2, 3)  # start with any random config

    viz = vh.Visualizer()
    viz.add_point_data(input_data, data_labels)
    viz.add_perceptron(p, vh.Color.GREY3)
    viz.show()

    convergence_cnt = 0
    while convergence_cnt < size:
        for i in range(0, len(input_data)):
            print('-----------------------------------------------------------------------')
            print('convergence_cnt: ' + str(convergence_cnt))
            d = input_data[i]
            l = data_labels[i]

            efforts_cnt = 0
            res = p.calc(d)
            score = 1 if res > 0 else 0

            if score == l:
                convergence_cnt += 1
                continue

            while score != l:
                convergence_cnt = 0
                print('For input ' + np.array2string(d) + ' perceptron result is ' + str(res) + ' and score is ' + str(score))

                pd = np.array([d[0], d[1], 1])
                adjust_perceptron(p, pd, learn_rate)
                res = p.calc(d)
                score = 1 if res > 0 else 0

                # viz.show()
                efforts_cnt += 1
                if efforts_cnt > 100:
                    raise Exception('Allowed perceptron adjustments number exceeded')

            if convergence_cnt == 0:
                break

    print()
    print()
    print('Perceptron automatic adjustment completed')
    print('Result config ' + np.array2string(p.C))

    viz.add_perceptron(p, vh.Color.YELLOW)
    viz.show()


def main():
    print('Hello Perceptron!')
    n_points_tuning(100, 0.01)


if __name__ == '__main__':
    main()

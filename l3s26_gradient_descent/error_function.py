#!/usr/bin/python3

import math

import numpy as np

import utils.data_helper as dh
import utils.viz_helper as vh


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


def calc_error(labels, pred):
    errors = labels.copy()
    for i in range(len(labels)):
        errors[i] = error_formula(labels[i], pred[i])
    mean_error = np.mean(errors)
    return mean_error


# Summary:
# Error grows along with the difference between label and prediction
# No difference means 0 error
# High difference results in exponentially high error
#
# If vector of labels/predictions is used:
# Each point in vector contributes to final mean error. But due to exponential grow
# contributions are not the same. Points which has higher difference between label and
# prediction contributes more to the final error.

def main():
    viz = vh.Visualizer()
    step = 40
    points = dh.generate_random_floats(step, 2)

    print('Single label/prediction errors:')
    label = 1
    prediction = 1

    pred_step = prediction / (step + 1)
    prediction -= pred_step
    for i in range(step):
        error = error_formula(label, prediction)
        points[i][0] = label - prediction
        points[i][1] = error
        print(str(label) + '\t' + str(prediction) + '\t\t\t' + str(error))
        prediction -= pred_step

    viz.add_point_group(points, vh.Color.PURPLE)
    viz.set_xylabel('diff [label - predicted prob]', 'error')
    viz.show()

    # Only change a single point in vector to see how much this change contributes to total error
    viz = vh.Visualizer()
    print()
    print('Vector of labels/prediction mean errors:')
    size = 100
    labels = np.ones(size)
    predictions = np.full(labels.shape, 0.9)
    points = np.zeros((size, 2))

    pred_step = predictions[0] / size
    for i in range(size):
        error = calc_error(labels, predictions)
        print(error)
        points[i][0] = i
        points[i][1] = error
        predictions[0] -= pred_step

    viz.add_point_group(points, vh.Color.GREEN)
    viz.set_xylabel('# of reductions of single element', 'error')
    viz.show()

    # Change all points in vector equally to see how much this contributes to total error
    viz = vh.Visualizer()
    print()
    print('Vector of labels/prediction mean errors:')
    size = 100
    labels = np.ones(size)
    predictions = np.full(labels.shape, 0.9)
    points = np.zeros((size, 2))

    pred_step = predictions[0] / size
    for i in range(size):
        error = calc_error(labels, predictions)
        print(error)
        points[i][0] = i
        points[i][1] = error
        for j in range(size):
            predictions[j] -= pred_step

    viz.add_point_group(points, vh.Color.GREEN)
    viz.set_xylabel('# of reductions of all elements', 'error')
    viz.show()


if __name__ == '__main__':
    main()

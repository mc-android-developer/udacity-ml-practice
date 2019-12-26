import math

import numpy as np


# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    exp_sum = 0
    for i in range(0, len(L)):
        exp_sum += pow(math.e, L[i])

    res = list()
    for i in range(0, len(L)):
        res.append(pow(math.e, L[i]) / exp_sum)

    return np.array(res)


################################################
# Udacity solution

def udacity_softmax(L):
    expL = np.exp(L)
    sumExpL = sum(expL)
    result = []
    for i in expL:
        result.append(i * 1.0 / sumExpL)
    return result

    # Note: The function np.divide can also be used here, as follows:
    # def softmax(L):
    #     expL = np.exp(L)
    #     return np.divide (expL, expL.sum())

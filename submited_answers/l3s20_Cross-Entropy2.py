import numpy as np


# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    # np.log = ln
    res = 0
    for i in range(0, len(Y)):
        t1 = Y[i] * np.log(P[i])
        t2 = (1 - Y[i]) * np.log(1 - P[i])
        # print('t1: ' + np.array2string(t1) +'\tt2: ' + np.array2string(t2))
        res += (t1 + t2)
    res *= -1
    return res


################################################
# Udacity solution

def udacity_cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

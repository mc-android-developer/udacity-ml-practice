import sys

import matplotlib.pyplot as mp
import numpy as np


class Visualizer:
    def __init__(self, title='', pointsize=6):
        self.pointsize = pointsize
        self.title = title
        self.point_groups = list()
        self.point_colors = list()
        self.perceptrons = list()
        self.perceptron_colors = list()

    def add_point_data(self, data):
        g0 = list()
        g1 = list()
        for i in data:
            if i[2] == 0:
                g0.append(i[0:2])
            elif i[2] == 1:
                g1.append(i[0:2])
            else:
                raise Exception('Unknown data label: ' + str(i[0:2]))
        g0 = np.array(g0)
        g1 = np.array(g1)

        self.add_point_group(g0, 'red')
        self.add_point_group(g1, 'blue')

    def add_point_group(self, group, color):
        s = group.shape
        if s[1] != 2:
            raise Exception('Bad shape ' + str(s) + ' Input data size must be Nx2')

        self.point_colors.append(color)
        self.point_groups.append(group)

    def add_perceptron(self, p, color):
        self.perceptrons.append(p)
        self.perceptron_colors.append(color)

    def show(self):

        min_x = sys.float_info.max
        max_x = sys.float_info.min
        min_y = sys.float_info.max
        max_y = sys.float_info.min

        for i in range(0, len(self.point_groups)):
            g = self.point_groups[i]
            c = self.point_colors[i]
            mp.plot(g[:, 0:1], g[:, 1:2], linestyle='none', marker='o', markerfacecolor=c, markersize=self.pointsize)

            min = np.asscalar(np.amin(g[:, 0:1], axis=0))
            if min < min_x:
                min_x = min
            min = np.asscalar(np.amin(g[:, 1:2], axis=0))
            if min < min_y:
                min_y = min

            max = np.asscalar(np.amax(g[:, 0:1], axis=0))
            if max > max_x:
                max_x = max
            max = np.asscalar(np.amax(g[:, 1:2], axis=0))
            if max > max_y:
                max_y = max

        if min_x == max_x:
            min_x -= 5
            max_x += 5
        if min_y == max_y:
            min_y -= 5
            max_y += 5

        for i in range(0, len(self.perceptrons)):
            c = self.perceptron_colors[i]
            p = self.perceptrons[i]

            x = min_x
            step = 0.1
            prep_func = list()
            while x <= max_x:
                y = p.calc_var(x)
                prep_func.append([x, y])
                x += step

            pg = np.array(prep_func)
            mp.plot(pg[:, 0:1], pg[:, 1:2], markerfacecolor=c)

        mp.axhline(y=0, color='gray')
        mp.axvline(x=0, color='gray')
        mp.title(self.title)
        mp.show()

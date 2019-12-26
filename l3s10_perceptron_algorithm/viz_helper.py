import sys

import matplotlib.pyplot as mp
import numpy as np


class Color:
    RED = [1, 0, 0]
    GREEN = [0, 1, 0]
    BLUE = [0, 0, 1]
    CYAN = [0, 0.7, 0.7]
    PURPLE = [0.7, 0, 0.7]
    YELLOW = [0.7, 0.7, 0]
    BLACK = [0, 0, 0]
    WHITE = [1, 1, 1]
    GREY3 = [0.3, 0.3, 0.3]
    GREY2 = [0.5, 0.5, 0.5]
    GREY1 = [0.7, 0.7, 0.7]


class Visualizer:
    def __init__(self, title='', pointsize=6):
        self.pointsize = pointsize
        self.title = title
        self.point_groups = list()
        self.point_colors = list()
        self.perceptrons = list()
        self.perceptron_colors = list()

    def clear(self):
        self.clear_perceptrons()
        self.clear_point_groups()

    def clear_point_groups(self):
        self.point_groups.clear()
        self.point_colors.clear()

    def clear_perceptrons(self):
        self.perceptrons.clear()
        self.perceptron_colors.clear()

    def add_point_data(self, data, labels):
        g0 = list()
        g1 = list()
        for i in range(0, len(labels)):
            l = labels[i]
            d = data[i]
            if l == 0:
                g0.append(d)
            elif l == 1:
                g1.append(d)
            else:
                raise Exception('Unknown data label: ' + str(i))
        g0 = np.array(g0)
        g1 = np.array(g1)

        self.add_point_group(g0, Color.RED)
        self.add_point_group(g1, Color.BLUE)

    def add_point_group(self, group, color):
        s = group.shape
        if len(s) == 1:
            return
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

        ext_pts = 3
        min_x -= ext_pts
        max_x += ext_pts
        min_y -= ext_pts
        max_y += ext_pts

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
            mp.plot(pg[:, 0:1], pg[:, 1:2], color=c)

        mp.axhline(y=0, color=Color.GREY2)
        mp.axvline(x=0, color=Color.GREY2)
        mp.title(self.title)

        cfm = mp.get_current_fig_manager()
        cfm.window.maximize()
        mp.show()

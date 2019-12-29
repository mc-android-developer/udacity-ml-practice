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


class VizFunc:
    """
     This is generic interface for function which can be rendered by Visualizer class
    """

    def f(self, x):
        return 0

    def limits(self):
        return 0, 0


class Visualizer:
    def __init__(self, title='', pointsize=6):
        self.pointsize = pointsize
        self.title = title
        self.point_groups = list()
        self.point_colors = list()
        self.func = list()
        self.func_colors = list()

    def clear(self):
        self.clear_funcs()
        self.clear_point_groups()

    def clear_point_groups(self):
        self.point_groups.clear()
        self.point_colors.clear()

    def clear_funcs(self):
        self.func.clear()
        self.func_colors.clear()

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

    def add_func(self, f, color):
        if not isinstance(f, VizFunc):
            raise Exception('F must be subclass of VizFunc')

        self.func.append(f)
        self.func_colors.append(color)

    def show(self):

        min_x = sys.float_info.max
        max_x = sys.float_info.min

        for i in range(0, len(self.point_groups)):
            points = self.point_groups[i]
            color = self.point_colors[i]
            mp.plot(points[:, 0:1], points[:, 1:2], linestyle='none', marker='o', markerfacecolor=color, markersize=self.pointsize)

            min = np.asscalar(np.amin(points[:, 0:1], axis=0))
            if min < min_x:
                min_x = min

            max = np.asscalar(np.amax(points[:, 0:1], axis=0))
            if max > max_x:
                max_x = max

        ext_pts = 3
        min_x -= ext_pts
        max_x += ext_pts

        for i in range(0, len(self.func)):
            color = self.func_colors[i]
            func = self.func[i]

            l1, l2 = func.limits()
            if l1 < l2:
                min_x = l1
                max_x = l2

            x = min_x
            step = 0.1
            func_data = list()
            while x <= max_x:
                y = func.f(x)
                func_data.append([x, y])
                x += step

            points = np.array(func_data)
            mp.plot(points[:, 0:1], points[:, 1:2], color=color)

        mp.axhline(y=0, color=Color.GREY2)
        mp.axvline(x=0, color=Color.GREY2)
        mp.title(self.title)

        cfm = mp.get_current_fig_manager()
        cfm.window.maximize()
        mp.show()

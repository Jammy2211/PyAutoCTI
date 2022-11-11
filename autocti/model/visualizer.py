from os import path
from autoconf import conf
from autoarray.plot.wrap.base import Output
from autoarray.plot.mat_plot.one_d import MatPlot1D
from autoarray.plot.mat_plot.two_d import MatPlot2D
from autoarray.plot.include.one_d import Include1D
from autoarray.plot.include.two_d import Include2D


def setting(section, name):
    return conf.instance["visualize"]["plots"][section][name]


def plot_setting(section, name):
    return setting(section, name)


class Visualizer:
    def __init__(self, visualize_path):

        self.visualize_path = visualize_path

        self.include_1d = Include1D()
        self.include_2d = Include2D()

    def mat_plot_1d_from(self, subfolders, format="png"):
        return MatPlot1D(
            output=Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def mat_plot_2d_from(self, subfolders, format="png"):
        return MatPlot2D(
            output=Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

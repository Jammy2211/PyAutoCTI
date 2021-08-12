from os import path
from autoconf import conf
from autoarray.plot.mat_wrap.wrap import wrap_base
from autoarray.plot.mat_wrap import mat_plot
from autoarray.plot.mat_wrap import include as inc


def setting(section, name):
    return conf.instance["visualize"]["plots"][section][name]


def plot_setting(section, name):
    return setting(section, name)


class Visualizer:
    def __init__(self, visualize_path):

        self.visualize_path = visualize_path

        self.include_1d = inc.Include1D()
        self.include_2d = inc.Include2D()

    def mat_plot_1d_from(self, subfolders, format="png"):
        return mat_plot.MatPlot1D(
            output=wrap_base.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def mat_plot_2d_from(self, subfolders, format="png"):
        return mat_plot.MatPlot2D(
            output=wrap_base.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

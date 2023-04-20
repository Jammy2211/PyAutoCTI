from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

from autoarray.plot.abstract_plotters import AbstractPlotter

from autocti.plot.get_visuals.one_d import GetVisuals1D
from autocti.plot.get_visuals.two_d import GetVisuals2D


class Plotter(AbstractPlotter):
    @property
    def get_1d(self):
        return GetVisuals1D(visuals=self.visuals_1d, include=self.include_1d)

    @property
    def get_2d(self):
        return GetVisuals2D(visuals=self.visuals_2d, include=self.include_2d)

    def text_manual_dict_from(self, region: str):

        try:
            dataset = self.dataset
        except AttributeError:
            dataset = self.fit.dataset

        try:
            fpr_value = dataset.fpr_value
        except AttributeError:
            fpr_value = None

        text_manual_dict = {}

        if fpr_value is not None and "eper" in region:
            fpr_dict = {"FPR (e-)": dataset.fpr_value}
            text_manual_dict = {**text_manual_dict, **fpr_dict}

        if dataset.settings_dict is not None:
            text_manual_dict = {**text_manual_dict, **dataset.settings_dict}

        return text_manual_dict

    def text_manual_dict_y_from(self, region: str):

        if region is None or "eper" in region:
            return 0.94
        return 0.34

from typing import Optional

from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

import autoarray.plot as aplt

from autoarray.plot.abstract_plotters import AbstractPlotter

from autocti.plot.get_visuals.one_d import GetVisuals1D
from autocti.plot.get_visuals.two_d import GetVisuals2D
from autocti.extract.settings import SettingsExtract

from autocti import exc

class Plotter(AbstractPlotter):
    def __init__(
        self,
        dataset,
        mat_plot_2d: aplt.MatPlot2D = None,
        visuals_2d: aplt.Visuals2D = None,
        include_2d: aplt.Include2D = None,
        mat_plot_1d: aplt.MatPlot1D = None,
        visuals_1d: aplt.Visuals1D = None,
        include_1d: aplt.Include1D = None,
    ):
        self.dataset = dataset

        super().__init__(
            mat_plot_2d=mat_plot_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
            mat_plot_1d=mat_plot_1d,
            visuals_1d=visuals_1d,
            include_1d=include_1d,
        )

    @property
    def get_1d(self):
        return GetVisuals1D(visuals=self.visuals_1d, include=self.include_1d)

    @property
    def get_2d(self):
        return GetVisuals2D(visuals=self.visuals_2d, include=self.include_2d)

    def title_str_from(self, region: Optional[str]) -> str:
        if self.dataset.settings_dict is not None:
            ccd_str = self.dataset.settings_dict.get("CCD")
        else:
            ccd_str = None

        if region is None:
            title_str = ""
        elif region == "fpr":
            title_str = "FPR"
        elif region == "eper":
            title_str = "EPER"
        elif region == "parallel_fpr":
            title_str = "Parallel FPR"
        elif region == "parallel_eper":
            title_str = "Parallel EPER"
        elif region == "serial_fpr":
            title_str = "Serial FPR"
        elif region == "serial_eper":
            title_str = "Serial EPER"

        if ccd_str is None:
            return title_str
        return f"{ccd_str} {title_str}"

    def title_str_2d_from(self) -> Optional[str]:
        if self.dataset.settings_dict is not None:
            ccd_str = self.dataset.settings_dict.get("CCD")
            ig1_str = self.dataset.settings_dict.get("CI_IG1")
            ig2_str = self.dataset.settings_dict.get("CI_IG2")
            id_delay_str = self.dataset.settings_dict.get("CI_IDDLY")

            return f"{ccd_str} IG1={ig1_str} IG2={ig2_str} IDD={id_delay_str}"

    def text_manual_dict_from(self, region: Optional[str] = None):
        try:
            fpr_value = self.dataset.fpr_value
        except AttributeError:
            fpr_value = None

        text_manual_dict = {}

        if region is not None:
            if fpr_value is not None and "eper" in region:
                fpr_dict = {"FPR (e-)": self.dataset.fpr_value}
                text_manual_dict = {**text_manual_dict, **fpr_dict}

        if self.dataset.settings_dict is not None:
            text_manual_dict = {**text_manual_dict, **self.dataset.settings_dict}

        return text_manual_dict

    def text_manual_dict_y_from(self, region: Optional[str] = None):
        if region is None or "eper" in region:
            return 0.94
        return 0.34

    def should_plot_zero_from(self, region: Optional[str]):
        if region is None:
            return False

        if "eper" in region:
            return True

        return False

    def fpr_mask_from(self):

        fpr_size = self.dataset.layout.parallel_rows_within_regions[0]

        if any(
            [
                fpr_size != fpr_size_of_row
                for fpr_size_of_row in self.dataset.layout.parallel_rows_within_regions
            ]
        ):
            raise exc.PlottingException(
                "The FPR in this dataset have a variable number of rows. This means that masknig the FPR in the"
                "figures_1d_data_binned method is not supported."
            )

        fpr_mask = self.dataset.layout.extract.parallel_fpr.mask_from(
            settings=SettingsExtract(pixels=(0, fpr_size)),
            pixel_scales=self.dataset.pixel_scales,
        )

        serial_prescan = self.dataset.layout.extract.serial_prescan.serial_prescan
        fpr_mask[
            serial_prescan.y0 : serial_prescan.y1, serial_prescan.x0 : serial_prescan.x1
        ] = True

        serial_overscan = self.dataset.layout.extract.serial_overscan.serial_overscan
        fpr_mask[
            serial_overscan.y0 : serial_overscan.y1,
            serial_overscan.x0 : serial_overscan.x1,
        ] = True

        return fpr_mask
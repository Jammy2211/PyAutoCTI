from typing import List, Optional, Union

from autoconf import conf
from autoconf.dictable import output_to_json

import autoarray as aa
import autofit as af

from autocti.charge_injection.imaging.imaging import ImagingCI
from autocti.charge_injection.model.result import ResultImagingCI
from autocti.clocker.one_d import Clocker1D
from autocti.clocker.two_d import Clocker2D
from autocti.dataset_1d.dataset_1d.dataset_1d import Dataset1D
from autocti.model.settings import SettingsCTI1D
from autocti.model.settings import SettingsCTI2D


class AnalysisCTI(af.Analysis):
    def __init__(
        self,
        dataset: Union[Dataset1D, ImagingCI],
        clocker: Union[Clocker1D, Clocker2D],
        settings_cti: Union[SettingsCTI1D, SettingsCTI2D],
        dataset_full: Optional[aa.AbstractDataset] = None,
    ):
        """
        Fits a CTI model to a CTI calibration dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit a CTI model via a `CTI2D` object to a charge injection
        imaging dataset.

        Parameters
        ----------
        dataset
            The charge injection dataset that the model is fitted to.
        clocker
            The CTI arctic clocker used by the non-linear search and model-fit.
        settings_cti
            The settings controlling aspects of the CTI model in this model-fit.
        dataset_full
            The full dataset, which is visualized separate from the `dataset` that is fitted, which for example may
            not have the FPR masked and thus enable visualization of the FPR.
        """
        super().__init__()

        self.dataset = dataset
        self.clocker = clocker
        self.settings_cti = settings_cti
        self.dataset_full = dataset_full

    def region_list_from(self) -> List:
        raise NotImplementedError

    def save_results_combined(self, paths: af.DirectoryPaths, result: ResultImagingCI):
        """
        At the end of a model-fit, this routine saves attributes of the `Analysis` object to the `files`
        folder such that they can be loaded after the analysis using PyAutoFit's database and aggregator tools.

        For this analysis it outputs the following:

        - The Israel et al requirement on the spurious ellipticity based on the errors of the fit.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        result
            The result of a model fit, including the non-linear search and samples.
        """

        if result.samples is None:
            return

        weight_list = []
        delta_ellipticity_list = []

        for sample in result.samples.sample_list:
            instance = sample.instance_for_model(model=result.samples.model)

            weight_list.append(sample.weight)
            delta_ellipticity_list.append(instance.cti.delta_ellipticity)

        (
            median_delta_ellipticity,
            upper_delta_ellipticity,
            lower_delta_ellipticity,
        ) = af.marginalize(
            parameter_list=delta_ellipticity_list,
            sigma=2.0,
            weight_list=weight_list,
        )

        delta_ellipticity = (upper_delta_ellipticity - lower_delta_ellipticity) / 2.0

        output_to_json(
            obj=delta_ellipticity,
            file_path=paths._files_path / "delta_ellipticity.json",
        )

    def in_ascending_fpr_order_from(self, quantity_list, fpr_value_list):
        if not conf.instance["visualize"]["general"]["general"][
            "subplot_ascending_fpr"
        ]:
            return quantity_list

        indexes = sorted(range(len(fpr_value_list)), key=lambda k: fpr_value_list[k])

        return [quantity_list[i] for i in indexes]

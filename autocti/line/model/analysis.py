from typing import List

import autofit as af
from autofit.non_linear.abstract_search import Analysis
from autocti.line.dataset import DatasetLine
from autocti.line.fit import FitDatasetLine
from autocti.line.model.visualizer import VisualizerDatasetLine
from autocti.model.result import ResultDataset
from autocti.line.model.result import ResultDatasetLine
from autocti.model.settings import SettingsCTI1D
from autocti.util.clocker import Clocker1D


class AnalysisDatasetLine(Analysis):
    def __init__(
        self,
        dataset_line: DatasetLine,
        clocker: Clocker1D,
        settings_cti: SettingsCTI1D = SettingsCTI1D(),
        results: List[ResultDataset] = None,
    ):

        super().__init__()

        self.dataset_line = dataset_line
        self.clocker = clocker
        self.settings_cti = settings_cti
        self.results = results

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Determine the fitness of a particular model

        Parameters
        ----------
        instance

        Returns
        -------
        fit: Fit
            How fit the model is and the model
        """

        self.settings_cti.check_total_density_within_range(traps=instance.cti.traps)

        fit = self.fit_from_instance(instance=instance)

        return fit.log_likelihood

    def fit_from_instance_and_dataset_line(
        self, instance: af.ModelInstance, dataset_line: DatasetLine
    ) -> FitDatasetLine:

        if instance.cti.traps is not None:
            traps = list(instance.cti.traps)
        else:
            traps = None

        post_cti_data = self.clocker.add_cti(
            pre_cti_data=dataset_line.pre_cti_data, traps=traps, ccd=instance.cti.ccd
        )

        return FitDatasetLine(dataset_line=dataset_line, post_cti_data=post_cti_data)

    def fit_from_instance(self, instance: af.ModelInstance) -> FitDatasetLine:

        return self.fit_from_instance_and_dataset_line(
            instance=instance, dataset_line=self.dataset_line
        )

    def visualize(
        self,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):

        fit = self.fit_from_instance(instance=instance)

        visualizer = VisualizerDatasetLine(visualize_path=paths.image_path)

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ) -> ResultDatasetLine:
        return ResultDatasetLine(
            samples=samples, model=model, analysis=self, search=search
        )

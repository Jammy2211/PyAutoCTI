from typing import List

from autofit.non_linear.samples import PDFSamples
from autofit.mapper.prior_model.collection import CollectionPriorModel as Collection
from autofit.mapper.model import ModelInstance
from autofit.non_linear.abstract_search import Analysis
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.abstract_search import NonLinearSearch

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

    def log_likelihood_function(self, instance: ModelInstance) -> float:
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
        self, instance: ModelInstance, dataset_line: DatasetLine
    ) -> FitDatasetLine:

        if instance.cti.traps is not None:
            traps = list(instance.cti.traps)
        else:
            traps = None

        post_cti_data = self.clocker.add_cti(
            data=dataset_line.pre_cti_data, trap_list=traps, ccd=instance.cti.ccd
        )

        return FitDatasetLine(dataset_line=dataset_line, post_cti_data=post_cti_data)

    def fit_from_instance(self, instance: ModelInstance) -> FitDatasetLine:

        return self.fit_from_instance_and_dataset_line(
            instance=instance, dataset_line=self.dataset_line
        )

    def visualize(
        self, paths: DirectoryPaths, instance: ModelInstance, during_analysis: bool
    ):

        fit = self.fit_from_instance(instance=instance)

        visualizer = VisualizerDatasetLine(visualize_path=paths.image_path)

        visualizer.visualize_dataset_line(dataset_line=self.dataset_line)

        visualizer.visualize_fit_line(fit=fit, during_analysis=during_analysis)

    def make_result(
        self, samples: PDFSamples, model: Collection, search: NonLinearSearch
    ) -> ResultDatasetLine:
        return ResultDatasetLine(
            samples=samples, model=model, analysis=self, search=search
        )

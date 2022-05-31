from typing import List

from autofit.non_linear.samples import PDFSamples
from autofit.mapper.prior_model.collection import CollectionPriorModel
from autofit.mapper.model import ModelInstance
from autofit.non_linear.abstract_search import Analysis
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.abstract_search import NonLinearSearch

from autocti.dataset_1d.dataset_1d.dataset_1d import Dataset1D
from autocti.dataset_1d.fit import FitDataset1D
from autocti.dataset_1d.model.visualizer import VisualizerDataset1D
from autocti.model.result import ResultDataset
from autocti.dataset_1d.model.result import ResultDataset1D
from autocti.model.settings import SettingsCTI1D
from autocti.clocker.one_d import Clocker1D


class AnalysisDataset1D(Analysis):
    def __init__(
        self,
        dataset: Dataset1D,
        clocker: Clocker1D,
        settings_cti: SettingsCTI1D = SettingsCTI1D(),
        results: List[ResultDataset] = None,
    ):

        super().__init__()

        self.dataset = dataset
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

        self.settings_cti.check_total_density_within_range(traps=instance.cti.trap_list)

        fit = self.fit_via_instance_from(instance=instance)

        return fit.log_likelihood

    def fit_via_instance_and_dataset_from(
        self, instance: ModelInstance, dataset: Dataset1D
    ) -> FitDataset1D:

        post_cti_data = self.clocker.add_cti(
            data=dataset.pre_cti_data, cti=instance.cti
        )

        return FitDataset1D(dataset=dataset, post_cti_data=post_cti_data)

    def fit_via_instance_from(self, instance: ModelInstance) -> FitDataset1D:

        return self.fit_via_instance_and_dataset_from(
            instance=instance, dataset=self.dataset
        )

    def visualize(
        self, paths: DirectoryPaths, instance: ModelInstance, during_analysis: bool
    ):

        fit = self.fit_via_instance_from(instance=instance)

        visualizer = VisualizerDataset1D(visualize_path=paths.image_path)

        visualizer.visualize_dataset_1d(dataset_1d=self.dataset)

        visualizer.visualize_fit_line(fit=fit, during_analysis=during_analysis)

    def make_result(
        self,
        samples: PDFSamples,
        model: CollectionPriorModel,
        sigma=1.0,
        use_errors=True,
        use_widths=False,
    ) -> ResultDataset1D:
        return ResultDataset1D(samples=samples, model=model, analysis=self)

import os

import autoarray as aa
import autofit as af

from autocti.charge_injection.imaging.imaging import ImagingCI
from autocti.charge_injection.fit import FitImagingCI
from autocti.charge_injection.model.visualizer import VisualizerImagingCI
from autocti.charge_injection.model.result import ResultImagingCI
from autocti.clocker.two_d import Clocker2D
from autocti.charge_injection.hyper import HyperCINoiseCollection
from autocti.model.settings import SettingsCTI2D
from autocti.preloads import Preloads

from autocti import exc


class AnalysisImagingCI(af.Analysis):
    def __init__(
        self, dataset: ImagingCI, clocker: Clocker2D, settings_cti=SettingsCTI2D()
    ):

        super().__init__()

        self.dataset = dataset
        self.clocker = clocker
        self.settings_cti = settings_cti

        self.preloads = Preloads()

        parallel_fast_index_list = None
        parallel_fast_column_lists = None

        serial_fast_index_list = None
        serial_fast_row_lists = None

        if self.clocker.parallel_fast_mode and not self.clocker.serial_fast_mode:

            (
                parallel_fast_index_list,
                parallel_fast_column_lists,
            ) = clocker.fast_indexes_from(data=dataset.pre_cti_data, for_parallel=True)

        elif not self.clocker.parallel_fast_mode and self.clocker.serial_fast_mode:

            serial_fast_index_list, serial_fast_row_lists = clocker.fast_indexes_from(
                data=dataset.pre_cti_data, for_parallel=False
            )

        elif self.clocker.parallel_fast_mode and self.clocker.serial_fast_mode:

            raise exc.ClockerException(
                "Both parallel fast model and serial fast mode cannot be turned on.\n"
                "Only switch on parallel fast mode for parallel + serial clocking."
            )

        self.preloads = Preloads(
            parallel_fast_index_list=parallel_fast_index_list,
            parallel_fast_column_lists=parallel_fast_column_lists,
            serial_fast_index_list=serial_fast_index_list,
            serial_fast_row_lists=serial_fast_row_lists,
        )

    def modify_before_fit(self, paths: af.DirectoryPaths, model: af.Collection):
        """
        PyAutoFit calls this function immediately before the non-linear search begins, therefore it can be used to
        perform tasks using the final model parameterization.

        This function:

         1) Visualizes the charge injection imaging dataset, which does not change during the analysis and thus can be
         done once.

         2) Checks if the noise-map is fixed (it is not if hyper functionality is on), and if it is fixed it
         sets the noise-normalization to the preloads for computational speed.

        Parameters
        ----------
        paths
            The PyAutoFit paths object which manages all paths, e.g. where the non-linear search outputs are stored,
            visualization and the pickled objects used by the aggregator output by this function.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        """

        if not paths.is_complete:

            if not model.has(HyperCINoiseCollection):

                noise_normalization = aa.util.fit.noise_normalization_with_mask_from(
                    noise_map=self.dataset.noise_map, mask=self.dataset.mask
                )

                self.preloads.noise_normalization = noise_normalization

            if not os.environ.get("PYAUTOFIT_TEST_MODE") == "1":

                visualizer = VisualizerImagingCI(visualize_path=paths.image_path)

                visualizer.visualize_imaging_ci(imaging_ci=self.dataset)

                try:
                    visualizer.visualize_imaging_ci_lines(
                        imaging_ci=self.dataset, region="parallel_fpr"
                    )
                    visualizer.visualize_imaging_ci_lines(
                        imaging_ci=self.dataset, region="parallel_eper"
                    )

                except (exc.RegionException, TypeError, ValueError):
                    pass

                try:
                    visualizer.visualize_imaging_ci(imaging_ci=self.dataset)
                    visualizer.visualize_imaging_ci_lines(
                        imaging_ci=self.dataset, region="serial_fpr"
                    )
                    visualizer.visualize_imaging_ci_lines(
                        imaging_ci=self.dataset, region="serial_eper"
                    )

                except (exc.RegionException, TypeError, ValueError):
                    pass

        return self

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

        self.settings_cti.check_total_density_within_range(
            parallel_traps=instance.cti.parallel_trap_list,
            serial_traps=instance.cti.serial_trap_list,
        )

        fit = self.fit_via_instance_and_dataset_from(
            instance=instance, imaging_ci=self.dataset, hyper_noise_scale=True
        )

        return fit.figure_of_merit

    def fit_via_instance_and_dataset_from(
        self,
        instance: af.ModelInstance,
        imaging_ci: ImagingCI,
        hyper_noise_scale: bool = True,
    ) -> FitImagingCI:

        hyper_noise_scalar_dict = None

        if hyper_noise_scale and hasattr(instance, "hyper_noise"):
            hyper_noise_scalar_dict = instance.hyper_noise.as_dict

        post_cti_data = self.clocker.add_cti(
            data=imaging_ci.pre_cti_data, cti=instance.cti, preloads=self.preloads
        )

        return FitImagingCI(
            dataset=imaging_ci,
            post_cti_data=post_cti_data,
            hyper_noise_scalar_dict=hyper_noise_scalar_dict,
            preloads=self.preloads
        )

    def fit_via_instance_from(
        self, instance: af.ModelInstance, hyper_noise_scale: bool = True
    ) -> FitImagingCI:

        return self.fit_via_instance_and_dataset_from(
            instance=instance,
            imaging_ci=self.dataset,
            hyper_noise_scale=hyper_noise_scale,
        )

    def visualize(
        self,
        paths: af.DirectoryPaths,
        instance: af.ModelInstance,
        during_analysis: bool,
    ):

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            return

        fit = self.fit_via_instance_from(instance=instance)

        visualizer = VisualizerImagingCI(visualize_path=paths.image_path)

        try:
            visualizer.visualize_fit_ci(fit=fit, during_analysis=during_analysis)
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit, during_analysis=during_analysis, region="parallel_fpr"
            )
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit, during_analysis=during_analysis, region="parallel_eper"
            )
        except (exc.RegionException, TypeError, ValueError):
            pass

        try:
            visualizer.visualize_fit_ci(fit=fit, during_analysis=during_analysis)
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit, during_analysis=during_analysis, region="serial_fpr"
            )
            visualizer.visualize_fit_ci_1d_lines(
                fit=fit, during_analysis=during_analysis, region="serial_eper"
            )
        except (exc.RegionException, TypeError, ValueError):
            pass

    def make_result(
        self,
        samples: af.SamplesPDF,
        model: af.CollectionPriorModel,
        sigma=1.0,
        use_errors=True,
        use_widths=False,
    ) -> ResultImagingCI:
        return ResultImagingCI(samples=samples, model=model, analysis=self)

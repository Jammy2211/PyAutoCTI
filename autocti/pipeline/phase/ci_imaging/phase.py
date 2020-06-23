from autocti.pipeline.phase.settings import PhaseSettingsCIImaging
from autocti.charge_injection import ci_mask
from autocti.pipeline.phase.dataset.phase import PhaseDataset
from autocti.pipeline.phase.ci_imaging.analysis import Analysis
from autocti.pipeline.phase.ci_imaging.meta_ci_imaging import MetaCIImaging
from autocti.pipeline.phase.ci_imaging.result import Result
from autofit.non_linear.paths import convert_paths
from autofit.tools.phase_property import PhaseProperty


class PhaseCIImaging(PhaseDataset):

    hyper_noise_scalar_of_ci_regions = PhaseProperty("hyper_noise_scalar_of_ci_regions")
    hyper_noise_scalar_of_parallel_trails = PhaseProperty(
        "hyper_noise_scalar_of_parallel_trails"
    )
    hyper_noise_scalar_of_serial_trails = PhaseProperty(
        "hyper_noise_scalar_of_serial_trails"
    )
    hyper_noise_scalar_of_serial_overscan_no_trails = PhaseProperty(
        "hyper_noise_scalar_of_serial_overscan_no_trails"
    )

    Analysis = Analysis
    Result = Result

    @convert_paths
    def __init__(
        self,
        paths,
        *,
        search,
        parallel_traps=None,
        parallel_ccd_volume=None,
        serial_traps=None,
        serial_ccd_volume=None,
        hyper_noise_scalar_of_ci_regions=None,
        hyper_noise_scalar_of_parallel_trails=None,
        hyper_noise_scalar_of_serial_trails=None,
        hyper_noise_scalar_of_serial_overscan_no_trails=None,
        settings=PhaseSettingsCIImaging(),
    ):

        """

        A phase in an cti pipeline. Uses the set non_linear search to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        search: class
            The class of a non_linear search
        sub_size: int
            The side length of the subgrid
        """

        super().__init__(
            paths=paths,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
            settings=settings,
            search=search,
        )

        self.hyper_noise_scalar_of_ci_regions = hyper_noise_scalar_of_ci_regions
        self.hyper_noise_scalar_of_parallel_trails = (
            hyper_noise_scalar_of_parallel_trails
        )
        self.hyper_noise_scalar_of_serial_trails = hyper_noise_scalar_of_serial_trails
        self.hyper_noise_scalar_of_serial_overscan_no_trails = (
            hyper_noise_scalar_of_serial_overscan_no_trails
        )

        self.meta_dataset = MetaCIImaging(model=self.model, settings=settings)

    def make_analysis(self, datasets, clocker, results=None, pool=None):
        """
        Create an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
        change the behaviour of the phase.

        Parameters
        ----------
        clocker
        datasets
        pool
        results: [Results]
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear search calls to determine the fit of a set of values
        """

        mask = ci_mask.CIMask.unmasked(
            shape_2d=datasets[0].image.shape_2d, pixel_scales=datasets[0].image.shape_2d
        )

        masks = [
            self.meta_dataset.mask_for_analysis_from_dataset(dataset=dataset, mask=mask)
            for dataset in datasets
        ]

        noise_scaling_maps_list = self.meta_dataset.noise_scaling_maps_list_from_total_images_and_results(
            total_images=len(datasets), results=results
        )

        masked_ci_datasets = [
            self.meta_dataset.masked_ci_dataset_from_dataset(
                dataset=dataset, mask=mask, noise_scaling_maps=maps
            )
            for dataset, mask, maps in zip(datasets, masks, noise_scaling_maps_list)
        ]

        return Analysis(
            masked_ci_imagings=masked_ci_datasets,
            clocker=clocker,
            parallel_total_density_range=self.meta_dataset.settings.parallel_total_density_range,
            serial_total_density_range=self.meta_dataset.settings.serial_total_density_range,
            image_path=self.search.paths.image_path,
            results=results,
            pool=pool,
        )

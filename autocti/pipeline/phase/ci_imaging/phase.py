from autocti.pipeline.phase.settings import SettingsPhaseCIImaging
from autocti.charge_injection import ci_mask, ci_imaging
from autocti.pipeline.phase.dataset.phase import PhaseDataset
from autocti.pipeline.phase.ci_imaging.analysis import Analysis
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
        parallel_ccd=None,
        serial_traps=None,
        serial_ccd=None,
        hyper_noise_scalar_of_ci_regions=None,
        hyper_noise_scalar_of_parallel_trails=None,
        hyper_noise_scalar_of_serial_trails=None,
        hyper_noise_scalar_of_serial_overscan_no_trails=None,
        settings=SettingsPhaseCIImaging(),
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
            parallel_ccd=parallel_ccd,
            serial_traps=serial_traps,
            serial_ccd=serial_ccd,
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

    def make_analysis(self, datasets, clocker, results=None, pool=None):
        """
        Returns an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
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
            self.mask_for_analysis_from_dataset(dataset=dataset, mask=mask)
            for dataset in datasets
        ]

        noise_scaling_maps_list = self.noise_scaling_maps_list_from_total_images_and_results(
            total_images=len(datasets), results=results
        )

        settings_masked_ci_imaging = self.settings.settings_masked_ci_imaging.modify_via_fit_type(
            is_parallel_fit=self.is_parallel_fit, is_serial_fit=self.is_serial_fit
        )

        masked_ci_imagings = [
            ci_imaging.MaskedCIImaging(
                ci_imaging=dataset,
                mask=mask,
                noise_scaling_maps=maps,
                settings=settings_masked_ci_imaging,
            )
            for dataset, mask, maps in zip(datasets, masks, noise_scaling_maps_list)
        ]

        return Analysis(
            masked_ci_imagings=masked_ci_imagings,
            clocker=clocker,
            settings_cti=self.settings.settings_cti,
            image_path=self.search.paths.image_path,
            results=results,
            pool=pool,
        )

    def mask_for_analysis_from_dataset(self, dataset, mask):

        mask = self.mask_for_analysis_from_cosmic_ray_map(
            cosmic_ray_map=dataset.cosmic_ray_map, mask=mask
        )

        return ci_mask.CIMask.masked_front_edges_and_trails_from_ci_frame(
            mask=mask, ci_frame=dataset.image, settings=self.settings.settings_ci_mask
        )

    def noise_scaling_maps_list_from_total_images_and_results(
        self, total_images, results
    ):

        if self.model.hyper_noise_scalar_of_ci_regions is not None:
            noise_scaling_maps_list_of_ci_regions = (
                results.last.noise_scaling_maps_list_of_ci_regions
            )
        else:
            noise_scaling_maps_list_of_ci_regions = total_images * [None]

        if self.model.hyper_noise_scalar_of_parallel_trails is not None:
            noise_scaling_maps_list_of_parallel_trails = (
                results.last.noise_scaling_maps_list_of_parallel_trails
            )
        else:
            noise_scaling_maps_list_of_parallel_trails = total_images * [None]

        if self.model.hyper_noise_scalar_of_serial_trails is not None:
            noise_scaling_maps_list_of_serial_trails = (
                results.last.noise_scaling_maps_list_of_serial_trails
            )
        else:
            noise_scaling_maps_list_of_serial_trails = total_images * [None]

        if self.model.hyper_noise_scalar_of_serial_overscan_no_trails is not None:
            noise_scaling_maps_list_of_serial_overscan_no_trails = (
                results.last.noise_scaling_maps_list_of_serial_overscan_no_trails
            )
        else:
            noise_scaling_maps_list_of_serial_overscan_no_trails = total_images * [None]

        noise_scaling_maps_list = []

        for image_index in range(total_images):
            noise_scaling_maps_list.append(
                [
                    noise_scaling_maps_list_of_ci_regions[image_index],
                    noise_scaling_maps_list_of_parallel_trails[image_index],
                    noise_scaling_maps_list_of_serial_trails[image_index],
                    noise_scaling_maps_list_of_serial_overscan_no_trails[image_index],
                ]
            )

        for image_index in range(total_images):
            noise_scaling_maps_list[image_index] = [
                noise_scaling_map
                for noise_scaling_map in noise_scaling_maps_list[image_index]
                if noise_scaling_map is not None
            ]

        noise_scaling_maps_list = list(filter(None, noise_scaling_maps_list))

        if len(noise_scaling_maps_list) == 0:
            return total_images * [None]

        return noise_scaling_maps_list

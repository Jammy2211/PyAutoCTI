from astropy import cosmology as cosmo

import autofit as af
from autocti.pipeline import tagging
from autocti.pipeline.phase import dataset
from autocti.pipeline.phase.ci_imaging.analysis import Analysis
from autocti.pipeline.phase.ci_imaging.meta_ci_imaging import MetaCIImaging
from autocti.pipeline.phase.ci_imaging.result import Result


class PhaseCIImaging(dataset.PhaseDataset):

    hyper_noise_scalar_of_ci_regions = af.PhaseProperty(
        "hyper_noise_scalar_of_ci_regions"
    )
    hyper_noise_scalar_of_parallel_trails = af.PhaseProperty(
        "hyper_noise_scalar_of_parallel_trails"
    )
    hyper_noise_scalar_of_serial_trails = af.PhaseProperty(
        "hyper_noise_scalar_of_serial_trails"
    )
    hyper_noise_scalar_of_serial_overscan_no_trails = af.PhaseProperty(
        "hyper_noise_scalar_of_serial_overscan_no_trails"
    )

    Analysis = Analysis
    Result = Result

    @af.convert_paths
    def __init__(
        self,
        paths,
        *,
        parallel_traps=(),
        parallel_ccd_volume=None,
        serial_traps=(),
        serial_ccd_volume=None,
        hyper_noise_scalar_of_ci_regions=None,
        hyper_noise_scalar_of_parallel_trails=None,
        hyper_noise_scalar_of_serial_trails=None,
        hyper_noise_scalar_of_serial_overscan_no_trails=None,
        non_linear_class=af.MultiNest,
        columns=None,
        rows=None,
        parallel_front_edge_mask_rows=None,
        parallel_trails_mask_rows=None,
        parallel_total_density_range=None,
        serial_front_edge_mask_columns=None,
        serial_trails_mask_columns=None,
        serial_total_density_range=None,
        cosmic_ray_parallel_buffer=10,
        cosmic_ray_serial_buffer=10,
        cosmic_ray_diagonal_buffer=3,
    ):

        """

        A phase in an cti pipeline. Uses the set non_linear optimizer to try to fit models and hyper_galaxies
        passed to it.

        Parameters
        ----------
        non_linear_class: class
            The class of a non_linear optimizer
        sub_size: int
            The side length of the subgrid
        """

        phase_tag = tagging.phase_tag_from_phase_settings(
            columns=columns,
            rows=rows,
            parallel_front_edge_mask_rows=parallel_front_edge_mask_rows,
            parallel_trails_mask_rows=parallel_trails_mask_rows,
            serial_front_edge_mask_columns=serial_front_edge_mask_columns,
            serial_trails_mask_columns=serial_trails_mask_columns,
            parallel_total_density_range=parallel_total_density_range,
            serial_total_density_range=serial_total_density_range,
            cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
            cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
            cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer,
        )
        paths.phase_tag = phase_tag

        super().__init__(
            paths,
            parallel_traps=parallel_traps,
            parallel_ccd_volume=parallel_ccd_volume,
            serial_traps=serial_traps,
            serial_ccd_volume=serial_ccd_volume,
            hyper_noise_scalar_of_ci_regions=hyper_noise_scalar_of_ci_regions,
            hyper_noise_scalar_of_parallel_trails=hyper_noise_scalar_of_parallel_trails,
            hyper_noise_scalar_of_serial_trails=hyper_noise_scalar_of_serial_trails,
            hyper_noise_scalar_of_serial_overscan_no_trails=hyper_noise_scalar_of_serial_overscan_no_trails,
            non_linear_class=non_linear_class,
        )

        self.hyper_noise_scalar_of_ci_regions = hyper_noise_scalar_of_ci_regions
        self.hyper_noise_scalar_of_parallel_trails = (
            hyper_noise_scalar_of_parallel_trails
        )
        self.hyper_noise_scalar_of_serial_trails = hyper_noise_scalar_of_serial_trails
        self.hyper_noise_scalar_of_serial_overscan_no_trails = (
            hyper_noise_scalar_of_serial_overscan_no_trails
        )

        self.meta_dataset = MetaCIImaging(
            model=self.model,
            columns=columns,
            rows=rows,
            parallel_front_edge_mask_rows=parallel_front_edge_mask_rows,
            parallel_trails_mask_rows=parallel_trails_mask_rows,
            parallel_total_density_range=parallel_total_density_range,
            serial_front_edge_mask_columns=serial_front_edge_mask_columns,
            serial_trails_mask_columns=serial_trails_mask_columns,
            serial_total_density_range=serial_total_density_range,
            cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
            cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
            cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer,
        )

    def make_analysis(self, ci_datas, cti_settings, results=None, pool=None):
        """
        Create an analysis object. Also calls the prior passing and image modifying functions to allow child classes to
        change the behaviour of the phase.

        Parameters
        ----------
        cti_settings
        ci_datas
        pool
        results: [Results]
            The result from the previous phase

        Returns
        -------
        analysis: Analysis
            An analysis object that the non-linear optimizer calls to determine the fit of a set of values
        """

        masks = self.meta_dataset.masks_for_analysis_from_ci_datas(ci_datas=ci_datas)

        noise_scaling_maps_list = self.meta_dataset.noise_scaling_maps_list_from_total_images_and_results(
            total_images=len(ci_datas), results=results
        )

        ci_datas_masked_extracted = [
            self.meta_dataset.ci_datas_masked_extracted_from_ci_data(
                ci_data=data, mask=mask, noise_scaling_maps_list=maps
            )
            for data, mask, maps in zip(ci_datas, masks, noise_scaling_maps_list)
        ]

        masked_ci_dataset_full = list(
            map(
                lambda data, mask, maps: ci_imaging.MaskedCIImaging(
                    image=data.image,
                    noise_map=data.noise_map,
                    ci_pre_cti=data.ci_pre_cti,
                    mask=mask,
                    ci_pattern=data.ci_pattern,
                    ci_frame=data.ci_frame,
                    noise_scaling_maps_list=maps,
                ),
                ci_datas,
                masks,
                noise_scaling_maps_list,
            )
        )

        analysis = self.__class__.Analysis(
            ci_datas_masked_extracted=ci_datas_masked_extracted,
            masked_ci_dataset_full=masked_ci_dataset_full,
            cti_settings=cti_settings,
            parallel_total_density_range=self.parallel_total_density_range,
            serial_total_density_range=self.serial_total_density_range,
            phase_name=self.phase_name,
            results=results,
            pool=pool,
        )
        return analysis

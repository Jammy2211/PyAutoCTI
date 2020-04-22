from autocti.dataset import imaging
from autocti.charge_injection import ci_mask, ci_imaging
from autocti.pipeline.phase.dataset import meta_dataset


class MetaCIImaging(meta_dataset.MetaDataset):
    def __init__(
        self,
        model,
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

        super().__init__(
            model=model,
            parallel_total_density_range=parallel_total_density_range,
            serial_total_density_range=serial_total_density_range,
            cosmic_ray_parallel_buffer=cosmic_ray_parallel_buffer,
            cosmic_ray_serial_buffer=cosmic_ray_serial_buffer,
            cosmic_ray_diagonal_buffer=cosmic_ray_diagonal_buffer,
        )

        if not self.is_parallel_fit:
            columns = None

        if not self.is_serial_fit:
            rows = None

        self.columns = columns
        self.rows = rows
        self.parallel_front_edge_mask_rows = parallel_front_edge_mask_rows
        self.parallel_trails_mask_rows = parallel_trails_mask_rows
        self.serial_front_edge_mask_columns = serial_front_edge_mask_columns
        self.serial_trails_mask_columns = serial_trails_mask_columns

    def mask_for_analysis_from_dataset(self, dataset, mask):

        mask = self.mask_for_analysis_from_cosmic_ray_map(
            cosmic_ray_map=dataset.cosmic_ray_map, mask=mask
        )

        if self.parallel_front_edge_mask_rows is not None:

            parallel_front_edge_mask = ci_mask.CIMask.masked_parallel_front_edge_from_ci_frame(
                ci_frame=dataset.image, rows=self.parallel_front_edge_mask_rows
            )

            mask = mask + parallel_front_edge_mask

        if self.parallel_trails_mask_rows is not None:

            parallel_trails_mask = ci_mask.CIMask.masked_parallel_trails_from_ci_frame(
                ci_frame=dataset.image, rows=self.parallel_trails_mask_rows
            )

            mask = mask + parallel_trails_mask

        if self.serial_front_edge_mask_columns is not None:

            serial_front_edge_mask = ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(
                ci_frame=dataset.image, columns=self.serial_front_edge_mask_columns
            )

            mask = mask + serial_front_edge_mask

        if self.serial_trails_mask_columns is not None:

            serial_trails_mask = ci_mask.CIMask.masked_serial_trails_from_ci_frame(
                ci_frame=dataset.image, columns=self.serial_trails_mask_columns
            )

            mask = mask + serial_trails_mask

        return mask

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

        if noise_scaling_maps_list == []:
            noise_scaling_maps_list = total_images * [None]

        return noise_scaling_maps_list

    def masked_ci_dataset_from_dataset(self, dataset, mask, noise_scaling_maps=None):

        return ci_imaging.MaskedCIImaging(
            ci_imaging=dataset,
            mask=mask,
            noise_scaling_maps=noise_scaling_maps,
            parallel_columns=self.columns,
            serial_rows=self.rows,
        )

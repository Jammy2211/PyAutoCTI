from autocti.dataset import imaging
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

    def masked_ci_dataset_from_dataset(
        self, dataset, mask, noise_scaling_maps_list=None
    ):

        if self.is_only_parallel_fit:
            return dataset.for_parallel_from_columns(
                columns=(
                    0,
                    self.columns
                    or dataset.ci_frame.frame_geometry.parallel_overscan.total_columns,
                ),
                mask=mask,
                noise_scaling_maps=noise_scaling_maps_list,
            )

        elif self.is_only_serial_fit:
            return dataset.for_serial_from_rows(
                rows=self.rows or (0, dataset.ci_pattern.regions[0].total_rows),
                mask=mask,
                noise_scaling_maps=noise_scaling_maps_list,
            )

        elif self.is_parallel_and_serial_fit:
            return dataset.parallel_serial_ci_data_masked_from_mask(
                mask=mask, noise_scaling_maps_list=noise_scaling_maps_list
            )

from autocti.charge_injection import ci_mask, ci_imaging
from autocti.pipeline.phase.dataset import meta_dataset


class MetaCIImaging(meta_dataset.MetaDataset):
    def __init__(self, model, settings):

        self.model = model

        if not self.is_parallel_fit:
            settings.columns = None

        if not self.is_serial_fit:
            settings.rows = None

        super().__init__(model=model, settings=settings)

    def mask_for_analysis_from_dataset(self, dataset, mask):

        mask = self.mask_for_analysis_from_cosmic_ray_map(
            cosmic_ray_map=dataset.cosmic_ray_map, mask=mask
        )

        if self.settings.parallel_front_edge_mask_rows is not None:

            parallel_front_edge_mask = ci_mask.CIMask.masked_parallel_front_edge_from_ci_frame(
                ci_frame=dataset.image, rows=self.settings.parallel_front_edge_mask_rows
            )

            mask = mask + parallel_front_edge_mask

        if self.settings.parallel_trails_mask_rows is not None:

            parallel_trails_mask = ci_mask.CIMask.masked_parallel_trails_from_ci_frame(
                ci_frame=dataset.image, rows=self.settings.parallel_trails_mask_rows
            )

            mask = mask + parallel_trails_mask

        if self.settings.serial_front_edge_mask_columns is not None:

            serial_front_edge_mask = ci_mask.CIMask.masked_serial_front_edge_from_ci_frame(
                ci_frame=dataset.image,
                columns=self.settings.serial_front_edge_mask_columns,
            )

            mask = mask + serial_front_edge_mask

        if self.settings.serial_trails_mask_columns is not None:

            serial_trails_mask = ci_mask.CIMask.masked_serial_trails_from_ci_frame(
                ci_frame=dataset.image, columns=self.settings.serial_trails_mask_columns
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

        if len(noise_scaling_maps_list) == 0:
            return total_images * [None]

        return noise_scaling_maps_list

    def masked_ci_dataset_from_dataset(self, dataset, mask, noise_scaling_maps=None):

        return ci_imaging.MaskedCIImaging(
            ci_imaging=dataset,
            mask=mask,
            noise_scaling_maps=noise_scaling_maps,
            parallel_columns=self.settings.columns,
            serial_rows=self.settings.rows,
        )

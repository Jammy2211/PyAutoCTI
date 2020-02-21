class TestCIMaskImaged:
    def test__map(self, ci_pattern_7x7):

        data = ac.ci_imaging(
            image=ac.ci_frame.full(
                fill_value=1.0, shape_2d=(1, 1), ci_pattern=ci_pattern_7x7
            ),
            noise_map=3,
            ci_pre_cti=4,
            cosmic_ray_map=None,
        )

        result = data.map_to_ci_data_masked(func=lambda x: 2 * x, mask=1)
        assert isinstance(result, masked_dataset.MaskedCIImaging)
        assert result.image == 1.0
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.cosmic_ray_map == None

        data = ac.ci_imaging(image=1, noise_map=3, ci_pre_cti=4, cosmic_ray_map=10)
        result = data.map_to_ci_data_masked(func=lambda x: 2 * x, mask=1)
        assert isinstance(result, masked_dataset.MaskedCIImaging)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.cosmic_ray_map == 10

    def test__map_including_noise_scaling_maps(self):

        data = ac.ci_imaging(image=1, noise_map=3, ci_pre_cti=4, cosmic_ray_map=None)
        result = data.map_to_ci_data_masked(
            func=lambda x: 2 * x, mask=1, noise_scaling_maps=[1, 2, 3]
        )
        assert isinstance(result, masked_dataset.MaskedCIImaging)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.noise_scaling_maps == [2, 4, 6]
        assert result.cosmic_ray_map == None

        data = ac.ci_imaging(image=1, noise_map=3, ci_pre_cti=4, cosmic_ray_map=10)
        result = data.map_to_ci_data_masked(
            func=lambda x: 2 * x, mask=1, noise_scaling_maps=[1, 2, 3]
        )
        assert isinstance(result, masked_dataset.MaskedCIImaging)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.noise_scaling_maps == [2, 4, 6]
        assert result.cosmic_ray_map == 10

    def test__parallel_serial_ci_data_fit_from_mask(self):

        data = ac.ci_imaging(image=1, noise_map=3, ci_pre_cti=4, cosmic_ray_map=10)

        def parallel_serial_extractor():
            def extractor(obj):
                return 2 * obj

            return extractor

        data.parallel_serial_ci_imaging = parallel_serial_extractor
        result = data.parallel_serial_ci_data_masked_from_mask(mask=1)

        assert isinstance(result, masked_dataset.MaskedCIImaging)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.cosmic_ray_map == 10

    def test__parallel_serial_ci_data_fit_from_mask__include_noise_scaling_maps(self):
        data = ac.ci_imaging(image=1, noise_map=3, ci_pre_cti=4, cosmic_ray_map=10)

        def parallel_serial_extractor():
            def extractor(obj):
                return 2 * obj

            return extractor

        data.parallel_serial_ci_imaging = parallel_serial_extractor
        result = data.parallel_serial_ci_data_masked_from_mask(
            mask=1, noise_scaling_maps=[2, 3]
        )

        assert isinstance(result, masked_dataset.MaskedCIImaging)
        assert result.image == 2
        assert result.noise_map == 6
        assert result.ci_pre_cti == 8
        assert result.noise_scaling_maps == [4, 6]
        assert result.cosmic_ray_map == 10

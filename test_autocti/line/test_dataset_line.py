import autocti as ac


class TestApplyMask:
    def test__construtor__masks_arrays_correctly(self, dataset_line_7):

        mask = ac.Mask1D.unmasked(
            shape_slim=dataset_line_7.data.shape_slim,
            pixel_scales=dataset_line_7.data.pixel_scales,
        )

        mask[0] = True

        masked_dataset_line = dataset_line_7.apply_mask(mask=mask)

        assert (masked_dataset_line.mask == mask).all()

        masked_data = dataset_line_7.data
        masked_data[0] = 0.0

        assert (masked_dataset_line.data == masked_data).all()

        masked_noise_map = dataset_line_7.noise_map
        masked_noise_map[0] = 0.0

        assert (masked_dataset_line.noise_map == masked_noise_map).all()

        assert (masked_dataset_line.line_pre_cti == dataset_line_7.line_pre_cti).all()

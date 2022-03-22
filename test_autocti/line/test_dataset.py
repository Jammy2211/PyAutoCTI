from astropy.io import fits
import numpy as np
import os
from os import path
import pytest
import shutil

import autocti as ac

fits_path = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "arrays"
)


def create_fits(fits_path, shape_1d=(7,)):

    if path.exists(fits_path):
        shutil.rmtree(fits_path)

    os.makedirs(fits_path)

    hdu_list = fits.HDUList()
    hdu_list.append(fits.ImageHDU(np.ones(shape_1d)))
    hdu_list.writeto(path.join(fits_path, "3_ones.fits"))

    hdu_list = fits.HDUList()
    hdu_list.append(fits.ImageHDU(2.0 * np.ones(shape_1d)))
    hdu_list.writeto(path.join(fits_path, "3_twos.fits"))

    hdu_list = fits.HDUList()
    hdu_list.append(fits.ImageHDU(3.0 * np.ones(shape_1d)))
    hdu_list.writeto(path.join(fits_path, "3_threes.fits"))

    hdu_list = fits.HDUList()
    hdu_list.append(fits.ImageHDU(np.ones(shape_1d)))
    hdu_list.append(fits.ImageHDU(2.0 * np.ones(shape_1d)))
    hdu_list.append(fits.ImageHDU(3.0 * np.ones(shape_1d)))
    hdu_list.writeto(path.join(fits_path, "3_multiple_hdu.fits"))


def clean_fits(fits_path):

    if path.exists(fits_path):
        shutil.rmtree(fits_path)


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

        assert (masked_dataset_line.pre_cti_data == dataset_line_7.pre_cti_data).all()


class TestDataset1D:
    def test__from_fits__load_all_data_components__has_correct_attributes(
        self, layout_7
    ):

        create_fits(fits_path=fits_path)

        imaging = ac.Dataset1D.from_fits(
            pixel_scales=1.0,
            layout=layout_7,
            data_path=path.join(fits_path, "3_ones.fits"),
            data_hdu=0,
            noise_map_path=path.join(fits_path, "3_twos.fits"),
            noise_map_hdu=0,
            pre_cti_data_path=path.join(fits_path, "3_threes.fits"),
            pre_cti_data_hdu=0,
        )

        assert (imaging.data.native == np.ones((7,))).all()
        assert (imaging.noise_map.native == 2.0 * np.ones((7,))).all()
        assert (imaging.pre_cti_data.native == 3.0 * np.ones((7,))).all()

        assert imaging.layout == layout_7

        clean_fits(fits_path=fits_path)

    def test__from_fits__load_all_data_components__load_from_multi_hdu_fits(
        self, layout_7
    ):

        create_fits(fits_path=fits_path)

        imaging = ac.Dataset1D.from_fits(
            pixel_scales=1.0,
            layout=layout_7,
            data_path=path.join(fits_path, "3_multiple_hdu.fits"),
            data_hdu=0,
            noise_map_path=path.join(fits_path, "3_multiple_hdu.fits"),
            noise_map_hdu=1,
            pre_cti_data_path=path.join(fits_path, "3_multiple_hdu.fits"),
            pre_cti_data_hdu=2,
        )

        assert (imaging.data.native == np.ones((7,))).all()
        assert (imaging.noise_map.native == 2.0 * np.ones((7,))).all()
        assert (imaging.pre_cti_data.native == 3.0 * np.ones((7,))).all()

        assert imaging.layout == layout_7

        clean_fits(fits_path=fits_path)

    def test__from_fits__noise_map_from_single_value(self, layout_7):

        create_fits(fits_path=fits_path)

        imaging = ac.Dataset1D.from_fits(
            pixel_scales=1.0,
            layout=layout_7,
            data_path=path.join(fits_path, "3_ones.fits"),
            data_hdu=0,
            noise_map_from_single_value=10.0,
            pre_cti_data_path=path.join(fits_path, "3_threes.fits"),
            pre_cti_data_hdu=0,
        )

        assert (imaging.data.native == np.ones((7,))).all()
        assert (imaging.noise_map.native == 10.0 * np.ones((7,))).all()
        assert (imaging.pre_cti_data.native == 3.0 * np.ones((7,))).all()

        assert imaging.layout == layout_7

        clean_fits(fits_path=fits_path)

    def test__from_fits__load_pre_cti_data_data_from_the_layout_ci_and_data(self):

        create_fits(fits_path=fits_path)

        layout_ci = ac.Layout1D(shape_1d=(7,), region_list=[(0, 7)])

        imaging = ac.Dataset1D.from_fits(
            pixel_scales=1.0,
            layout=layout_ci,
            data_path=path.join(fits_path, "3_ones.fits"),
            data_hdu=0,
            noise_map_path=path.join(fits_path, "3_twos.fits"),
            noise_map_hdu=0,
            pre_cti_data_path=path.join(fits_path, "3_threes.fits"),
            pre_cti_data_hdu=0,
        )

        assert (imaging.data.native == np.ones((7,))).all()
        assert (imaging.noise_map.native == 2.0 * np.ones((7,))).all()
        assert (imaging.pre_cti_data.native == 3.0 * np.ones((7,))).all()

        assert imaging.layout == layout_ci

        clean_fits(fits_path=fits_path)

    def test__output_to_fits___all_arrays(self, layout_7):

        create_fits(fits_path=fits_path)

        imaging = ac.Dataset1D.from_fits(
            pixel_scales=1.0,
            layout=layout_7,
            data_path=path.join(fits_path, "3_ones.fits"),
            data_hdu=0,
            noise_map_path=path.join(fits_path, "3_twos.fits"),
            noise_map_hdu=0,
            pre_cti_data_path=path.join(fits_path, "3_threes.fits"),
            pre_cti_data_hdu=0,
        )

        output_data_dir = path.join(fits_path, "output_test")

        if path.exists(output_data_dir):
            shutil.rmtree(output_data_dir)

        os.makedirs(output_data_dir)

        imaging.output_to_fits(
            data_path=path.join(output_data_dir, "data.fits"),
            noise_map_path=path.join(output_data_dir, "noise_map.fits"),
            pre_cti_data_path=path.join(output_data_dir, "pre_cti_data.fits"),
        )

        imaging = ac.Dataset1D.from_fits(
            pixel_scales=1.0,
            layout=layout_7,
            data_path=path.join(output_data_dir, "data.fits"),
            data_hdu=0,
            noise_map_path=path.join(output_data_dir, "noise_map.fits"),
            noise_map_hdu=0,
            pre_cti_data_path=path.join(output_data_dir, "pre_cti_data.fits"),
            pre_cti_data_hdu=0,
        )

        assert (imaging.data.native == np.ones((7,))).all()
        assert (imaging.noise_map.native == 2.0 * np.ones((7,))).all()
        assert (imaging.pre_cti_data.native == 3.0 * np.ones((7,))).all()

        clean_fits(fits_path=fits_path)


class TestSimulatorDataset1D:
    def test__no_instrumental_effects_input__only_cti_simulated(
        self, clocker_1d, traps_x2, ccd
    ):

        layout = ac.Layout1D(shape_1d=(5,), region_list=[(0, 5)])

        simulator = ac.SimulatorDataset1D(
            pixel_scales=1.0, normalization=10.0, add_poisson_noise=False
        )

        dataset_line = simulator.from_layout(
            layout=layout, clocker=clocker_1d, trap_list=traps_x2, ccd=ccd
        )

        assert dataset_line.data == pytest.approx(
            np.array([9.43, 9.43, 9.43, 9.43, 9.43]), 1e-1
        )
        assert dataset_line.layout == layout

    def test__include_read_noise__is_added_after_cti(self, clocker_1d, traps_x2, ccd):

        layout = ac.Layout1D(shape_1d=(5,), region_list=[(0, 5)])

        simulator = ac.SimulatorDataset1D(
            pixel_scales=1.0,
            normalization=10.0,
            read_noise=1.0,
            add_poisson_noise=False,
            noise_seed=1,
        )

        dataset_line = simulator.from_layout(
            layout=layout, clocker=clocker_1d, trap_list=traps_x2, ccd=ccd
        )

        # Use seed to give us a known read noises map we'll test_autocti for

        assert dataset_line.data == pytest.approx(
            np.array([11.05513, 9.36790, 9.47129, 8.92700, 10.8654]), 1e-2
        )
        assert dataset_line.layout == layout

    def test__pre_cti_data_from(self):

        simulator = ac.SimulatorDataset1D(normalization=10.0, pixel_scales=1.0)

        layout = ac.Layout1D(shape_1d=(3,), region_list=[(0, 2)])

        pre_cti_data = simulator.pre_cti_data_from(layout=layout, pixel_scales=1.0)

        print(pre_cti_data)

        assert (pre_cti_data.native == np.array([10.0, 10.0, 0.0])).all()

    def test__from_pre_cti_data(self, clocker_1d, traps_x2, ccd):

        layout = ac.Layout1D(shape_1d=(5,), region_list=[(0, 5)])

        simulator = ac.SimulatorDataset1D(
            pixel_scales=1.0,
            normalization=10.0,
            read_noise=4.0,
            add_poisson_noise=False,
            noise_seed=1,
        )

        dataset_line = simulator.from_layout(
            layout=layout, clocker=clocker_1d, trap_list=traps_x2, ccd=ccd
        )

        pre_cti_data = simulator.pre_cti_data_from(layout=layout, pixel_scales=1.0)

        dataset_line_via_pre_cti_data = simulator.from_pre_cti_data(
            pre_cti_data=pre_cti_data.native,
            layout=layout,
            clocker=clocker_1d,
            trap_list=traps_x2,
            ccd=ccd,
        )

        assert (dataset_line.data == dataset_line_via_pre_cti_data.data).all()
        assert (dataset_line.noise_map == dataset_line_via_pre_cti_data.noise_map).all()
        assert (
            dataset_line.pre_cti_data == dataset_line_via_pre_cti_data.pre_cti_data
        ).all()

    def test__from_post_cti_data(self, clocker_1d, traps_x2, ccd):

        layout = ac.Layout1D(shape_1d=(5,), region_list=[(0, 5)])
        simulator = ac.SimulatorDataset1D(
            pixel_scales=1.0,
            normalization=10.0,
            read_noise=4.0,
            add_poisson_noise=False,
            noise_seed=1,
        )

        dataset_line = simulator.from_layout(
            layout=layout, clocker=clocker_1d, trap_list=traps_x2, ccd=ccd
        )

        pre_cti_data = simulator.pre_cti_data_from(
            layout=layout, pixel_scales=1.0
        ).native

        post_cti_data = clocker_1d.add_cti(
            data=pre_cti_data, trap_list=traps_x2, ccd=ccd
        )

        dataset_line_via_post_cti_data = simulator.from_post_cti_data(
            post_cti_data=post_cti_data, pre_cti_data=pre_cti_data.native, layout=layout
        )

        assert (dataset_line.data == dataset_line_via_post_cti_data.data).all()
        assert (
            dataset_line.noise_map == dataset_line_via_post_cti_data.noise_map
        ).all()
        assert (
            dataset_line.pre_cti_data == dataset_line_via_post_cti_data.pre_cti_data
        ).all()

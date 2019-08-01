import os
import shutil

import numpy as np
import pytest
from astropy.io import fits

from autocti.model import arctic_settings


@pytest.fixture(name="hdr_path")
def test_header_info():
    hdr_path = "{}/../test_files/cti_settings/header_info/".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    if os.path.exists(hdr_path):
        shutil.rmtree(hdr_path)

    os.mkdir(hdr_path)

    return hdr_path


class TestArcticSettings:
    class TestConstructor:
        def test__sets_up_parameters_with_correct_values(self):

            parallel_settings = arctic_settings.Settings(
                well_depth=84700,
                niter=1,
                express=5,
                n_levels=2000,
                charge_injection_mode=True,
                readout_offset=0,
            )

            serial_settings = arctic_settings.Settings(
                well_depth=84700, niter=1, express=5, n_levels=2000, readout_offset=0
            )

            arctic_both = arctic_settings.ArcticSettings(
                neomode="NEO", parallel=parallel_settings, serial=serial_settings
            )

            assert arctic_both.neomode == "NEO"

            assert arctic_both.parallel.well_depth == 84700
            assert arctic_both.parallel.niter == 1
            assert arctic_both.parallel.express == 5
            assert arctic_both.parallel.n_levels == 2000
            assert arctic_both.parallel.charge_injection_mode is True
            assert arctic_both.parallel.readout_offset == 0

            assert arctic_both.serial.well_depth == 84700
            assert arctic_both.serial.niter == 1
            assert arctic_both.serial.express == 5
            assert arctic_both.serial.n_levels == 2000
            assert arctic_both.serial.charge_injection_mode is False
            assert arctic_both.serial.readout_offset == 0

    class TestFitsHeaderInfo:
        def test__parallel_only__sets_up_header_info_consistent_with_previous_vis_pf(
            self, hdr_path
        ):
            parallel_settings = arctic_settings.Settings(
                well_depth=84700,
                niter=1,
                express=5,
                n_levels=2000,
                charge_injection_mode=False,
                readout_offset=0,
            )

            arctic_parallel = arctic_settings.ArcticSettings(
                neomode="NEO", parallel=parallel_settings
            )

            hdu = fits.PrimaryHDU(np.ones((1, 1)), fits.Header())
            hdu.header = arctic_parallel.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + "/test.fits")

            hdu = fits.open(hdr_path + "/test.fits")
            ext_header = hdu[0].header

            assert ext_header["cte_pite"] == 1
            assert ext_header["cte_pwld"] == 84700
            assert ext_header["cte_pnts"] == 2000

        def test__serial_only__sets_up_header_info_consistent_with_previous_vis_pf(
            self, hdr_path
        ):
            serial_settings = arctic_settings.Settings(
                well_depth=84700,
                niter=1,
                express=5,
                n_levels=2000,
                charge_injection_mode=False,
                readout_offset=0,
            )

            arctic_serial = arctic_settings.ArcticSettings(
                neomode="NEO", serial=serial_settings
            )

            hdu = fits.PrimaryHDU(np.ones((1, 1)), fits.Header())
            hdu.header = arctic_serial.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + "/test.fits")

            hdu = fits.open(hdr_path + "/test.fits")
            ext_header = hdu[0].header

            assert ext_header["cte_site"] == 1
            assert ext_header["cte_swld"] == 84700
            assert ext_header["cte_snts"] == 2000

        def test__parallel_and_serial__sets_up_header_info_consistent_with_previous_vis_pf(
            self, hdr_path
        ):
            parallel_settings = arctic_settings.Settings(
                well_depth=84700,
                niter=1,
                express=5,
                n_levels=2000,
                charge_injection_mode=False,
                readout_offset=0,
            )

            serial_settings = arctic_settings.Settings(
                well_depth=84700,
                niter=1,
                express=5,
                n_levels=2000,
                charge_injection_mode=False,
                readout_offset=0,
            )

            arctic_both = arctic_settings.ArcticSettings(
                neomode="NEO", parallel=parallel_settings, serial=serial_settings
            )

            hdu = fits.PrimaryHDU(np.ones((1, 1)), fits.Header())
            hdu.header = arctic_both.update_fits_header_info(ext_header=hdu.header)
            hdu.writeto(hdr_path + "/test.fits")

            hdu = fits.open(hdr_path + "/test.fits")
            ext_header = hdu[0].header

            assert ext_header["cte_pite"] == 1
            assert ext_header["cte_pwld"] == 84700
            assert ext_header["cte_pnts"] == 2000

            assert ext_header["cte_site"] == 1
            assert ext_header["cte_swld"] == 84700
            assert ext_header["cte_snts"] == 2000

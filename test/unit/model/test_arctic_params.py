import os
import shutil

import pytest

from autocti.model import arctic_params


class TestParams:
    def test__1_species__sets_values_correctly(self):

        parallel_1_species = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)

        parameters = arctic_params.ArcticParams(parallel_species=parallel_1_species)

        assert type(parameters) == arctic_params.ArcticParams
        assert type(parameters.parallel_species) == arctic_params.Species

        assert parameters.parallel_species.trap_density == 0.1
        assert parameters.parallel_species.trap_lifetime == 1.0

    def test__parallel_and_serial_species__sets_value_correctly(self):

        parallel_species_0 = arctic_params.Species(trap_density=0.1, trap_lifetime=1.0)
        parallel_species_1 = arctic_params.Species(trap_density=0.2, trap_lifetime=2.0)

        parallel_ccd = arctic_params.CCD(
            well_notch_depth=0.01,
            well_fill_alpha=0.2,
            well_fill_beta=0.8,
            well_fill_gamma=2.0,
        )

        serial_species_0 = arctic_params.Species(trap_density=0.3, trap_lifetime=3.0)
        serial_species_1 = arctic_params.Species(trap_density=0.4, trap_lifetime=4.0)

        serial_ccd = arctic_params.CCD(
            well_notch_depth=1.02,
            well_fill_alpha=1.1,
            well_fill_beta=1.4,
            well_fill_gamma=1.6,
        )

        parameters = arctic_params.ArcticParams(
            parallel_species=[parallel_species_0, parallel_species_1],
            serial_species=[serial_species_0, serial_species_1],
            parallel_ccd=parallel_ccd,
            serial_ccd=serial_ccd,
        )

        assert type(parameters) == arctic_params.ArcticParams
        assert type(parameters.parallel_species[0]) == arctic_params.Species
        assert type(parameters.parallel_species[1]) == arctic_params.Species
        assert type(parameters.parallel_ccd) == arctic_params.CCD

        assert parameters.parallel_species[0].trap_density == 0.1
        assert parameters.parallel_species[0].trap_lifetime == 1.0
        assert parameters.parallel_species[1].trap_density == 0.2
        assert parameters.parallel_species[1].trap_lifetime == 2.0
        assert parameters.parallel_ccd.well_notch_depth == 0.01
        assert parameters.parallel_ccd.well_fill_alpha == 0.2
        assert parameters.parallel_ccd.well_fill_beta == 0.8
        assert parameters.parallel_ccd.well_fill_gamma == 2.0

        assert type(parameters) == arctic_params.ArcticParams
        assert type(parameters.serial_species[0]) == arctic_params.Species
        assert type(parameters.serial_species[1]) == arctic_params.Species
        assert type(parameters.serial_ccd) == arctic_params.CCD

        assert parameters.serial_species[0].trap_density == 0.3
        assert parameters.serial_species[0].trap_lifetime == 3.0
        assert parameters.serial_species[1].trap_density == 0.4
        assert parameters.serial_species[1].trap_lifetime == 4.0
        assert parameters.serial_ccd.well_notch_depth == 1.02
        assert parameters.serial_ccd.well_fill_alpha == 1.1
        assert parameters.serial_ccd.well_fill_beta == 1.4
        assert parameters.serial_ccd.well_fill_gamma == 1.6

    def test__delta_ellpiticity_of_species(self):

        species = arctic_params.Species(trap_density=0.5, trap_lifetime=2.0)

        assert species.delta_ellipticity == pytest.approx(0.047378295117617694, 1.0e-5)

    def test__delta_ellipticity_of_arctic_params(self):

        parallel_1_species = arctic_params.Species(trap_density=0.1, trap_lifetime=4.0)
        parallel_2_species = arctic_params.Species(trap_density=0.1, trap_lifetime=4.0)
        serial_1_species = arctic_params.Species(trap_density=0.2, trap_lifetime=2.0)
        serial_2_species = arctic_params.Species(trap_density=0.7, trap_lifetime=7.0)

        parameters = arctic_params.ArcticParams(parallel_species=[parallel_1_species])
        assert parameters.delta_ellipticity == parallel_1_species.delta_ellipticity

        parameters = arctic_params.ArcticParams(
            parallel_species=[parallel_1_species, parallel_2_species]
        )
        assert (
            parameters.delta_ellipticity
            == parallel_1_species.delta_ellipticity
            + parallel_2_species.delta_ellipticity
        )

        parameters = arctic_params.ArcticParams(serial_species=[serial_1_species])
        assert parameters.delta_ellipticity == serial_1_species.delta_ellipticity

        parameters = arctic_params.ArcticParams(
            serial_species=[serial_1_species, serial_2_species]
        )
        assert (
            parameters.delta_ellipticity
            == serial_1_species.delta_ellipticity + serial_2_species.delta_ellipticity
        )

        parameters = arctic_params.ArcticParams(
            parallel_species=[parallel_1_species, parallel_2_species],
            serial_species=[serial_1_species, serial_2_species],
        )

        assert parameters.delta_ellipticity == pytest.approx(
            parallel_1_species.delta_ellipticity
            + parallel_2_species.delta_ellipticity
            + serial_1_species.delta_ellipticity
            + serial_2_species.delta_ellipticity,
            1.0e-6,
        )


class TestParallelDensityVary:
    def test_1_species__density_01__1000_column_pixels__1_row_pixel_so_100_traps__posison_density_near_01(
        self
    ):  #

        parallel_vary = arctic_params.Species.poisson_species(
            species=list(
                map(
                    lambda density: arctic_params.Species(
                        trap_density=density, trap_lifetime=1.0
                    ),
                    (0.1,),
                )
            ),
            shape=(1000, 1),
            seed=1,
        )

        assert [species.trap_density for species in parallel_vary] == [0.098]

    def test__1_species__density_1__1000_column_pixels_so_1000_traps__1_row_pixel__posison_value_is_near_1(
        self
    ):
        parallel_vary = arctic_params.Species.poisson_species(
            species=list(
                map(
                    lambda density: arctic_params.Species(
                        trap_density=density, trap_lifetime=1.0
                    ),
                    (1.0,),
                )
            ),
            shape=(1000, 1),
            seed=1,
        )

        assert [species.trap_density for species in parallel_vary] == [0.992]

    def test__1_species__density_1___2_row_pixels__posison_value_is_near_1(self):
        parallel_vary = arctic_params.Species.poisson_species(
            species=list(
                map(
                    lambda density: arctic_params.Species(
                        trap_density=density, trap_lifetime=1.0
                    ),
                    (1.0,),
                )
            ),
            shape=(1000, 2),
            seed=1,
        )

        assert [species.trap_density for species in parallel_vary] == [0.992, 0.962]

    def test__2_species__1_row_pixel__poisson_for_each_species_drawn(self):
        parallel_vary = arctic_params.Species.poisson_species(
            species=list(
                map(
                    lambda density: arctic_params.Species(
                        trap_density=density, trap_lifetime=1.0
                    ),
                    (1.0, 2.0),
                )
            ),
            shape=(1000, 1),
            seed=1,
        )

        assert [species.trap_density for species in parallel_vary] == [0.992, 1.946]

    def test__2_species__2_row_pixel__poisson_for_each_species_drawn(self):
        parallel_vary = arctic_params.Species.poisson_species(
            species=list(
                map(
                    lambda density: arctic_params.Species(
                        trap_density=density, trap_lifetime=1.0
                    ),
                    (1.0, 2.0),
                )
            ),
            shape=(1000, 2),
            seed=1,
        )

        assert [species.trap_density for species in parallel_vary] == [
            0.992,
            1.946,
            0.968,
            1.987,
        ]

    def test__same_as_above_but_3_species_and_new_values(self):
        parallel_vary = arctic_params.Species.poisson_species(
            species=list(
                map(
                    lambda density: arctic_params.Species(
                        trap_density=density, trap_lifetime=1.0
                    ),
                    (1.0, 2.0, 0.1),
                )
            ),
            shape=(1000, 3),
            seed=1,
        )

        assert [species.trap_density for species in parallel_vary] == [
            0.992,
            1.946,
            0.09,
            0.991,
            1.99,
            0.098,
            0.961,
            1.975,
            0.113,
        ]

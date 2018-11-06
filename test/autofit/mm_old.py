#
# Copyright (C) 2012-2020 Euclid Science Ground Segment
#
# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
File: tests/python/ModelMapper_test.py

Created on: 02/14/18
Author: James Nightingale
"""

from __future__ import division, print_function
import sys
import os

if sys.version_info[0] < 3:
    from future_builtins import *

import pytest
from VIS_CTI_Calibrate import ModelMapper as mm
from VIS_CTI_Config import Config as conf
from VIS_CTI_PyArCTIC import ArcticParams as arctic_params


@pytest.fixture(name='uniform_simple')
def make_uniform_simple():
    return mm.UniformPrior(lower_limit=0., upper_limit=1.)


@pytest.fixture(name='uniform_half')
def make_uniform_half():
    return mm.UniformPrior(lower_limit=0.5, upper_limit=1.)


@pytest.fixture(name='test_config')
def make_test_config():
    return conf.DefaultPriorConfig(config_folder_path="{}/files/config/priors/default".
                                   format(os.path.dirname(os.path.realpath(__file__))))


@pytest.fixture(name="width_config")
def make_width_config():
    return conf.WidthConfig(
        config_folder_path="{}/files/config/priors/width".format(os.path.dirname(os.path.realpath(__file__))))


class TestAddition(object):
    def test_abstract_plus_abstract(self):
        one = mm.AbstractModel()
        two = mm.AbstractModel()
        one.a = 'a'
        two.b = 'b'

        three = one + two

        assert three.a == 'a'
        assert three.b == 'b'

    def test_instance_plus_instance(self):
        one = mm.ModelInstance()
        two = mm.ModelInstance()
        one.a = 'a'
        two.b = 'b'

        three = one + two

        assert three.a == 'a'
        assert three.b == 'b'

    def test_mapper_plus_mapper(self, test_config):
        one = mm.ModelMapper()
        two = mm.ModelMapper()
        one.a = mm.PriorModel(arctic_params.ParallelOneSpecies, config=test_config)
        two.b = mm.PriorModel(arctic_params.ParallelThreeSpecies, config=test_config)

        three = one + two

        assert three.total_parameters == 16


class TestUniformPrior(object):

    def test__simple_assumptions(self, uniform_simple):
        assert uniform_simple.value_for(0.) == 0.
        assert uniform_simple.value_for(1.) == 1.
        assert uniform_simple.value_for(0.5) == 0.5

    def test__non_zero_lower_limit(self, uniform_half):
        assert uniform_half.value_for(0.) == 0.5
        assert uniform_half.value_for(1.) == 1.
        assert uniform_half.value_for(0.5) == 0.75


class MockClassMM(object):
    def __init__(self, one, two):
        self.one = one
        self.two = two


class MockConfig(conf.DefaultPriorConfig):

    def __init__(self, d=None):
        super(MockConfig, self).__init__("")
        if d is not None:
            self.d = d
        else:
            self.d = {}

    def get_for_nearest_ancestor(self, cls, attribute_name):
        return self.get(None, cls.__name__, attribute_name)

    def get(self, _, class_name, var_name):
        try:
            return self.d[class_name][var_name]
        except KeyError:
            return ["u", 0, 1]


class MockSpecies(object):
    def __init__(self, trap_densities=(1.0, 2.0), well_fill_beta=0.8):
        self.trap_densities = trap_densities
        self.well_fill_beta = well_fill_beta


class TestModelingCollection(object):

    def test__argument_extraction(self):
        collection = mm.ModelMapper(MockConfig())
        collection.mock_class = MockClassMM
        assert 1 == len(collection.prior_models)

        assert len(collection.priors_ordered_by_id) == 2

    def test__config_limits(self):
        collection = mm.ModelMapper(MockConfig({"MockClassMM": {"one": ["u", 1., 2.]}}))

        collection.mock_class = MockClassMM

        assert collection.mock_class.one.lower_limit == 1.
        assert collection.mock_class.one.upper_limit == 2.

    def test__config_prior_type(self):
        collection = mm.ModelMapper(MockConfig({"MockClassMM": {"one": ["g", 1., 2.]}}))

        collection.mock_class = MockClassMM

        assert isinstance(collection.mock_class.one, mm.GaussianPrior)

        assert collection.mock_class.one.mean == 1.
        assert collection.mock_class.one.sigma == 2.

    def test__attribution(self):
        collection = mm.ModelMapper(MockConfig())

        collection.mock_class = MockClassMM

        assert hasattr(collection, "mock_class")
        assert hasattr(collection.mock_class, "one")

    def test__tuple_arg(self):
        collection = mm.ModelMapper(MockConfig())

        collection.mock_class = MockSpecies

        assert 3 == len(collection.priors_ordered_by_id)


class TestModelInstance(object):

    def test_instances_of(self):
        instance = mm.ModelInstance()
        instance.mock_1 = MockClassMM(one=1, two=2)
        instance.mock_2 = MockClassMM(one=2, two=2)
        assert instance.instances_of(MockClassMM) == [instance.mock_1, instance.mock_2]

    def test_instances_of_filtering(self):
        instance = mm.ModelInstance()
        instance.mock_1 = MockClassMM(one=1, two=2)
        instance.mock_2 = MockClassMM(one=2, two=2)
        instance.other = arctic_params.ParallelOneSpecies()
        assert instance.instances_of(MockClassMM) == [instance.mock_1, instance.mock_2]

    def test_instances_from_list(self):
        instance = mm.ModelInstance()
        mock_1 = MockClassMM(one=1, two=2)
        mock_2 = MockClassMM(one=2, two=2)
        instance.mocks = [mock_1, mock_2]
        assert instance.instances_of(MockClassMM) == [mock_1, mock_2]

    def test_non_trivial_instances_of(self):
        instance = mm.ModelInstance()
        mock_1 = MockClassMM(one=1, two=2)
        mock_2 = MockClassMM(one=2, two=2)
        instance.mocks = [mock_1, mock_2, arctic_params.ParallelOneSpecies]
        instance.mock_3 = MockClassMM(one=3, two=3)
        instance.parallel = arctic_params.ParallelTwoSpecies()

        assert instance.instances_of(MockClassMM) == [instance.mock_3, mock_1, mock_2]

    def test__simple_model(self):
        collection = mm.ModelMapper(MockConfig())

        collection.mock_class = MockClassMM

        model_map = collection.instance_from_unit_vector([1., 1.])

        assert isinstance(model_map.mock_class, MockClassMM)
        assert model_map.mock_class.one == 1.
        assert model_map.mock_class.two == 1.

    def test__two_object_model(self):
        collection = mm.ModelMapper(MockConfig())

        collection.mock_class_1 = MockClassMM
        collection.mock_class_2 = MockClassMM

        model_map = collection.instance_from_unit_vector([1., 0., 0., 1.])

        assert isinstance(model_map.mock_class_1, MockClassMM)
        assert isinstance(model_map.mock_class_2, MockClassMM)

        assert model_map.mock_class_1.one == 1.
        assert model_map.mock_class_1.two == 0.

        assert model_map.mock_class_2.one == 0.
        assert model_map.mock_class_2.two == 1.

    def test__swapped_prior_construction(self):
        collection = mm.ModelMapper(MockConfig())

        collection.mock_class_1 = MockClassMM
        collection.mock_class_2 = MockClassMM

        collection.mock_class_2.one = collection.mock_class_1.one

        model_map = collection.instance_from_unit_vector([1., 0., 0.])

        assert isinstance(model_map.mock_class_1, MockClassMM)
        assert isinstance(model_map.mock_class_2, MockClassMM)

        assert model_map.mock_class_1.one == 1.
        assert model_map.mock_class_1.two == 0.

        assert model_map.mock_class_2.one == 1.
        assert model_map.mock_class_2.two == 0.

    def test__prior_replacement(self):
        collection = mm.ModelMapper(MockConfig())

        collection.mock_class = MockClassMM

        collection.mock_class.one = mm.UniformPrior(100, 200)

        model_map = collection.instance_from_unit_vector([0., 0.])

        assert model_map.mock_class.one == 100.

    def test__tuple_arg(self):
        collection = mm.ModelMapper(MockConfig())

        collection.mock_species = MockSpecies

        model_map = collection.instance_from_unit_vector([1., 0., 0.])

        assert model_map.mock_species.trap_densities == (1., 0.)
        assert model_map.mock_species.well_fill_beta == 0.

    def test__modify_tuple(self):
        collection = mm.ModelMapper(MockConfig())

        collection.mock_species = MockSpecies

        collection.mock_species.trap_densities.trap_densities_0 = mm.UniformPrior(1., 10.)

        model_map = collection.instance_from_unit_vector([1., 1., 1.])

        assert model_map.mock_species.trap_densities == (10., 1.)
        assert model_map.mock_species.well_fill_beta == 1.0

    def test__match_tuple(self):
        collection = mm.ModelMapper(MockConfig())

        collection.mock_species = MockSpecies

        collection.mock_species.trap_densities.trap_densities_1 = \
            collection.mock_species.trap_densities.trap_densities_0

        model_map = collection.instance_from_unit_vector([1., 1.])

        assert model_map.mock_species.trap_densities == (1., 1.)
        assert model_map.mock_species.well_fill_beta == 1.0


class TestRealClasses(object):

    def test_combination(self):
        collection = mm.ModelMapper(MockConfig(), parallel_traps=arctic_params.ParallelOneSpecies,
                                    serial_traps=arctic_params.SerialThreeSpecies)

        model = collection.instance_from_unit_vector([1 for _ in range(len(collection.priors_ordered_by_id))])

        assert isinstance(model.parallel_traps, arctic_params.ParallelOneSpecies)
        assert isinstance(model.serial_traps, arctic_params.SerialThreeSpecies)

    def test_attribute(self):
        collection = mm.ModelMapper(MockConfig())
        collection.cls_1 = MockClassMM

        assert 1 == len(collection.prior_models)
        assert isinstance(collection.cls_1, mm.PriorModel)


class TestConfigFunctions:

    def test_loading_config(self, test_config):
        config = test_config

        assert ['u', 0.0, 1.0] == config.get("ArcticParams", "ParallelOneSpecies", "trap_densities_0")
        assert ['u', 0.0, 2.0] == config.get("ArcticParams", "ParallelOneSpecies", "trap_lifetimes_0")
        assert ['u', 0.0, 1.0] == config.get("ArcticParams", "ParallelOneSpecies", "well_notch_depth")
        assert ['u', 0.0, 1.0] == config.get("ArcticParams", "ParallelOneSpecies", "well_fill_beta")

        assert ['u', 0.0, 1.0] == config.get("ArcticParams", "SerialTwoSpecies", "trap_densities_0")
        assert ['u', 0.0, 2.0] == config.get("ArcticParams", "SerialTwoSpecies", "trap_densities_1")
        assert ['u', 0.0, 1.0] == config.get("ArcticParams", "SerialTwoSpecies", "trap_lifetimes_0")
        assert ['u', 0.0, 2.0] == config.get("ArcticParams", "SerialTwoSpecies", "trap_lifetimes_1")
        assert ['g', 0.0, 1.0] == config.get("ArcticParams", "SerialTwoSpecies", "well_notch_depth")
        assert ['u', 0.0, 1.0] == config.get("ArcticParams", "SerialTwoSpecies", "well_fill_beta")

    def test_true_config(self, test_config):
        collection = mm.ModelMapper(test_config,
                                    parallel=arctic_params.ParallelOneSpecies, serial=arctic_params.SerialTwoSpecies)

        model_map = collection.instance_from_unit_vector(
            [1 for _ in range(len(collection.priors_ordered_by_id))])

        assert isinstance(model_map.parallel, arctic_params.ParallelOneSpecies)
        assert isinstance(model_map.serial, arctic_params.SerialTwoSpecies)


class TestClassMappingCollection(object):

    def test__argument_extraction(self):
        collection = mm.ModelMapper(MockConfig())
        collection.mock_class = MockClassMM
        assert 1 == len(collection.prior_models)

        assert len(collection.priors_ordered_by_id) == 2

    def test_config_limits(self):
        collection = mm.ModelMapper(MockConfig({"MockClassMM": {"one": ["u", 1., 2.]}}))

        collection.mock_class = MockClassMM

        assert collection.mock_class.one.lower_limit == 1.
        assert collection.mock_class.one.upper_limit == 2.

    def test_config_prior_type(self):
        collection = mm.ModelMapper(MockConfig({"MockClassMM": {"one": ["g", 1., 2.]}}))

        collection.mock_class = MockClassMM

        assert isinstance(collection.mock_class.one, mm.GaussianPrior)

        assert collection.mock_class.one.mean == 1.
        assert collection.mock_class.one.sigma == 2.

    def test_attribution(self):
        collection = mm.ModelMapper(MockConfig())

        collection.mock_class = MockClassMM

        assert hasattr(collection, "mock_class")
        assert hasattr(collection.mock_class, "one")

    def test_tuple_arg(self):
        collection = mm.ModelMapper(MockConfig())
        collection.mock_species = MockSpecies

        assert 3 == len(collection.priors_ordered_by_id)


class TestModelInstancesRealClasses(object):

    def test__in_order_of_class_constructor__one_species(self, test_config):
        collection = mm.ModelMapper(test_config, species_1=arctic_params.ParallelOneSpecies)

        model_map = collection.instance_from_unit_vector([0.25, 0.5, 0.75, 0.8, 0.9, 1.0])

        assert model_map.species_1.trap_densities == (0.25,)
        assert model_map.species_1.trap_lifetimes == (1.0,)
        assert model_map.species_1.well_notch_depth == 0.75
        assert model_map.species_1.well_fill_alpha == 0.8
        assert model_map.species_1.well_fill_beta == 0.9
        assert model_map.species_1.well_fill_gamma == 1.0

    def test__in_order_of_class_constructor___multiple_species(self, test_config):
        collection = mm.ModelMapper(test_config, parallel=arctic_params.ParallelOneSpecies,
                                    serial=arctic_params.SerialTwoSpecies)

        hypercube_vector = [0.5] * 14

        physical_vector = collection.physical_vector_from_hypercube_vector(hypercube_vector)
        model = collection.instance_from_unit_vector(hypercube_vector)

        assert model.parallel.trap_densities == (physical_vector[0],) == (0.5,)
        assert model.parallel.trap_lifetimes == (physical_vector[1],) == (1.0,)
        assert model.parallel.well_notch_depth == physical_vector[2] == 0.5
        assert model.parallel.well_fill_alpha == physical_vector[3] == 0.5
        assert model.parallel.well_fill_beta == physical_vector[4] == 0.5
        assert model.parallel.well_fill_gamma == physical_vector[5] == 0.5

        assert model.serial.trap_densities == (physical_vector[6], physical_vector[7]) == (0.5, 1.0)
        assert model.serial.trap_lifetimes == (physical_vector[8], physical_vector[9]) == (0.5, 1.0)
        assert model.serial.well_notch_depth == physical_vector[10] == 0.0
        assert model.serial.well_fill_alpha == physical_vector[11] == 0.5
        assert model.serial.well_fill_beta == physical_vector[12] == 0.5
        assert model.serial.well_fill_gamma == physical_vector[13] == 0.5

    def test__check_order_for_different_unit_values(self, test_config):
        collection = mm.ModelMapper(test_config, serial=arctic_params.SerialTwoSpecies)

        collection.serial.trap_densities.trap_densities_0 = mm.UniformPrior(0.0, 1.0)
        collection.serial.trap_densities.trap_densities_1 = mm.UniformPrior(0.0, 2.0)
        collection.serial.trap_lifetimes.trap_lifetimes_0 = mm.UniformPrior(0.0, 1.0)
        collection.serial.trap_lifetimes.trap_lifetimes_1 = mm.UniformPrior(0.0, 1.0)
        collection.serial.well_notch_depth = mm.UniformPrior(0.0, 1.0)
        collection.serial.well_fill_alpha = mm.UniformPrior(0.0, 1.0)
        collection.serial.well_fill_beta = mm.UniformPrior(0.0, 1.0)
        collection.serial.well_fill_gamma = mm.UniformPrior(0.0, 1.0)

        unit_vector = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        hypercube_vector = collection.physical_vector_from_hypercube_vector(unit_vector)
        model = collection.instance_from_unit_vector(unit_vector)

        assert model.serial.trap_densities == (hypercube_vector[0], hypercube_vector[1]) == (0.0, 0.2)
        assert model.serial.trap_lifetimes == (hypercube_vector[2], hypercube_vector[3]) == (0.2, 0.3)
        assert model.serial.well_notch_depth == hypercube_vector[4] == 0.4
        assert model.serial.well_fill_alpha == hypercube_vector[5] == 0.5
        assert model.serial.well_fill_beta == hypercube_vector[6] == 0.6
        assert model.serial.well_fill_gamma == hypercube_vector[7] == 0.7

    def test__check_order_for_different_unit_values_and_set_priors_equal_to_one_another(self, test_config):
        collection = mm.ModelMapper(test_config, serial=arctic_params.SerialTwoSpecies)

        collection.serial.trap_densities.trap_densities_0 = mm.UniformPrior(0.0, 1.0)
        collection.serial.trap_densities.trap_densities_1 = mm.UniformPrior(0.0, 5.0)
        collection.serial.trap_lifetimes.trap_lifetimes_0 = mm.UniformPrior(0.0, 1.0)
        collection.serial.trap_lifetimes.trap_lifetimes_1 = mm.UniformPrior(0.0, 1.0)
        collection.serial.well_notch_depth = mm.UniformPrior(0.0, 1.0)
        collection.serial.well_fill_alpha = mm.UniformPrior(0.0, 1.0)
        collection.serial.well_fill_beta = mm.UniformPrior(0.0, 1.0)
        collection.serial.well_fill_gamma = mm.UniformPrior(0.0, 1.0)

        collection.serial.trap_densities.trap_densities_0 = collection.serial.trap_densities.trap_densities_1

        unit_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        hypercube_vector = collection.physical_vector_from_hypercube_vector(unit_vector)
        model = collection.instance_from_unit_vector(unit_vector)

        assert model.serial.trap_densities == (hypercube_vector[0], hypercube_vector[0]) == (0.5, 0.5)
        assert model.serial.trap_lifetimes == (hypercube_vector[1], hypercube_vector[2]) == (0.2, 0.3)
        assert model.serial.well_notch_depth == hypercube_vector[3] == 0.4
        assert model.serial.well_fill_alpha == hypercube_vector[4] == 0.5
        assert model.serial.well_fill_beta == hypercube_vector[5] == 0.6
        assert model.serial.well_fill_gamma == hypercube_vector[6] == 0.7

    def test__check_order_for_physical_values(self, test_config):
        collection = mm.ModelMapper(test_config,
                                    parallel=arctic_params.ParallelOneSpecies, serial=arctic_params.SerialTwoSpecies)

        physical_vector = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 2.0]
        model = collection.instance_from_physical_vector(physical_vector)

        assert model.parallel.trap_densities == (physical_vector[0],) == (0.1,)
        assert model.parallel.trap_lifetimes == (physical_vector[1],) == (0.2,)
        assert model.parallel.well_notch_depth == physical_vector[2] == 0.3
        assert model.parallel.well_fill_alpha == physical_vector[3] == 0.4
        assert model.parallel.well_fill_beta == physical_vector[4] == 0.5
        assert model.parallel.well_fill_gamma == physical_vector[5] == 0.6

        assert model.serial.trap_densities == (physical_vector[6], physical_vector[7]) == (0.7, 0.8)
        assert model.serial.trap_lifetimes == (physical_vector[8], physical_vector[9]) == (0.9, 1.0)
        assert model.serial.well_notch_depth == physical_vector[10] == 1.1
        assert model.serial.well_fill_alpha == physical_vector[11] == 1.2
        assert model.serial.well_fill_beta == physical_vector[12] == 1.3
        assert model.serial.well_fill_gamma == physical_vector[13] == 2.0

    def test__from_prior_medians__one_model(self, test_config):
        collection = mm.ModelMapper(test_config, species_1=arctic_params.ParallelOneSpecies)

        model_map_unit = collection.instance_from_unit_vector([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        model_map_median = collection.instance_from_prior_medians()

        assert model_map_unit.species_1.trap_densities == model_map_median.species_1.trap_densities == (0.5,)
        assert model_map_unit.species_1.trap_lifetimes == model_map_median.species_1.trap_lifetimes == (1.0,)
        assert model_map_unit.species_1.well_notch_depth == model_map_median.species_1.well_notch_depth == 0.5
        assert model_map_unit.species_1.well_fill_alpha == model_map_median.species_1.well_fill_alpha == 0.5
        assert model_map_unit.species_1.well_fill_beta == model_map_median.species_1.well_fill_beta == 0.5
        assert model_map_unit.species_1.well_fill_gamma == model_map_median.species_1.well_fill_gamma == 0.5

    def test__from_prior_medians__multiple_model(self, test_config):
        collection = mm.ModelMapper(test_config, parallel=arctic_params.ParallelOneSpecies,
                                    serial=arctic_params.SerialTwoSpecies)

        model_map_unit = collection.instance_from_unit_vector([0.5] * 14)
        model_map_median = collection.instance_from_prior_medians()

        assert model_map_unit.parallel.trap_densities == model_map_median.parallel.trap_densities == (0.5,)
        assert model_map_unit.parallel.trap_lifetimes == model_map_median.parallel.trap_lifetimes == (1.0,)
        assert model_map_unit.parallel.well_notch_depth == model_map_median.parallel.well_notch_depth == 0.5
        assert model_map_unit.parallel.well_fill_alpha == model_map_median.parallel.well_fill_alpha == 0.5
        assert model_map_unit.parallel.well_fill_beta == model_map_median.parallel.well_fill_beta == 0.5
        assert model_map_unit.parallel.well_fill_gamma == model_map_median.parallel.well_fill_gamma == 0.5

        assert model_map_unit.serial.trap_densities == model_map_median.serial.trap_densities == (0.5, 1.0)
        assert model_map_unit.serial.trap_lifetimes == model_map_median.serial.trap_lifetimes == (0.5, 1.0)
        assert model_map_unit.serial.well_notch_depth == model_map_median.serial.well_notch_depth == 0.0
        assert model_map_unit.serial.well_fill_alpha == model_map_median.serial.well_fill_alpha == 0.5
        assert model_map_unit.serial.well_fill_beta == model_map_median.serial.well_fill_beta == 0.5
        assert model_map_unit.serial.well_fill_gamma == model_map_median.serial.well_fill_gamma == 0.5

    def test__from_prior_medians__one_model__set_one_parameter_to_another(self, test_config):
        collection = mm.ModelMapper(test_config, parallel=arctic_params.ParallelOneSpecies,
                                    serial=arctic_params.SerialTwoSpecies)

        collection.serial.trap_densities.trap_densities_0 = collection.serial.trap_densities.trap_densities_1

        model_map_unit = collection.instance_from_unit_vector([0.5] * 13)
        model_map_median = collection.instance_from_prior_medians()

        assert model_map_unit.parallel.trap_densities == model_map_median.parallel.trap_densities == (0.5,)
        assert model_map_unit.parallel.trap_lifetimes == model_map_median.parallel.trap_lifetimes == (1.0,)
        assert model_map_unit.parallel.well_notch_depth == model_map_median.parallel.well_notch_depth == 0.5
        assert model_map_unit.parallel.well_fill_alpha == model_map_median.parallel.well_fill_alpha == 0.5
        assert model_map_unit.parallel.well_fill_beta == model_map_median.parallel.well_fill_beta == 0.5
        assert model_map_unit.parallel.well_fill_gamma == model_map_median.parallel.well_fill_gamma == 0.5

        assert model_map_unit.serial.trap_densities == model_map_median.serial.trap_densities == (1.0, 1.0)
        assert model_map_unit.serial.trap_lifetimes == model_map_median.serial.trap_lifetimes == (0.5, 1.0)
        assert model_map_unit.serial.well_notch_depth == model_map_median.serial.well_notch_depth == 0.0
        assert model_map_unit.serial.well_fill_alpha == model_map_median.serial.well_fill_alpha == 0.5
        assert model_map_unit.serial.well_fill_beta == model_map_median.serial.well_fill_beta == 0.5
        assert model_map_unit.serial.well_fill_gamma == model_map_median.serial.well_fill_gamma == 0.5

    def test_physical_vector_from_prior_medians(self, test_config):
        collection = mm.ModelMapper(test_config, parallel=arctic_params.ParallelOneSpecies)

        assert collection.physical_vector_from_prior_medians == [0.5, 1.0, 0.5, 0.5, 0.5, 0.5]


class TestUtility(object):

    def test_class_priors_dict(self):
        collection = mm.ModelMapper(MockConfig(), mock_class=MockClassMM)

        assert list(collection.class_priors_dict.keys()) == ["mock_class"]
        assert len(collection.class_priors_dict["mock_class"]) == 2

        collection = mm.ModelMapper(MockConfig(), mock_class_1=MockClassMM, mock_class_2=MockClassMM)

        collection.mock_class_1.one = collection.mock_class_2.one
        collection.mock_class_1.two = collection.mock_class_2.two

        assert collection.class_priors_dict["mock_class_1"] == collection.class_priors_dict["mock_class_2"]

    def test_value_vector_for_hypercube_vector(self):
        collection = mm.ModelMapper(MockConfig(), mock_class=MockClassMM)

        collection.mock_class.two = mm.UniformPrior(upper_limit=100.)

        assert collection.physical_vector_from_hypercube_vector([1., 0.5]) == [1., 50.]


class TestPriorReplacement(object):

    def test_prior_replacement(self, width_config):
        mapper = mm.ModelMapper(MockConfig(), width_config, mock_class=MockClassMM)
        result = mapper.mapper_from_gaussian_tuples(
            [(10, 3), (5, 3)])

        assert isinstance(result.mock_class.one, mm.GaussianPrior)

    def test_replace_priors_with_gaussians_from_tuples(self, width_config):
        mapper = mm.ModelMapper(MockConfig(), width_config, mock_class=MockClassMM)
        result = mapper.mapper_from_gaussian_tuples([(10, 3), (5, 3)])

        assert isinstance(result.mock_class.one, mm.GaussianPrior)

    def test_replacing_priors_for_species(self, width_config):
        mapper = mm.ModelMapper(MockConfig(), width_config=width_config, mock_class=MockSpecies)

        result = mapper.mapper_from_gaussian_tuples([(10, 3), (5, 3), (5, 3)])

        assert isinstance(result.mock_class.trap_densities.priors[0][1], mm.GaussianPrior)
        assert isinstance(result.mock_class.trap_densities.priors[1][1], mm.GaussianPrior)
        assert isinstance(result.mock_class.well_fill_beta, mm.GaussianPrior)

    def test_replace_priors_for_two_classes(self, width_config):
        mapper = mm.ModelMapper(MockConfig(), width_config=width_config, one=MockClassMM, two=MockClassMM)

        result = mapper.mapper_from_gaussian_tuples([(1, 1), (2, 1), (3, 1), (4, 1)])

        assert result.one.one.mean == 1
        assert result.one.two.mean == 2
        assert result.two.one.mean == 3
        assert result.two.two.mean == 4


class TestArguments(object):

    def test_same_argument_name(self, test_config):
        mapper = mm.ModelMapper()

        mapper.one = mm.PriorModel(MockClassMM, test_config)
        mapper.two = mm.PriorModel(MockClassMM, test_config)

        instance = mapper.instance_from_physical_vector([1, 2, 3, 4])

        assert instance.one.one == 1
        assert instance.one.two == 2
        assert instance.two.one == 3
        assert instance.two.two == 4


class TestIndependentPriorModel(object):
    def test_associate_prior_model(self):
        prior_model = mm.PriorModel(MockClassMM, MockConfig())

        mapper = mm.ModelMapper(MockConfig())

        mapper.prior_model = prior_model

        assert len(mapper.prior_models) == 1

        instance = mapper.instance_from_physical_vector([1, 2])

        assert instance.prior_model.one == 1
        assert instance.prior_model.two == 2


@pytest.fixture(name="list_prior_model")
def make_list_prior_model():
    return mm.ListPriorModel(
        [mm.PriorModel(MockClassMM, MockConfig()), mm.PriorModel(MockClassMM, MockConfig())])


class TestListPriorModel(object):
    def test_instance_from_physical_vector(self, list_prior_model):
        mapper = mm.ModelMapper(MockConfig())
        mapper.list = list_prior_model

        instance = mapper.instance_from_physical_vector([1, 2, 3, 4])

        assert isinstance(instance.list, list)
        assert len(instance.list) == 2
        assert instance.list[0].one == 1
        assert instance.list[0].two == 2
        assert instance.list[1].one == 3
        assert instance.list[1].two == 4

    def test_prior_results_for_gaussian_tuples(self, list_prior_model, width_config):
        mapper = mm.ModelMapper(MockConfig(), width_config)
        mapper.list = list_prior_model

        gaussian_mapper = mapper.mapper_from_gaussian_tuples([(1, 5), (2, 5), (3, 5), (4, 5)])

        assert isinstance(gaussian_mapper.list, list)
        assert len(gaussian_mapper.list) == 2
        assert gaussian_mapper.list[0].one.mean == 1
        assert gaussian_mapper.list[0].two.mean == 2
        assert gaussian_mapper.list[1].one.mean == 3
        assert gaussian_mapper.list[1].two.mean == 4
        assert gaussian_mapper.list[0].one.sigma == 5
        assert gaussian_mapper.list[0].two.sigma == 5
        assert gaussian_mapper.list[1].one.sigma == 5
        assert gaussian_mapper.list[1].two.sigma == 5

    def test_prior_results_for_gaussian_tuples__override_sigmas_via_width_config(self, list_prior_model, width_config):
        mapper = mm.ModelMapper(MockConfig(), width_config)
        mapper.list = list_prior_model

        gaussian_mapper = mapper.mapper_from_gaussian_tuples([(1, 0), (2, 0), (3, 0), (4, 0)])

        assert isinstance(gaussian_mapper.list, list)
        assert len(gaussian_mapper.list) == 2
        assert gaussian_mapper.list[0].one.mean == 1
        assert gaussian_mapper.list[0].two.mean == 2
        assert gaussian_mapper.list[1].one.mean == 3
        assert gaussian_mapper.list[1].two.mean == 4
        assert gaussian_mapper.list[0].one.sigma == 1
        assert gaussian_mapper.list[0].two.sigma == 2
        assert gaussian_mapper.list[1].one.sigma == 1
        assert gaussian_mapper.list[1].two.sigma == 2

    def test_automatic_boxing(self):
        mapper = mm.ModelMapper(MockConfig())
        mapper.list = [mm.PriorModel(MockClassMM, MockConfig()),
                       mm.PriorModel(MockClassMM, MockConfig())]

        assert isinstance(mapper.list, mm.ListPriorModel)


@pytest.fixture(name="mock_with_constant")
def make_mock_with_constant():
    mock_with_constant = mm.PriorModel(MockClassMM, MockConfig())
    mock_with_constant.one = mm.Constant(3)
    return mock_with_constant


class TestConstant(object):
    def test_constant_prior_count(self, mock_with_constant):
        mapper = mm.ModelMapper()
        mapper.mock_class = mock_with_constant

        assert len(mapper.prior_set) == 1

    def test_retrieve_constants(self, mock_with_constant):
        assert len(mock_with_constant.constants) == 1

    def test_constant_prior_reconstruction(self, mock_with_constant):
        mapper = mm.ModelMapper()
        mapper.mock_class = mock_with_constant

        instance = mapper.instance_from_arguments({mock_with_constant.two: 5})

        assert instance.mock_class.one == 3
        assert instance.mock_class.two == 5

    def test_constant_in_config(self):
        mapper = mm.ModelMapper()
        config = MockConfig({"MockClassMM": {"one": ["c", 3]}})

        mock_with_constant = mm.PriorModel(MockClassMM, config)

        mapper.mock_class = mock_with_constant

        instance = mapper.instance_from_arguments({mock_with_constant.two: 5})

        assert instance.mock_class.one == 3
        assert instance.mock_class.two == 5

    def test_constant_exchange(self, mock_with_constant, width_config):
        mapper = mm.ModelMapper(width_config=width_config)
        mapper.mock_class = mock_with_constant

        new_mapper = mapper.mapper_from_gaussian_means([1])

        assert len(new_mapper.mock_class.constants) == 1


@pytest.fixture(name="mapper_with_one")
def make_mapper_with_one(test_config, width_config):
    mapper = mm.ModelMapper(width_config=width_config)
    mapper.one = mm.PriorModel(MockClassMM, config=test_config)
    return mapper


@pytest.fixture(name="mapper_with_list")
def make_mapper_with_list(test_config, width_config):
    mapper = mm.ModelMapper(width_config=width_config)
    mapper.list = [mm.PriorModel(MockClassMM, config=test_config),
                   mm.PriorModel(MockClassMM, config=test_config)]
    return mapper


class TestGaussianWidthConfig(object):

    def test_config(self, width_config):
        assert 1 == width_config.get('ModelMapper_test', 'MockClassMM', 'one')
        assert 2 == width_config.get('ModelMapper_test', 'MockClassMM', 'two')

    def test_prior_classes(self, mapper_with_one):
        assert mapper_with_one.prior_class_dict == {mapper_with_one.one.one: MockClassMM,
                                                    mapper_with_one.one.two: MockClassMM}

    def test_prior_classes_list(self, mapper_with_list):
        assert mapper_with_list.prior_class_dict == {mapper_with_list.list[0].one: MockClassMM,
                                                     mapper_with_list.list[0].two: MockClassMM,
                                                     mapper_with_list.list[1].one: MockClassMM,
                                                     mapper_with_list.list[1].two: MockClassMM}

    def test_basic_gaussian_for_mean(self, mapper_with_one):
        gaussian_mapper = mapper_with_one.mapper_from_gaussian_means([3, 4])

        assert gaussian_mapper.one.one.sigma == 1
        assert gaussian_mapper.one.two.sigma == 2
        assert gaussian_mapper.one.one.mean == 3
        assert gaussian_mapper.one.two.mean == 4

    def test_gaussian_mean_for_list(self, mapper_with_list):
        gaussian_mapper = mapper_with_list.mapper_from_gaussian_means([3, 4, 5, 6])

        assert gaussian_mapper.list[0].one.sigma == 1
        assert gaussian_mapper.list[0].two.sigma == 2
        assert gaussian_mapper.list[1].one.sigma == 1
        assert gaussian_mapper.list[1].two.sigma == 2
        assert gaussian_mapper.list[0].one.mean == 3
        assert gaussian_mapper.list[0].two.mean == 4
        assert gaussian_mapper.list[1].one.mean == 5
        assert gaussian_mapper.list[1].two.mean == 6

    def test_gaussian_for_mean(self, test_config, width_config):
        mapper = mm.ModelMapper(width_config=width_config)
        mapper.one = mm.PriorModel(MockClassMM, config=test_config)
        mapper.two = mm.PriorModel(MockClassMM, config=test_config)

        gaussian_mapper = mapper.mapper_from_gaussian_means([3, 4, 5, 6])

        assert gaussian_mapper.one.one.sigma == 1
        assert gaussian_mapper.one.two.sigma == 2
        assert gaussian_mapper.two.one.sigma == 1
        assert gaussian_mapper.two.two.sigma == 2
        assert gaussian_mapper.one.one.mean == 3
        assert gaussian_mapper.one.two.mean == 4
        assert gaussian_mapper.two.one.mean == 5
        assert gaussian_mapper.two.two.mean == 6

    def test_no_override(self, test_config):
        mapper = mm.ModelMapper()

        mapper.one = mm.PriorModel(MockClassMM, config=test_config)

        mm.ModelMapper()

        assert mapper.one is not None


class TestFlatPriorModel(object):

    def test_flatten_list(self, width_config, test_config):
        mapper = mm.ModelMapper(width_config=width_config)
        mapper.list = [mm.PriorModel(MockClassMM, config=test_config)]

        assert len(mapper.flat_prior_models) == 1
        assert mapper.flat_prior_models[0][1].cls == MockClassMM
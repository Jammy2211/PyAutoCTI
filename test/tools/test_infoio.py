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
File: tests/python/infoio_test.py

Created on: 04/23/18
Author: user
"""

from __future__ import division, print_function
import sys



from autocti.tools import infoio

import pytest
import shutil
import os

@pytest.fixture(name='info_path')
def test_info():

    info_path = "{}/files/cti_params/info/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(info_path):
        shutil.rmtree(info_path)

    os.mkdir(info_path)

    return info_path

@pytest.fixture(name='class_info_path')
def test_class_info():

    class_info_path = "{}/files/cti_params/info/".format(os.path.dirname(os.path.realpath(__file__)))

    if not os.path.exists("{}/files/cti_params/".format(os.path.dirname(os.path.realpath(__file__)))):
        os.mkdir("{}/files/cti_params/".format(os.path.dirname(os.path.realpath(__file__))))

    if os.path.exists(class_info_path):
        shutil.rmtree(class_info_path)

    os.mkdir(class_info_path)

    return class_info_path

@pytest.fixture(name='pickle_path')
def test_pickle():

    pickle_path = "{}/files/cti_params/pickle/".format(os.path.dirname(os.path.realpath(__file__)))

    if os.path.exists(pickle_path):
        shutil.rmtree(pickle_path)

    os.mkdir(pickle_path)

    return pickle_path

class MockClass(object):

    def __init__(self, param1, param2):

        self.param1 = param1
        self.param2 = param2

    def generate_info(self):
        info = infoio.generate_class_info(self, prefix='', include_types=[int, float, list])
        return info


class MockClassNoInfo(object):

    def __init__(self, param1, param2):

        self.param1 = param1
        self.param2 = param2


class TestPickle:

    def test__reading_info_file_sets_up_correct_settings(self, pickle_path):

        test_class = MockClass(1,2)

        infoio.output_class_via_pickle(test_class, path=pickle_path, filename='test')

        test_class_pickle = infoio.load_class_via_pickle(path=pickle_path, filename='test')

        assert test_class.param1 == test_class_pickle.param1
        assert test_class.param2 == test_class_pickle.param2

    def test__class_check(self, pickle_path):

        test_class = MockClass(1,2)

        infoio.output_class_via_pickle(test_class, path=pickle_path, filename='test')

        with pytest.raises(IOError):
            infoio.load_class_via_pickle(path=pickle_path, filename='test', cls_check=int)


class TestGenerateClassInfo:

    def test__no_prefix__integers_only(self):

        cls = MockClass(param1=1, param2=2)

        info = infoio.generate_class_info(cls, prefix='', include_types=[int])

        assert info == r'param1 = 1' + '\n' + 'param2 = 2' + '\n'

    def test__no_prefix__integers_only_but_not_in_include_list__returns_none(self):

        cls = MockClass(param1=1, param2=2)

        info = infoio.generate_class_info(cls, prefix='', include_types=[])

        assert info is None

    def test__no_prefix__interger_and_float__include_just_float__only_float_info_generated(self):

        cls = MockClass(param1=1, param2=2.0)

        info = infoio.generate_class_info(cls, prefix='', include_types=[float])

        assert info == r'param2 = 2.0' + '\n'

    def test__no_prefix__same_as_above_but_with_list(self):

        cls = MockClass(param1=1, param2=[1.0, 2.0, 3.0])

        info = infoio.generate_class_info(cls, prefix='', include_types=[list])

        assert info == r'param2 = [1.0, 2.0, 3.0]' + '\n'

    def test__prefix_included(self):

        cls = MockClass(param1=1, param2=[1.0, 2.0, 3.0])

        info = infoio.generate_class_info(cls, prefix='pre_', include_types=[list])

        assert info == r'pre_param2 = [1.0, 2.0, 3.0]' + '\n'


class TestOutputClassInfo:

    def test__mock_class__input_integers__outputs_file_with_correct_name_and_all_attributes(self, class_info_path):

        test_class = MockClass(1, 2)

        infoio.output_class_info(cls=test_class, path=class_info_path, filename='MockClassNLOx4')

        info_file = open(class_info_path + 'MockClassNLOx4.info')

        info = info_file.readlines()

        assert info[0] == r'param1 = 1' + '\n'

    def test__mock_class__input_float_and_list__outputs_file_with_correct_name_and_all_attributes(self, class_info_path):

        test_class = MockClass(1.0, [1, 2])

        infoio.output_class_info(cls=test_class, path=class_info_path, filename='MockClassNLOx4')

        info_file = open(class_info_path + 'MockClassNLOx4.info')

        info = info_file.readlines()

        assert info[0] == r'param1 = 1.0' + '\n'
        assert info[1] == r'param2 = [1, 2]' + '\n'

    def test__mock_class__has_no_generate_info_function__raises_error(self, class_info_path):

        test_class = MockClassNoInfo(1, 2)

        with pytest.raises(IOError):
            infoio.output_class_info(cls=test_class, path=class_info_path, filename='MockClassNoInfo')


class TestAllAreNone:

    def test__all_none__returns_true(self):

        assert infoio.are_all_inputs_none(None, None, None) is True

    def test__one_is_not_none__returns_false(self):

        assert infoio.are_all_inputs_none(None, 1.0, None) is False


class TestCheckAllTuples:

    def test__all_are_tuples__no_error(self):

        infoio.check_all_tuples((1.0,), (2.0,))

    def test__one_is_not_tuple__raises_error(self):

        with pytest.raises(AttributeError):
            infoio.check_all_tuples(1.0, (2.0,))

    def test__another_is_not_tuple__raises_error(self):

        with pytest.raises(AttributeError):
            infoio.check_all_tuples(1.0)


class TestCheckAllEqualLength:

    def test__all_are_same_length__does_not_raise_error(self):

        infoio.check_all_tuples_and_equal_length((1, 2), (3, 4))

    def test__not_same_length__raises_error(self):

        with pytest.raises(AttributeError):
            infoio.check_all_tuples_and_equal_length((1), (3, 4))


class TestCheckAllFilled:

    def test__all_are_filled__does_not_raise_error(self):

        infoio.check_all_not_none(1, 2)

    def test__all_are_none__does_not_raise_error(self):

        infoio.check_all_not_none(None, None)

    def test__one_is_none__raises_error(self):

        with pytest.raises(AttributeError):
            infoio.check_all_not_none(1, None)

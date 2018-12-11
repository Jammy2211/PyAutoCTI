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
File: python/SHE_ArCTIC/InfoIO.py

Created on: 04/23/18
Author: user
"""

from __future__ import division, print_function
import sys
import os
import pickle




def make_path_if_does_not_exist(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def output_class_via_pickle(cls, path, filename):
    """ Pickle the instance of a class at a specified path and in a specified in_file"""

    make_path_if_does_not_exist(path)

    with open(path + filename + '.pkl', 'wb') as f:
        pickle.dump(cls, f)

def load_class_via_pickle(path, filename, cls_check=None):

    with open(path + filename + '.pkl', 'rb') as f:
        cls = pickle.load(f)

    if cls_check is not None:

        if isinstance(cls, cls_check) is False:

            raise IOError('Tools.load_class_via_pickle - the class being loaded is not the same type as the class '
                          'check')

    return cls

def generate_class_info(cls, prefix='', include_types=None):
    """Generate the information of a class, filtering for certain ci_data types.

    This information is kept as a string.

    Params
    -----------
    prefix : str
        The prefix of the cti_params labels in the header_info string.
    include_types : [types]
        The types of ci_data which are retained for the information (e.g. removes class methods)
    """
    info = ''
    for key, value in vars(cls).items():
        if type(value) in include_types:
            info += prefix + str(key) + ' = ' + str(value) + '\n'
            info += ''

    if info == '':
        return None

    return info

def output_class_info(cls, path, filename):
    """Output information on an instance of a class to a .header_info file in text format.

    This routine use's the classes generate_info routine, which specifies which attributes of the class instance are \
    to be output to the header_info file.

    Params
    ----------
    path : str
        The output directory path of the ci_data
    """

    cls_name = type(cls).__name__

    try :
        info = cls.generate_info()
    except AttributeError:
        raise IOError('The class ' + cls_name + ' input into the function IOTools.output_class_info has no '
                                                'generate_info function.')

    make_path_if_does_not_exist(path)

    with open(path + filename + ".info", "w") as info_file:
        info_file.write(info)
    info_file.close()

def are_all_inputs_none(*values):

    if all(x is None for x in (values)):
        return True
    else:
        return False

def check_all_tuples(*values):

    if all(type(x) == tuple for x in values):
        return
    else:
        raise AttributeError('A trap density or lifetime was not input as a tuple.')

def check_all_tuples_and_equal_length(*values):

    check_all_tuples(*values)

    n = len(values[0])

    if all(len(x) == n for x in values):
        return
    else:
        raise AttributeError('The number of trap densities and trap lifetimes are not equal.')

def check_all_not_none(*values):

    if None not in values or all(x is None for x in (values)):
        return
    else:
        raise AttributeError('One or more parameter input into setup functions is missing.')
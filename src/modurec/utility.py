#!/usr/bin/env python3

import numpy as np
import importlib
import os
import sys
import re

np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)

# def import_sage(module_name, package=None, path=None):
#     """
#     Import or reload SageMath modules with preparse if the sage file exist.
#     """
#
#     sage_name = module_name + ".sage"
#     python_name = module_name + ".sage.py"
#
#     path_from_package_name = re.sub(r'\.', r'\\', package)
#     file_path = os.path.join('', path, path_from_package_name)
#
#     sage_path = os.path.join(file_path, sage_name)
#     python_path = os.path.join(file_path, python_name)
#     module_path = os.path.join(file_path, module_name)
#
#     if os.path.isfile(sage_path):
#         os.system('sage --preparse {}'.format(sage_path));
#         os.system('mv {} {}.py'.format(python_path, module_path))
#
#     if package is not None:
#         module_name = package + "." + module_name
#
#     if module_name in sys.modules:
#         return importlib.reload(sys.modules[module_name])
#     return importlib.import_module(module_name, package=package)


def import_sage(module_name, package=None, path=''):
    """
    Import or reload SageMath modules with preparse if the sage file exist.
    """

    sage_name = module_name + ".sage"
    python_name = module_name + ".sage.py"
    if package is not None:
        path_from_package_name = re.sub(r'\.', r'\\', package)
        path = os.path.join('', path, path_from_package_name)
    else:
        path = os.path.join('', path)

    sage_path = os.path.join(path, sage_name)
    python_path = os.path.join(path, python_name)
    module_path = os.path.join(path, module_name)

    if os.path.isfile(sage_path):
        os.system('sage --preparse {}'.format(sage_path))
        os.system('mv {} {}.py'.format(python_path, module_path))

    if package is not None:
        module_name = package + "." + module_name

    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name, package=package)


def parse_sage(module_name):

    dir_path = os.path.dirname(__file__)

    sage_name = os.path.join(dir_path, module_name + ".sage")
    python_name = os.path.join(dir_path, module_name + ".sage.py")
    module_name = os.path.join(dir_path, module_name + ".py")

    os.system('sage --preparse {}'.format(sage_name))
    os.system('mv {} {}'.format(python_name, module_name))


def rolling_window(a, window=2, stride_step=1):

    """
    Take in an array and return array of rolling windows of specified length
    Parameters:
    - a: numpy array that will be windowed
    - window: integer that will be the length of the window
    - stride_step: integer, number of positions to be skipped
    Returns:
    - a_windowed: array where each entry is an array of length window
    e.g. if the numpy array is [t_1,...,t_n] and window
    is 2, then the a_windowed would be
    [(t_0, t_1), (t_1, t_2),...,(t_(n-1), t_n)]
    """

    # if a.shape >= window:
    #     msg = 'dimension_embed larger than length of numpy array'
    #     raise IndexError, ()
    # check if numpy type

    modified_shape = a.shape[-1] - (window - 1) * stride_step

    shape = a.shape[:-1] + (modified_shape,  window)
    strides = a.strides + (a.strides[-1] * stride_step,)
    a_windowed = np.lib.stride_tricks.as_strided(a, shape=shape,
                                                 strides=strides)
    return a_windowed

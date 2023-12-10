#!/usr/bin/env python3

# import numpy as np
# from numpy import inf
# import numpy.ma as ma
# import pandas as pd
# import os
# import sys
# import warnings
# import re
# import math
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from ripser import Rips
# import functools

# from . import constants as C
from . import pandex
from . import features
# from .config import PkgConfig
# from .utility import rolling_window
from .spocheck import spot_check


# BPSK = C.BPSK
# QPSK = C.QPSK
# PSK8 = C.QPSK
# QAM = C.QAM
# PSK16 = C.PSK16
#
# np.set_printoptions(suppress=True);
# np.set_printoptions(threshold=np.inf);
#
#
# TBD dtype extensions, DEBUG mod, shape for numpy type?
# add to_list method to NumpySeries
# check return for __get_axes
# check inplace
# constants managment
# save more complex pandas numpy
# change loop in plots
# consider normalization
# from sklearn.preprocessing import MinMaxScaler
#
#
# def map_scaler(data):
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     scaler.fit(data.reshape(-1, 1))
#     return scaler.transform(data.reshape(-1, 1)).reshape(-1)
# df['norm_ser'] = df['signal_sample'].map(lambda x: map_scaler(x))

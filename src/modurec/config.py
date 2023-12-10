#!/usr/bin/env python3

import os

class PkgConfig:
    def __init__(self):

        data_path = r'../data/'
        self.data_path = data_path

        test_data = r'signal_sample.csv'
        self.test_data = os.path.join(data_path, test_data)

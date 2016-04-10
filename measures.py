#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt


def mean_absolute_error(predictions, values):
    return np.sum(np.absolute(values - predictions)) / len(values)


def root_mean_square_error(predictions, values):
    sum_of_squared_errors = np.sum(np.square(values - predictions))
    return sqrt(sum_of_squared_errors / len(values))

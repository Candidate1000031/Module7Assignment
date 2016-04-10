#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def seasonal_moving_average(demand, seasonality=336, p=1):
    offset = p * seasonality

    seasonal_sum = np.zeros(len(demand) - offset)
    for i in xrange(p):
        seasonal_sum += demand[(i * seasonality):(-offset + i * seasonality)]

    predictions = seasonal_sum / p
    values = demand[offset:]

    return predictions, values


def seasonal_random_walk(demand, seasonality=336):
    return seasonal_moving_average(demand, seasonality, p=1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time

# used for scaling of demand values
__demand_min = 17500
__demand_max = 56000

__demand_min_24h = 1100000
__demand_max_24h = 2250000


def naive(data):
    """
    preprocesses raw data to return a simple time series
    :data: raw data from csv file
    :returns: data in file as numpy array
    """

    # converts time to seconds since epoch
    to_time = np.vectorize(time.mktime)

    date, period, demand = (data['settlement_date'], data['settlement_period'],
                            data['indo'])

    # one period is 30 min in length, i.e. 30 * 60 seconds
    timescale = to_time(date) + 30 * 60 * (period - 1)

    return timescale, demand


def calendar_features(data):
    get_weekday = np.vectorize(lambda t: t.tm_wday)
    get_yearday = np.vectorize(lambda t: t.tm_yday)

    weekdays = get_weekday(data['settlement_date']) / float(6)
    yeardays = (get_yearday(data['settlement_date']) - 1) / float(365)
    periods = (data['settlement_period'] - 1) / float(47)

    features = np \
        .dstack((weekdays, yeardays, periods)) \
        .reshape(len(weekdays), 3)

    return features, data['indo']


def calendar_and_current_demand(data, offset=1):
    basic_calendar_data, values = calendar_features(data)

    length = len(values)

    # do some reshaping and scale to [0, 1]
    unscaled_previous_values = values[:-offset].reshape(length - offset, 1)
    previous_values = ((unscaled_previous_values - __demand_min)
                       / (__demand_max - __demand_min))
    current_values = values[offset:]

    features = np \
        .concatenate((basic_calendar_data[offset:], previous_values), axis=1) \
        .reshape(len(previous_values), 4)

    return features, current_values


def calendar_current_and_previous_week(data, offset1=1, offset2=336):
    if offset1 >= offset2:
        raise ValueError("offset2 must be greater than offset1")

    basic_calendar_data, values = calendar_features(data)

    length = len(values) - offset2

    # do some reshaping and scale to [0, 1]
    scaled_values = (values - __demand_min) / (__demand_max - __demand_min)
    feature_values1 = scaled_values[(offset2 - offset1):-offset1] \
        .reshape(length, 1)
    feature_values2 = scaled_values[:-offset2] \
        .reshape(length, 1)

    current_values = values[offset2:]

    features = np \
        .concatenate((basic_calendar_data[offset2:], feature_values1,
                      feature_values2), axis=1) \
        .reshape(length, 5)

    return features, current_values


def calendar_and_previous_day(data, offset=1):
    basic_calendar_data, values = calendar_features(data)

    c = np.cumsum(values)
    demand_24h = c[47:] - np.concatenate([[0], c[:-48]])

    total_offset = 47 + offset
    length = len(values)

    # do some reshaping and scale to roughly [0, 1]
    scaled_demand_24h = ((demand_24h - __demand_min_24h)
                         / (__demand_max_24h - __demand_min_24h))
    reshaped_demand_24h = scaled_demand_24h[:-offset] \
        .reshape(length - total_offset, 1)

    features = np \
        .concatenate((basic_calendar_data[total_offset:], reshaped_demand_24h),
                     axis=1) \
        .reshape(length - total_offset, 4)

    return features, values[total_offset:]

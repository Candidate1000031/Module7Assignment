#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time


def demand_data(filename):
    """
    imports demand data from csv file and does minimal preprocessing
    :filename: name and path of file to read from
    :returns: data in file as numpy array
    """

    dt = [('settlement_date', 'object'), ('settlement_period', '<f8'),
          ('indo', '<f8'), ('england_wales', '<f8'), ('embedded_wind', '<f8'),
          ('embedded_solar', '<f8'), ('non_bm_stor', '<f8'),
          ('i014_demand', '<f8'), ('i014_TGSD', '<f8'), ('pumping', '<f8'),
          ('french_import', '<f8'), ('britned_import', '<f8'),
          ('moyle_import', '<f8'), ('east_west_import', '<f8')]

    data = np.genfromtxt(
        filename,
        delimiter=",",
        names=True,
        dtype=dt,
        usecols=("settlement_date", "settlement_period", "indo"),
        converters={"settlement_date": lambda x: time.strptime(x, "%d/%m/%Y")})

    return data


def holiday_data(filename):
    return

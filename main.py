#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plot
import preprocess as pp
import measures as m
from data_source import demand_data
from sklearn.svm import SVR


data_train = demand_data("./Data/DemandData_Historic-Train.csv")
data_test = demand_data("./Data/DemandData_Historic-Test.csv")

regressor = SVR(kernel="rbf", C=75000, epsilon=80, gamma=1.)

# Calendar feature based SVR

print "Calendar features only"

features_train, values_train = pp.calendar_features(data_train)
features_test, values_test = pp.calendar_features(data_test)

predictions = regressor \
    .fit(features_train, values_train) \
    .predict(features_test)

print "MAE: ", m.mean_absolute_error(predictions, values_test)
print "RMSE: ", m.root_mean_square_error(predictions, values_test)


# Calendar feature and current value based SVR

print "Calendar features and current values"
for i in [1, 2, 4, 8]:
    print "Horizon: ", i / 2., "h"
    features_train, values_train = pp.calendar_and_current_demand(data_train,
                                                                  offset=i)
    features_test, values_test = pp.calendar_and_current_demand(data_test,
                                                                offset=i)

    predictions = regressor \
        .fit(features_train, values_train) \
        .predict(features_test)

    print "MAE: ", m.mean_absolute_error(predictions, values_test)
    print "RMSE: ", m.root_mean_square_error(predictions, values_test)


# Calendar feature, current value, and previous week based SVR

print "Calendar features, current value and previous week"
for i in [1, 2, 4, 8]:
    print "Horizon: ", i / 2., "h"
    features_train, values_train = \
        pp.calendar_current_and_previous_week(data_train, offset1=i)
    features_test, values_test = \
        pp.calendar_current_and_previous_week(data_test, offset1=i)

    predictions = regressor \
        .fit(features_train, values_train) \
        .predict(features_test)

    print "MAE: ", m.mean_absolute_error(predictions, values_test)
    print "RMSE: ", m.root_mean_square_error(predictions, values_test)

from sklearn.svm import SVR
import matplotlib.pyplot as plt
import measures as m


def predict(features_train, values_train, features_test, regressor):
    predictions = regressor \
        .fit(features_train, values_train) \
        .predict(features_test)

    return predictions


def grid_validation_rbf(features_train, values_train, features_test,
                        values_test):
    mae_min = 5000
    best = ()
    for C in [70000, 75000, 80000]:
        for epsilon in [80, 90, 100]:
            for gamma in [1, 1.25, 1.5]:
                print("Evaluating RBF-kernel with epsilon=", epsilon, ", C=",
                      C, ", gamma=", gamma)

                regressor = SVR(kernel="rbf", C=C, epsilon=epsilon,
                                gamma=gamma)
                predictions = predict(features_train, values_train,
                                      features_test, regressor)

                mae = m.mean_absolute_error(predictions, values_test)

                if mae < mae_min:
                    best = (epsilon, C, gamma, mae)
                    mae_min = mae

                print "MAE: ", mae
                print "RMSE: ", m.root_mean_square_error(predictions,
                                                         values_test)
    print mae_min
    print best

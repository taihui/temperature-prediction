import numpy as np

def generator(X, Y, lookback_hours, delay_hours):

    shape_X = X.shape
    input_X = []
    input_Y = []
    #min_index = min(shape_X[0] - lookback_hours, shape_X[0] - delay_hours)
    max_index = shape_X[0] - lookback_hours - delay_hours
    for i in range(max_index):

        seq = X[i:(i + lookback_hours)]
        input_X.append(seq)
        y_value = Y[i + lookback_hours + delay_hours - 1]
        input_Y.append(y_value)

    input_X = np.array(input_X)
    input_Y = np.array(input_Y)
    return input_X, input_Y
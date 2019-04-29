import pandas as pd
import matplotlib.pyplot as plt
from data_generator import generator
from keras.models import load_model
import numpy as np


def model_prediction(model_name, model_path,lookback_hours,delay_hours,batch_size):
    test_X_dir = "dataset/station_394_test_X_V2.csv"
    test_Y_dir = "dataset/station_394_test_Y.csv"
    test_X = pd.read_csv(test_X_dir)
    test_Y = pd.read_csv(test_Y_dir)


    # normalized test_X and test_Y
    nor_test_X = (test_X - test_X.mean())/ test_X.std()
    nor_test_Y = (test_Y - test_Y.mean())/ test_Y.std()

    nor_test_X = np.array(nor_test_X)
    nor_test_Y = np.array(nor_test_Y)

    input_test_X, input_test_Y = generator(nor_test_X, nor_test_Y, lookback_hours, delay_hours)
    model = load_model(model_path)

    Y_pred = model.predict(input_test_X, batch_size=batch_size)
    prediction_error = abs(input_test_Y - Y_pred)
    prediction_mae = sum(prediction_error)/len(prediction_error)
    prediction_mae = prediction_mae.tolist()
    prediction_mae = prediction_mae[0]

    y_std = (test_Y.std()).tolist()
    y_std = y_std[0]
    y_mean = (test_Y.mean()).tolist()
    y_mean = y_mean[0]

    prediction_C = prediction_mae * y_std

    Y_true_list = input_test_Y.tolist()
    Y_pred_list = Y_pred.tolist()

    target_Y_true = Y_true_list[0:168]
    target_Y_pred = Y_pred_list[0:168]

    output_true = []
    output_pred = []
    for i in range(len(target_Y_true)):
        temp_true = (target_Y_true[i][0]) * y_std + y_mean
        output_true.append(temp_true)
        temp_pred = (target_Y_pred[i][0]) * y_std + y_mean
        output_pred.append(temp_pred)

    true_file_dir = 'result/' + model_name[:-3] + "_true.txt"
    pred_file_dir = 'result/' + model_name[:-3] + "_pred.txt"
    true_file = open(true_file_dir, "a+")
    pred_file = open(pred_file_dir, "a+")

    for item_true in output_true:
        true_file.write(str(item_true) + "\n")
    true_file.close()

    for item_pred in output_pred:
        pred_file.write(str(item_pred) + "\n")
    pred_file.close()

    # write the results into a txt file

    result_dir = "result/" + model_name[:-3] + "_.txt"
    result_file = open(result_dir, 'a+')
    result_file.write(str(prediction_mae) + "\n")
    result_file.write(str(prediction_C) + "\n")

if __name__ == '__main__':
    lookback_hours = [1,24]
    delay_hours = [1,24]
    batch_size = 256
    for lh in lookback_hours:
        for dh in delay_hours:
            # prediction for SimpleRNN
            model_name = "SimpleRNN_" + str(lh) + "_" + str(dh) + ".h5"
            model_path = "model/RNN/" + model_name
            model_prediction(model_name, model_path, lh, dh, batch_size)

            # prediction for LSTM
            model_name = "LSTM_" + str(lh) + "_" + str(dh) + ".h5"
            model_path = "model/LSTM/" + model_name
            model_prediction(model_name, model_path, lh, dh, batch_size)

    print("Finish!")
    pass





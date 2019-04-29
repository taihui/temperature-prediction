import matplotlib.pyplot as plt

def get_actual_rnn_lstm(lookback_hours, delay_hours):
    lh = lookback_hours
    dh = delay_hours
    rnn_true_list = []
    rnn_pred_list = []
    lstm_pred_list = []

    rnn_true_dir = "result/"+ "SimpleRNN_" + str(lh) + "_" + str(dh) + "_true.txt"
    rnn_pred_dir = "result/"+ "SimpleRNN_" + str(lh) + "_" + str(dh) + "_pred.txt"
    lstm_pred_dir = "result/"+ "LSTM_" + str(lh) + "_" + str(dh) + "_pred.txt"

    rnn_true_file = open(rnn_true_dir, "r")
    rnn_pred_file = open(rnn_pred_dir, "r")
    lstm_pred_file = open(lstm_pred_dir, "r")

    line = rnn_true_file.readline().strip()
    while line:
        rnn_true_list.append(float(line))
        line = rnn_true_file.readline().strip()

    line = rnn_pred_file.readline().strip()
    while line:
        rnn_pred_list.append(float(line))
        line = rnn_pred_file.readline().strip()


    line = lstm_pred_file.readline().strip()
    while line:
        lstm_pred_list.append(float(line))
        line = lstm_pred_file.readline().strip()


    return rnn_true_list, rnn_pred_list, lstm_pred_list

if __name__ == '__main__':
    lookback_hours = [1,24]
    delay_hours = [1,24]
    for lh in lookback_hours:
        for dh in delay_hours:
            trur_list, rnn_pred, lstm_pred = get_actual_rnn_lstm(lh, dh)
            plt.figure(1)
            plt.subplot(2,1,1)
            plt.plot(range(len(trur_list)), trur_list, label = "Actual Temp")
            plt.plot(range(len(trur_list)), rnn_pred, label = "RNN")
            #plt.plot(range(len(trur_list)), lstm_pred, label="LSTM")
            plt.ylabel("temperature")
            plt.legend()
            figure_name = "RNN Prediction ( LB=" + str(lh) + ", LF=" + str(dh) + ")"
            plt.title(figure_name)

            plt.subplot(2,1,2)
            plt.plot(range(len(trur_list)), trur_list, label="Actual Temp")
            plt.plot(range(len(trur_list)), lstm_pred, label="LSTM")
            plt.ylabel("temperature")
            plt.legend()
            figure_name = "LSTM Prediction ( LB=" + str(lh) + ", LF=" + str(dh) + ")"
            plt.title(figure_name)

            plt.tight_layout()
            figure_name = "Prediction ( LB=" + str(lh) + ", LF=" + str(dh) + ")"
            plt.savefig("figure/final/"+ figure_name +".png")
            plt.clf()




import keras
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import RNN, SimpleRNN, LSTM, Dense, Dropout, BatchNormalization, Activation, Input
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_generator import generator
from prediction import model_prediction


def get_train_validation_data(Xfile = "dataset/station_394_train_X_V2.csv", Yfile = "dataset/station_394_train_Y.csv"):
    raw_X = pd.read_csv(Xfile)
    raw_Y = pd.read_csv(Yfile)

    validation_size = int(len(raw_X) * 0.2)
    train_size = len(raw_X) - validation_size
    train_X = raw_X.iloc[0:train_size]
    val_X = raw_X.iloc[train_size+1 : len(raw_X)]

    train_Y = raw_Y.iloc[0:train_size]
    val_Y = raw_Y.iloc[train_size + 1: len(raw_Y)]



    nor_train_X = (train_X - train_X.mean())/train_X.std()
    nor_train_Y = (train_Y - train_Y.mean()) / train_Y.std()

    nor_train_X = np.array(nor_train_X)
    nor_train_Y = np.array(nor_train_Y)


    nor_val_X = (val_X - val_X.mean())/val_X.std()
    nor_val_Y = (val_Y - val_Y.mean()) / val_Y.std()

    nor_val_X = np.array(nor_val_X)
    nor_val_Y = np.array(nor_val_Y)

    mn_std = [train_Y.mean(), train_Y.std(), val_Y.mean(), val_Y.std()]

    return nor_train_X, nor_train_Y, nor_val_X,nor_val_Y, mn_std


def rnn_model(lookback_hours, delay_hours, batch_size, num_epoch, model_path):
    # setting information

    input_features = 16
    output_features = 1

    train_X, train_Y, val_X, val_Y, mn_std = get_train_validation_data()
    input_train_X, input_train_Y = generator(train_X, train_Y, lookback_hours, delay_hours)
    input_val_X, input_val_Y = generator(val_X, val_Y, lookback_hours, delay_hours)


    ######## Build RNN model ########
    model = Sequential()
    model.add(LSTM(256, input_shape=(input_train_X.shape[1], input_train_X.shape[2]),
                        return_sequences=True,dropout=0.1,recurrent_dropout=0.5))
    model.add(LSTM(512,return_sequences=True, dropout=0.1,recurrent_dropout=0.5))
    model.add(LSTM(256, activation='elu', dropout=0.1,recurrent_dropout=0.5))
    #
    model.add(Dense(1))

    callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath= model_path,
    monitor = 'val_loss', save_best_only = True, mode = 'min', period=1),]

    #keras.callbacks.ReduceLROnPlateau(monitor= 'val_loss', factor = 0.1, patience = 2)

    model_optimizer = optimizers.Adam(lr=0.001)
    model.compile(optimizer = model_optimizer, loss='mae', metrics=['mae'])

    history = model.fit(input_train_X, input_train_Y,
                        epochs=num_epoch,
                        batch_size=batch_size,
                        callbacks = callbacks_list,
                        validation_data=(input_val_X, input_val_Y))


    ##########################################
    #######        Draw Figures        #######
    ##########################################
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']

    epochs = range(0, len(train_loss))

    plt.figure()
    plt.plot(epochs, train_mae, label='Training MAE')
    plt.plot(epochs, val_mae, label='Validation MAE')
    plt.title('RNN Training & Validation MAE (' +str(lookback_hours)+',' + str(delay_hours)+')')
    plt.xlabel('Training Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('figure/LSTM_Train_MAE_'+ str(lookback_hours) + '_' + str(delay_hours) + '.png')

    print(mn_std)




if __name__ == '__main__':
    lookback_hours = [1, 24]
    delay_hours = [1,24]
    batch_size = 256
    num_epoch = 100

    for lh in lookback_hours:
        for dh in delay_hours:
            model_name = "LSTM_" + str(lh) + "_" + str(dh) + ".h5"
            model_path = "model/" + model_name
            rnn_model(lh, dh, batch_size, num_epoch, model_path)
    #model_prediction(model_name, model_path, lookback_hours, delay_hours, batch_size)


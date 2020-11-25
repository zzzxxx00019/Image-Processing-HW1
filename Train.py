import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt

def Load_Energy_Data():
    Dataset = pd.read_csv('./energy_dataset.csv')
    Dataset = Dataset.iloc[:]['price actual']
    return Dataset

def normalize(train):
  train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train_norm

def LSTM_Model(input_shape):
    model = Sequential()
    model.add(LSTM(128, input_length = input_shape[1], input_dim=input_shape[2], return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(25))
    model.add(Dense(5,activation='linear'))
    model.add(Dense(1))
    model.compile(loss="mean_absolute_error", optimizer="Adam",metrics=['mape'])
    model.summary()
    return model

def Draw(history):
    fig = plt.figure()
    plt.plot(history.history['loss'], 'ro',label='training loss', color='blue')
    plt.plot(history.history['val_loss'], 'ro', label='val loss', color='red')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig('./result_image/loss.png')

def BuildTrain(data, past, future):
    X_train, Y_train = [], []
    for i in range(data.shape[0]-future-past):
        X_train.append(np.array(data.iloc[i:i+past]))
        Y_train.append(np.array(data.iloc[i+past:i+past+future]['price actual']))
    return np.array(X_train), np.array(Y_train)    

def PricePredict(predicted_data, actual_result, slide, fig_name):
    fig = plt.figure()
    
    predict = []
    actual = []
    for i in range (0, predicted_data.shape[0], slide):
        predict.append(predicted_data[i][0])
        actual.append(actual_result[i][0])
    
    plt.plot(actual, color = 'red', label = 'Real Price')
    plt.plot(predict, color = 'blue', label = 'Predict Price')
    plt.xlabel('Time')
    plt.ylabel('Price')    
    plt.legend()
    fig.savefig('./result_image/'+fig_name)
        
def Training_Model(x, y):
    x_train, x_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25)
    
    model = LSTM_Model(x_train.shape)
    history = model.fit(x_train, Y_train, epochs=500, batch_size=64, validation_data=(x_test, Y_test))
    Draw(history)

    model.save('LSTM.h')
    
    PricePredict(model.predict(x_train) ,Y_train, 200, 'Train_Predict')
    PricePredict(model.predict(x_test) ,Y_test, 50, 'validation_Predict')

if __name__ == '__main__' :
    Weather = pd.read_csv('weather_dataset.csv').drop(columns=['Unnamed: 0'])
    Energy = Load_Energy_Data()
    data = pd.concat ( [Weather, Energy], axis=1 )
    
    x_train, y_train = BuildTrain(data, 20, 1)
    Training_Model(x_train, y_train)
    

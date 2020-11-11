import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt

def Read_Data():
    Barcelona = pd.read_csv('./city_var2/Barcelona.csv').drop(columns=['Unnamed: 0', 'city_name'])
    Bilbao = pd.read_csv('./city_var2/Bilbao.csv').drop(columns=['Unnamed: 0'])
    Madrid = pd.read_csv('./city_var2/Madrid.csv').drop(columns=['Unnamed: 0'])
    Seville = pd.read_csv('./city_var2/Seville.csv').drop(columns=['Unnamed: 0'])
    Valencia = pd.read_csv('./city_var2/Valencia.csv').drop(columns=['Unnamed: 0'])
    return Barcelona, Bilbao, Madrid, Seville, Valencia

def Load_Energy_Data():
    Dataset = pd.read_csv('./energy_dataset.csv').drop(columns=['time', 'forecast wind offshore eday ahead'])
    Dataset = Dataset.iloc[:]['price actual']
    #Dataset = Dataset.drop( columns = ['time'] )
    return Dataset

def normalize(train):
  #train = train.drop(["Date"], axis=1)
  train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train_norm

def LSTM_Model(input_shape):
    model = Sequential()
    model.add(LSTM(256, input_length = input_shape[1], input_dim=input_shape[2], return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(5,activation='linear'))
    model.add(Dense(1))
    model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
    model.summary()
    return model

def Draw(history):
    fig = plt.figure()
    plt.plot(history.history['loss'],label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig('VGG16'+'loss.png')

def BuildTrain(data, past, future):
    X_train, Y_train = [], []
    for i in range(data.shape[0]-future-past):
        X_train.append(np.array(data.iloc[i:i+past]))
        Y_train.append(np.array(data.iloc[i+past:i+past+future]['price actual']))
    return np.array(X_train), np.array(Y_train)    

def Training_Model(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True)    
    y_train = y_train[:,:,np.newaxis]
    y_test = y_test[:,:,np.newaxis]

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    model = LSTM_Model(x_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test), callbacks=[callback])
    Draw(history)

    model.save('LSTM.h')
    

if __name__ == '__main__' :
    Barcelona, Bilbao, Madrid, Seville, Valencia = Read_Data() 
    Energy = Load_Energy_Data()

    data = pd.concat ( [Barcelona, Energy], axis=1 )

    print(data)

    #data = normalize(data)

    x_train, y_train = BuildTrain(data, 30, 1)
    print(x_train.shape)
    print(y_train.shape)
    Training_Model(x_train, y_train)


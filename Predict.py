import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_model():
    model = keras.models.load_model('LSTM.h')
    model.summary()
    
    return model

def load_data():
    City = pd.read_csv('./Barcelona.csv').drop(columns = ['Unnamed: 0', 'city_name'])
    Data = pd.read_csv('./energy_dataset.csv')
    Data = Data.iloc[:]['price actual']
    
    Dataset = pd.concat ( [City, Data], axis=1 )

    print (Dataset)
    return Dataset

def build_data(data, past, future):
    X_train, Y_train = [], []
    for i in range(data.shape[0]-future-past):
        X_train.append(np.array(data.iloc[i:i+past]))
        Y_train.append(np.array(data.iloc[i+past:i+past+future]['price actual']))
    return np.array(X_train), np.array(Y_train) 

def draw_predict(predicted_data, actual_result, slide, fig_name):

    print(predicted_data.shape)
    print(actual_result.shape)

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
    fig.savefig(fig_name)

if __name__ == '__main__' :
    data = load_data()
    x, y = build_data(data , 20, 1)
    
    model = load_model()
    draw_predict(model.predict(x), y, 250, 'predict result')
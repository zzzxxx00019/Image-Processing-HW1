import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_model():
    model = keras.models.load_model('LSTM.h')
    model.summary()
    
    return model

def load_data():
    City = pd.read_csv('./dataset/weather_dataset.csv').drop(columns = ['Unnamed: 0'])
    Data = pd.read_csv('./dataset/energy_dataset.csv')
    Data = Data.iloc[:]['price actual']
    
    Dataset = pd.concat ( [City, Data], axis=1 )

    print (Dataset)
    return Dataset

'''devide data to train and predict train'''
def build_data(data, past, future):
    X_train, Y_train = [], []
    for i in range(data.shape[0]-future-past):
        X_train.append(np.array(data.iloc[i:i+past]))
        Y_train.append(np.array(data.iloc[i+past:i+past+future]['price actual']))
    return np.array(X_train), np.array(Y_train) 

'''plot predict'''
def draw_predict(predicted_data, actual_result, slide, fig_name):
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
    fig.savefig('./result_image/' + fig_name)

'''draw CDF mape <= 25'''
def draw_CDF(data):
    probability = []
    for i in range (1, 26, 1):
        tmp = np.sum(data < i) / data.shape[0]
        probability.append(tmp)
        print('Mape < ' + str(i) + ' : ' + str(tmp))
    
    fig = plt.figure()
    plt.plot(probability)
    plt.xlabel('mape')
    plt.ylabel('probability')
    fig.savefig('./result_image/CDF result')
    
if __name__ == '__main__' :
    data = load_data()
    x, y = build_data(data , 20, 1)
    
    model = load_model()
    pred_y = model.predict(x)
    
    MAPE_value = []
    for i in range (y.shape[0]):
        mape = (np.abs((pred_y[i]-y[i])/y[i]))*100
        MAPE_value.append(mape)
        
    MAPE_value = np.array(MAPE_value)
    draw_CDF(MAPE_value)
    draw_predict(pred_y, y, 250, 'predict result')
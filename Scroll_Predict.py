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

    return Dataset

'''devide data to train and predict train'''
def build_data(data, past, future):
    X_train, Y_train = [], []
    for i in range(data.shape[0]-future-past):
        X_train.append(np.array(data.iloc[i:i+past]))
        Y_train.append(np.array(data.iloc[i+past:i+past+future]['price actual']))
    return np.array(X_train), np.array(Y_train) 

def scroll_pred(data, y, model):
    current_frame = data[0].copy()
    Mape = []
    for i in range (10):
        current_frame = current_frame.reshape(1, current_frame.shape[0], current_frame.shape[1])
        pred_value =  model.predict(current_frame)[0]
        
        tmp_mape = (abs( (pred_value[0] - y[i][0])/y[i][0] ))*100
        Mape.append(tmp_mape)
        
        next_frame = data[i+1].copy()
        next_frame[next_frame.shape[0]-1][next_frame.shape[1]-1] = pred_value[0]
        current_frame = next_frame

    return Mape

if __name__ == '__main__' :
    data = load_data()
    x, y = build_data(data , 20, 1)
    x = x[int(x.shape[0]*0.75):]
    y = y[int(y.shape[0]*0.75):]
    
    print(x.shape)
    print(y.shape)
    
    model = load_model()
    
    totol_mape = []
    for i in range (x.shape[0]-20):
        print('start to predict : ',str(i))
        scroll_data = x[i : i+20]
        real_value  = y[i : i+20]    
        totol_mape.append( scroll_pred(scroll_data, real_value, model) )
    
    totol_mape = np.array(totol_mape)
    result = np.mean(totol_mape, axis=0)
    
    print(result)
    
    fig = plt.figure()
    plt.plot(result, color = 'red', label = 'Scroll Predict Mape')
    plt.xlabel('Scroll times')
    plt.ylabel('Mape')    
    plt.legend()
    fig.savefig('./result_image/Scroll Mape.jpg')    
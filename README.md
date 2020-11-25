# Power Predict

## 工具版本
* Python 3.8.3
* Tensorflow 2.3.1
* Keras 2.4.3

## Table.py
* 對不同城市的資料合併 ( 加總取平均 )

## Train.py
* Keras 版本不一致可能會導致 error 發生
* LSTM_Model -> 直接在這更改模型結構
* Normalize -> 正規劃數值，使數值收斂
* Draw -> 畫出模型損失表格，輸出 loss.png
* BuildTrain -> 
  * past : 選擇幾天前的資料 
  * future : 預測未來幾天的 price actual
* Training_Model -> 訓練模型，輸出 LSTM.h

## Predict.py
* 載入 LSTM 模型，預測整筆資料
* 將實際值與預測值以圖形顯示
* 每 250 筆資料為一個預測點，避免數據過於密集
* 計算並繪出 mape 低於 1 ~ 24 的概率
  * mape < 1 : 0.1266
  * mape < 2 : 0.2566
  * mape < 3 : 0.3874
  * mape < 4 : 0.5172
  * mape < 5 : 0.6430
  * mape < 10 : 0.9270
  * mape < 20 : 0.9876

## TO DO
1. 處理 dataset.csv 裡面的 bad data
2. 調整模型參數
3. 加入 time label 重新訓練模型

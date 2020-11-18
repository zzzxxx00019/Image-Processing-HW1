# Power Predict

## Table.py
* 對不同城市的資料合併 ( 加總取平均 )

## Train.py
* LSTM_Model -> 直接在這更改模型結構
* Normalize -> 正規劃數值，使數值收斂
* Draw -> 畫出模型損失表格，輸出 loss.png
* BuildTrain -> 
  * past:選擇幾天前的資料 future
  * 預測未來幾天的 price actual
* Training_Model -> 訓練模型，輸出 LSTM.h

## Predict.py
* 載入 LSTM 模型，預測整筆資料
* 將實際值與預測值以圖形顯示
* 每 250 筆資料為一個預測點，避免數據過於密集

## TO DO
1. 處理 dataset.csv 裡面的 bad data
2. 調整模型參數

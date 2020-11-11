# Power Predict

## Table.py
* Data_Replace 可以直接對五個表格動作

## Train.py
* LSTM_Model -> 直接在這更改模型結構
* Normalize -> 正規劃數值 ( 還在研究怎麼用 )
* Draw -> 畫出模型損失表格，輸出 loss.png
* BuildTrain -> past:選擇幾天前的資料 future:預測未來幾天的 price actual
* Training_Model -> 訓練模型，輸出 LSTM.h

## TO DO
1. 五個城市的資料合併
2. 處理 dataset.csv 裡面的 bad data
3. 調整模型參數

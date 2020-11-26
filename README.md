# Power Predict

## 工具版本
* Python 3.8.3
* Tensorflow 2.3.1
* Keras 2.4.3

## Model
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 20, 128)           83456     
_________________________________________________________________
lstm_1 (LSTM)                (None, 20, 256)           394240    
_________________________________________________________________
dropout (Dropout)            (None, 20, 256)           0         
_________________________________________________________________
lstm_2 (LSTM)                (None, 20, 256)           525312    
_________________________________________________________________
dropout_1 (Dropout)          (None, 20, 256)           0         
_________________________________________________________________
flatten (Flatten)            (None, 5120)              0         
_________________________________________________________________
dense (Dense)                (None, 25)                128025    
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 130       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 6         
=================================================================
Total params: 1,131,169
Trainable params: 1,131,169
Non-trainable params: 0
_________________________________________________________________
```

## Train.py
* Keras 版本不一致可能會導致 error 發生
* LSTM_Model -> 直接在這更改模型結構
* Normalize -> 正規劃數值，使數值收斂
* Draw -> 畫出模型損失表格，輸出 loss.png
* BuildTrain -> 
  * past : 選擇幾天前的資料 
  * future : 預測未來幾天的 price actual
* Training_Model -> 訓練模型，輸出 LSTM.h

![image](https://github.com/zzzxxx00019/Power_Predicted/blob/main/result_image/loss.png)

* 模型訓練結果：
  * loss: 1.2547 - mape: 2.4196 - val_loss: 2.0603 - val_mape: 3.9563

## Predict.py
* 載入 LSTM 模型，預測整筆資料
* 將實際值與預測值以圖形顯示
* 每 250 筆資料為一個預測點，避免數據過於密集
* 計算並繪出 mape 低於 1 ~ 25 的概率
  * Mape < 1 : 0.40296207516479754
  * Mape < 2 : 0.676939759723768
  * Mape < 3 : 0.8140570156664669
  * Mape < 4 : 0.8801757840367549
  * Mape < 5 : 0.9161886824758154
  * Mape < 10 : 0.9761721313814457
  * Mape < 15 : 0.9902976343349599
  * Mape < 25 : 0.9970607539308849

![image](https://github.com/zzzxxx00019/Power_Predicted/blob/main/result_image/CDF%20result.png)

## 整體預測結果
![image](https://github.com/zzzxxx00019/Power_Predicted/blob/main/result_image/Train_Predict.png)

## TO DO
1. 處理 dataset.csv 裡面的 bad data
2. 調整模型參數
3. 改善 overfitting 問題
4. Tabel 針對日出日落時間更動 UV 數值
5. 針對人口分布改變合併權重

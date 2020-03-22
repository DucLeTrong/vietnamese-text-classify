# Vietnamese text classifier using LSTM model
<<<<<<< HEAD
[![Accuracy](https://img.shields.io/badge/accuracy-88%2E98%25-green.svg)](https://github.com/DucLeTrong/vienamese-text-classify)
=======
[![Accuracy](https://img.shields.io/badge/accuracy-89%2E19%25-green.svg)](https://github.com/DucLeTrong/vienamese-text-classify)
>>>>>>> 2d15652d3ddbd6a600e4f87a7dae1d1f127dd24c
## About data set
The Vietnamese dataset consists of 10 different topics
- [VNTC](https://github.com/duyvuleo/VNTC)

## Training
- Install requirement
```
>> pip install -r requirements.txt
```
- Prepare data:
```
>> cd data/ && unrar x -r Test_Full.rar && unrar x -r Train_Full.rar
>> python load_data.py
```
- Train model 
```
>> python train.py
```
## Results
![png](images/accuracy.png)

- Classify text
```
<<<<<<< HEAD
>> python infer.py --prime "Đêm hôm qua, đội tuyển Việt Nam đã bay đến Trung Quốc chuẩn bị cho giải vô địch Châu Á."
=======
>> python infer.py --prime "Đêm hôm qua, đội tuyển Việt Nam đã có chuyến bay đến Trung Quốc"
>>>>>>> 2d15652d3ddbd6a600e4f87a7dae1d1f127dd24c
```



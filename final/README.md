# Listen and Translate
Team: NTU_r06946003_鬼氏企業 / d05921027 張鈞閔 r03902085 吳家謙 r06922030 傅敏桓 r06946003 湯忠憲

### Package : 
`keras==2.0.8` &nbsp; ` numpy==1.13.3`  &nbsp;` pandas==0.20.3` &nbsp; `sklearn==0.19.1` &nbsp;



### Usage :
**Training:**
```
cd src
python3 inference.py [train_data_path] [test_data_path] [train_caption_path] [test_csv_path] [output_path] [GPU number]
```
**Testing:**
```
cd src
python3 inference.py [train_data_path] [test_data_path] [train_caption_path] [test_csv_path] [output_path]
```

### Note :
The model was stored in ./model folder. <br>
In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```
### model architecture :
**RNN retrieval model**
![alt text](https://github.com/thtang/ML2017FALL/blob/master/final/output/RNN_archi.png)
**RNN+CNN retrieval model**
![alt text](https://github.com/thtang/ML2017FALL/blob/master/final/output/CNN_RNN_archi.png)
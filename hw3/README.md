# Image Sentiment Classification

### Package : 
`Keras` &nbsp; ` Numpy`  &nbsp;` pandas` &nbsp;


### Note :
In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```
To reduce download time consumption, I upload **a singel model** with acc about 0.70688 and 0.70743 on private and public board <br>
The best performance in kaggle were achieved by ensembling 7 models.<br>

Usage<br>

```
bash hw3_test.sh <test data> <output file>
```
### Best model architecture :
![alt text](https://github.com/thtang/ML2017FALL/blob/master/hw3/best_cnn.png)
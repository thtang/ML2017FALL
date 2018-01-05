
### Package : 
`Keras` &nbsp; ` Numpy`  &nbsp;` pandas` &nbsp; `sklearn` &nbsp;


### Note :

The model was stored on Dropbox. Run the hw5.sh/hw5_best.sh to download and predict.<br>
In my .py script, I used the following script to assign the task running on GPU 0.<br>

```
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
```
Usage<br>

```
bash hw5.sh <test.csv path> <prediction file path> <movies.csv path> <users.csv path>
bash hw5_best.sh <test.csv path> <prediction file path> <movies.csv path> <users.csv path>
```

### Best model architecture :

![alt text](https://github.com/thtang/ML2017FALL/blob/master/hw5/best_model.png)
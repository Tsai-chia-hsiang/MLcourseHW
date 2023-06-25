```python
import numpy as np
import os
```


```python
dnum = 10000
trainsize = np.floor(dnum*0.8).astype(np.int32)
def gth(x):
    return np.mean(np.sin(x), axis=1, keepdims=True)+1.352

x_all = np.random.randint(low=0, high=9, size=(dnum, 32))
y_all = gth(x_all)
print(y_all.shape)
shuffle = np.random.permutation(x_all.shape[0])
trainid=shuffle[:trainsize]
testid=shuffle[trainsize:]

validationsize = np.floor(trainsize*0.2).astype(np.int32)
validationid=trainid[:validationsize]
trainid = trainid[validationsize:]

x_train = x_all[trainid, :]
y_train = y_all[trainid, :]
x_val = x_all[validationid, :]
y_val = y_all[validationid, :]
x_test = x_all[testid, :]
y_test = y_all[testid, :]

print(f"train size {x_train.shape}")
print(f"validation size {x_val.shape}")
print(f"test size {x_test.shape}")
```

    (10000, 1)
    train size (6400, 32)
    validation size (1600, 32)
    test size (2000, 32)
    


```python
traindir= os.path.join("train")
if not os.path.exists(traindir):
    os.mkdir(traindir)

valdir= os.path.join("val")
if not os.path.exists(valdir):
    os.mkdir(valdir)

testdir= os.path.join("test")
if not os.path.exists(testdir):
    os.mkdir(testdir)
```


```python
np.save(os.path.join(traindir, "x"), x_train)
np.save(os.path.join(traindir, "y"), y_train)

np.save(os.path.join(valdir, "x"), x_val)
np.save(os.path.join(valdir, "y"), y_val)

np.save(os.path.join(testdir, "x"), x_test)
np.save(os.path.join(testdir, "y"), y_test)
```

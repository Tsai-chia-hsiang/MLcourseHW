# Assignment2 Regression 報告
資工四 408410098 蔡嘉祥

## 如何執行:
- 所需要的第三方套件:
  - numpy 
  - pandas
  - tqdm
  - matplotlib

下命令 : __```python regression.py```__ 來執行即可。

會自動在工作目錄下建一個 ```result_v1``` 的資料夾，結果即是寫在裡面。
- Q1 Q2 : ```./result_v1/linreg/```
- Q3 : ```./result_v1/polyreg/```
- Q4 : ```./result_v1/sinreg/```
- Q5 : ```./result_v1/sin_m*/```
- Q6 : ```./result_v1/regu_sin_15/```

## 結果
以下結果我都有附在 ```./result/``` 資料夾。

### Q1 & Q2 
|cv term|training mse|validation mse|
|-|-|-|
|0|0.396|2.203|
|1|0.655|0.818|
|2|0.700|0.568|
|3|0.536|1.221|
|4|0.770|0.187|
|avg|__0.612__|__1.000__|
|whole data|__0.649__|x|

fit line:

$$W=\begin{bmatrix}
w_0 \\ 
w_1
\end{bmatrix}=\begin{bmatrix}
0.637 \\ 
2.136
\end{bmatrix}$$

<img src="result\linreg\whole\fitline.jpg">

### Q3 
#### Degree 5:
|cv term|training mse|validation mse|
|-|-|-|
|0|0.409|1.447|
|1|0.441|42.199|
|2|0.509|1.187|
|3|0.276|11.296|
|4|0.284|3.364|
|avg|__0.384__|__11.899__|
|whole data|__0.531__|x|

fit line:
<img src="result\polyreg\p5\whole\fitline.jpg">

#### Degree 10
|cv term|training mse|validation mse|
|-|-|-|
|0|0.074|7358.430|
|1|0.092|4.630|
|2|0.007|2879.096|
|3|0.061|3.621|
|4|0.009|509.545|
|avg|0.049|2151.064|
|whole data|0.130|x|

fit line:
<img src="result\polyreg\p10\whole\fitline.jpg">

#### Degree 14
|cv term|training mse|validation mse|
|-|-|-|
|0|9.907|22.072|
|1|25.296|39.466|
|2|74.034|96851.973|
|3|6.602|11.210|
|4|74.810|113004.512|
|avg|38.130|41985.847|
|whole data|$1.5\times 10^{-10} \approx 0$|x|

fit line:
<img src="result\polyreg\p14\whole\fitline.jpg">

### Q1 Q2 Q3 討論

從cross validation 的training mse 與 validation mse 中可以發現，degree 越大，training mse 可以越小，但是validation mse 越大，可以很明顯的觀察到 overfitting 的現象。

不過有個問題是在 degree = 14 的時候，不知道為甚麼當訓練資料只有12個點的時候 ( i.e. a fold of 5-folds cross validation )，training mse 我無法下降到跟 whole data training 時一樣低。

為了這個問題，我做了一個實驗，可以執行 ```python test.py``` 來進行。

我將相同方法產生的數據，隨機選擇12 個去訓練以模擬 5-fold cross validation 其中一次fold 的狀況。

結果在 ```./result/polyreg/testing/cmp.csv``` :

|term|mse|
|-|-|
|random 12-p10 |0.171|
|random 12-p14 |34.009|
|whole data-p10|0.217|
|whole data-p14|4.8 $\times 10^{-11}\approx$ 0|

可以看到 degree = 10 在隨機選擇 12 個點的情況下反而比 degree = 14 mse還要來的小很多；但如果使用全部的點去訓練(15 個點)，則能符合「model 越強大，可以達到的training error 越小」。

目前我想到的原因是除了程式寫錯或次方太高導致 overflow 外，有可能是 __degree = 14 的多項方程式需要 15 個點來求解__ ，但是每次 cross validation 時只有 12 個點，所以才會在 cv 時不如其他較低的維度，但當上升到15個點的時候就有能力 fit 好，則能符合「model 越強大，可以達到的training error 越小」的論述。

依此結果，即使 model 強大，如果不夠多的訓練資料，其實就連 train 也 train 不起來嗎 ?


### Q4 

#### Degree 1 (linear regression):

|cv term|training mse|validation mse|
|-|-|-|
|0|0.246|0.265|
|1|0.223|0.320|
|2|0.224|0.406|
|3|0.290|0.043|
|4|0.138|0.907|
|avg|__0.224__|__0.388__|
|whole data|__0.239__|x|

fit line : <img src="result\sinreg\p1\whole\fitline.jpg">

#### Degree 5 :

|cv term|training mse|validation mse|
|-|-|-|
|0|0.0006|0.0012|
|1|0.0004|0.01583|
|2|0.0004|0.1042|
|3|0.0005|0.0020|
|4|0.0005|0.0016|
|avg|__0.0005__|__0.0250__|
|whole data|__0.0007__|x|

fit line : <img src="result\sinreg\p5\whole\fitline.jpg">

#### Degree 10 :

|cv term|training mse|validation mse|
|-|-|-|
|0 |$1.72\times 10^{-5}$|0.0008|
|1|$2.56\times 10^{-5}$|55.5000|
|2|0.00015|0.120|
|3|0.00014|0.480|
|4|$2.91\times 10^{-6}$|0.00046|
|avg|$6.64\times 10^{-5}$|11.20|
|whole data|$2.16\times 10^{-5}$|x|

fit line : <img src="result\sinreg\p10\whole\fitline.jpg">

#### Degree 14 :

|cv term|training mse|validation mse|
|-|-|-|
|0|0.063|0.045|
|1|0.834|1.546|
|2|0.015|0.025|
|3|0.029|0.027|
|4|0.003|2363.036|
|avg|__0.189__|__472.936__|
|whole data|__0.003__|x|

fit line : <img src="result\sinreg\p14\whole\fitline.jpg">

#### 討論

從我的實驗中，可以看到 degree = 5 是表現最好的model (有最小的average  validation mse)，linear regression 表現出 training mse & validation mse 都較大的 underfitting 的情況； degree = 10, 14 則是逐漸overfitting。

同樣，degree = 14 在cross validation 中的training mse表現依舊不如 degree = 5, 10，也遇到了 Q3 的問題。

這邊還有一個問題，degree = 14 連 whole data 的 training error 也比
degree = 5, 10 還大，一樣不符合「model 越強大，可以達到的training error 越小」。

這裡我猜測，由於產生的 $\epsilon \sim N(0,0.04)$，biases 距離 0 的距離高機率較小 (Gaussian sampling 的  variance 較小)，使得整體資料集更接近 groundtruth $\text{sin}(2\pi x)$，在0~1之間只有兩個 critical points，而太多 critical points 的degree = 14 多項式太過 overfitting 了，使得連 whole data training 也無法達到「model 越強大，可以達到的training error 越小」。

### Q5 

#### m = 10
|cv term|training mse|validation mse|
|-|-|-|
|0 |0.090|0.263|
|1|0.003|0.827|
|2|0.056|466725.143|
|3|0.054|0.009|
|4|0.085|0.065|
|avg|0.058|93345.261|
|whole data|$2.04\times 10^{-14}$|x|

fitline : <img src = "result\sin_m10\p14\whole\fitline.jpg">

#### m = 80
|cv term|training mse|validation mse|
|-|-|-|
|0|0.003|0.007|
|1|0.001|0.003|
|2|0.001|0.002|
|3|0.001|0.001|
|4|0.002|0.006|
|avg|__0.002__|__0.004__|
|whole data|__0.003__|x|

fitline : <img src = "result\sin_m80\p14\whole\fitline.jpg">

#### m = 320
|cv term|training mse|validation mse|
|-|-|-|
|0|0.002|0.002|
|1|0.002|0.002|
|2|0.002|0.003|
|3|0.019|0.015|
|4|0.003|0.003|
|avg|__0.006__|__0.005__|
|whole data|__0.020__|x|

fitline : <img src = "result\sin_m320\p14\whole\fitline.jpg">

#### 討論
從 fitting line 可以看到，越多的樣本，模型大致上訓練的越好。

至於為什麼 320 的比表現會低於 280，我猜測可能跟 degree = 14 本身其實太過 overfitting (Q4 討論中提到的) 有關。資料一多，可能造成的誤差因為點多而更高機率出現 fit 不太到的地方而變大。

還有一點很奇怪 whole data training error 在 m= 10 時候反而最小。
上述討論 (Q3) 時，我的猜測是點不夠，但此情況是 m = 10，比 15 個點還要少，如果 Q3 我的假設是對的，那應該 fit 不起來， m=10 training error 要最大。

我想可能是由於他的資料生成跟 groundtruth ($\text{sin}(2\pi x)$) 很像 ($\epsilon \sim N(0,0.04)$ variance 很小)，degree = 14 的多項式可能剛好有能力精準 fit 到 1 cycle sin wave 的 10 個點，才會突然能 fit 起來。

而像是 m=80, 320這兩個資料集，視覺上幾乎就是 sin wave了，所以反而體現出 polynomial 逼近 1 cycle sin wave 的真正狀態。

### Q6
#### $\lambda=0$
|cv term|training mse|validation mse|
|-|-|-|
|0|0.0034|0.00334|
|1|0.0008|0.0308|
|2|0.0393|17.5389|
|3|0.0080|0.1416|
|4|0.0002|0.0057|
|avg|__0.0103__|__3.5441__|
|whole data|__0.0026__|x|

fitting line:<img src="result\regu_sin_15\l0\p14\whole\fitline.jpg">

#### $\lambda=0.01/15=0.00067$
|cv term|training mse|validation mse|
|-|-|-|
|0|0.0010|0.0051|
|1|0.0011|0.0763|
|2|0.0011|0.0333|
|3|0.0014|0.0017|
|4|0.0013|0.0116|
|avg|__0.0012__|__0.0256__|
|whole data|__0.0013__|x|

fitting line:<img src="result\regu_sin_15\l1\p14\whole\fitline.jpg">

#### $\lambda = 1/15=0.067$

|cv term|training mse|validation mse|
|-|-|-|
|0|0.0970|0.1270|
|1|0.1077|0.0870|
|2|0.0662|0.3249|
|3|0.1051|0.1915|
|4|0.0973|0.1543|
|avg|__0.0947__|__0.1769__|
|whole data|__0.0967__|x|

fitting line:<img src="result\regu_sin_15\l2\p14\whole\fitline.jpg">

#### $\lambda = 1000/15=67$
|cv term|training mse|validation mse|
|-|-|-|
|0|0.4402|0.4498|
|1|0.4406|0.5229|
|2|0.4174|0.5861|
|3|0.4436|0.4494|
|4|0.4759|0.3367|
|avg|0.4435|0.4690|
|whole data|0.4435|x|

fitting line:<img src="result\regu_sin_15\l3\p14\whole\fitline.jpg">

#### 討論

根據我的實驗，$\lambda=0.00067$ 具有最小的 validation mse ，故為最適合的model。

另外，依據fitting line，$\lambda$ 越大，越接近水平線，也就是越傾向於把各 p>0 的項次的 weights 訓練成 0 。

## 結論

除了展示的 result 外，我還有跑好幾次實驗，其趨勢 (cv training/validation , whole data traing mse 的相對大小) 幾乎是相同的 (包含不合理論的情況)。

以上討論中遇到與理論不合的地方只是我依據我的數學知識 + google 猜測，有可能其實只是我的 code 寫錯。

另外，透過親自寫 code 實驗，讓我更加明白 Linear/Polynomial Regression 的運行方式，對於理論的體悟也有很大的幫助。
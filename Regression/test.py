import numpy as np
import pandas as pd
import os

def regression_fit(x, y, degree):
    X = np.power(x, np.arange(degree+1))
    square =X.T@X
    pinv = None
    try:
        pinv=np.linalg.inv(square)
    except:
        pinv = np.linalg.pinv(square)
    w = pinv@X.T@y
    pred = X@w
    mse = np.mean((pred-y)**2)
    return w, mse

def gendata(val_region=(-3,3), datanum=15):
    x = np.linspace(
        val_region[0],val_region[1],datanum, dtype=np.float64
    ).reshape(-1,1)
    y=2*x + np.random.randn(x.shape[0], x.shape[1])
    return x, y


if __name__ == "__main__":
    
    random_indiecs = np.random.permutation(15)
    print(random_indiecs)
    x, y = gendata()
    x_batch = x[random_indiecs[:12]]
    y_batch = y[random_indiecs[:12]]
    
    w0, mse0 = regression_fit(x=x_batch, y=y_batch, degree=10)
    print(f"random 12-p10 mse : {mse0}")
    
    w1, mse1 = regression_fit(x=x_batch, y=y_batch, degree=14)
    print(f"random 12-p14 mse : {mse1}")
    
    w2, mse2 =  regression_fit(x=x[random_indiecs], y=y[random_indiecs], degree=10)
    print(f"whole data-p10 mse : {mse2}")
    
    w3, mse3 =  regression_fit(x=x[random_indiecs], y=y[random_indiecs], degree=14)
    print(f"whole data-p14 mse : {mse3}")

    metrics = pd.DataFrame(
        {
            'term':["random 12-p10","random 12-p14","whole data-p10","whole data-p14"],
            'mse':[mse0, mse1, mse2, mse3]
        }
    )
    """
    testing_result_savepath = os.path.join("result_v1","polyreg","testing")
    if not os.path.exists(testing_result_savepath):
        os.mkdir(testing_result_savepath)
    metrics.to_csv(os.path.join(testing_result_savepath,"cmp.csv"), index=False)
    """


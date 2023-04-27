import os
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from Regressor import HyperPlane as hp
from Regressor import LinearRegressor as linreg
from Regressor import PolynomialRegressor as polyreg
from Regressor import Data1, Data2
from Regressor import reg_kfold_cv
from visualization import plotting2D, plotbarchart

def savearray(a, savedir, savename)->None:
    d = os.path.join(savedir, savename)
    if not os.path.exists(d):
        os.mkdir(d)
    
    np.save(os.path.join(d, savename), a)
    np.savetxt(
        os.path.join(d, f"{savename}.csv"), a, 
        delimiter=","
    )

def LinRegression_with1feature(
    sample:dict, data:dict, model_info:dict, k_folds_cv = 5,
    fitlinetitle="fitline", saveroot=None
):
    
    sampleset = np.hstack((sample['x'],sample['y']))
    dataset = np.hstack((data['x'],data['y']))
    
    cv_result = reg_kfold_cv(
        X=data['x'], Y=data['y'],
        dim=model_info['dim'],model_type=model_info['model'],
        k=k_folds_cv,weight_decacy=model_info['weight_decacy'], 
        record_path=saveroot
    )
    pd.DataFrame({
        'term':list(i for i in range(k_folds_cv))+['avg','whole data'],
        'training mse':cv_result['cv training mse'] + [cv_result['avg cv training mse'],cv_result['whole data mse']],
        'validation mse':cv_result['cv validation mse'] + [cv_result['avg cv val mse'], "n"]
    }).to_csv(os.path.join(saveroot, "metrics.csv"), index=False)

    if k_folds_cv > 1:
        print(f"avg train mse: {cv_result['avg cv training mse']}, avg val mse:{cv_result['avg cv val mse']}")
    print(f"training mse: {cv_result['whole data mse']}")
    for i, model in enumerate(tqdm(cv_result['models'])):
        version = f"cv{i}"
        
        sr = os.path.join(saveroot, f"{i}")
        if i == k_folds_cv:
            version = f"whole"
            sr = os.path.join(saveroot, f"whole")

        plotting2D(
            title=f"{fitlinetitle} {version}",
            plot_info={
                "sample":{"data":sampleset,"method":"plot"},
                "dataset":{"data":dataset,"color" : 'red',"method":"scatter"},
                "model:":{"data":np.hstack((sample['x'],model.f(sample['x']))),"method":"plot"}
            },
            saveplt=os.path.join(sr, "fitline.jpg")
        )
   
    

def LinearRegressionSample(
    sample, data, saveroot, cv=5, weight_decacy=0.0,
    **kwarg
):
    
    savearray(
        np.hstack((data['x'], data['y'])), 
        savedir=saveroot, savename="dataset"
    )
    
    LinRegression_with1feature(
        sample=sample,data=data, 
        model_info={
            'model':linreg, 'dim':data['x'].shape[1],
            'weight_decacy':weight_decacy
        },fitlinetitle=kwarg['fitlinetitle'],
        k_folds_cv=cv,saveroot=saveroot
    )

def PolynomialRegressionSample(
    sample, data, saveroot,poly_term = [1],cv=5,
    weight_decacy=0.0, **kwarg
):

    savearray(
        np.hstack((data['x'], data['y'])), 
        savedir=saveroot, savename="dataset"
    )
    
    for p in poly_term:
        print(f"fit with p{p}")
        savepath = os.path.join(saveroot,f"p{p}")
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        LinRegression_with1feature(
            sample=sample,data=data, 
            model_info={
                'model':polyreg, 'dim':p, 
                'weight_decacy':weight_decacy
            }, fitlinetitle=f"{kwarg['fitlinetitle']} p{p}",
            k_folds_cv=cv,saveroot=savepath
        )

def main():
    
    # Make the dir where results are saved at. 
    resultroot = os.path.join("result_v1")
    if not os.path.exists(resultroot ):
        os.mkdir(resultroot)

    # Q1: Generate the sample y = 2x+eps for x in [-3,3]
    sample1, data1 = Data1(datanum=15, val_region=(-3,3))
    print("="*3,end ="" )
    print("Q2: Using linear regression to fit the linear biased dataset", end="") 
    print("="*3)
    linreg_savepath = os.path.join(resultroot ,"linreg")
    if not os.path.exists(linreg_savepath):
       os.mkdir(linreg_savepath)
    LinearRegressionSample(
        sample=sample1,data=data1,
        saveroot=linreg_savepath,cv=5, 
        fitlinetitle="Linear Regression fitting y=2x+eps"
    )
 
    polyreg_saveroot = os.path.join(resultroot ,"polyreg")
    if not os.path.exists(polyreg_saveroot):
       os.mkdir(polyreg_saveroot)
    print("="*3,end ="" )
    print("Q3: Using Polynomial regression to fit the linear biased dataset", end="")
    print("="*3)
    PolynomialRegressionSample(
        sample=sample1, data=data1,
        saveroot= polyreg_saveroot,
        poly_term=[5,10,14],cv=5,
        fitlinetitle="Polynomial Regression fitting y=2x+eps"
    )
    

    # y = sin(2*pi*x) + eps for x in [0,1]
    sample2, data2 = Data2(datanum=15, val_region=(0,1))
    print("="*3,end ="" )
    print("Q4: Using linear/polynomial regression to fit sin dataset",end="")
    print("="*3)
    sin_saveroot = os.path.join(resultroot ,"sinreg")
    if not os.path.exists(sin_saveroot):
        os.mkdir(sin_saveroot)
    PolynomialRegressionSample(
        sample=sample2, data=data2, poly_term=[1,5,10,14],
        saveroot=sin_saveroot,
        fitlinetitle="Polynomial Regression fitting y=sin(2*pi*x)+eps"
    )
    print("="*3,end ="" )
    print("Q5: Using p = 14 to fit number 10, 80, 320 dataset",end="") 
    print("="*3)
    for m in [10, 80, 320]:
        r = os.path.join(resultroot, f'sin_m{m}')
        print(r)
        if not os.path.exists(r):
            os.mkdir(r)
        
        s, d = Data2(datanum=m, val_region=(0,1))
        PolynomialRegressionSample(
            sample=s, data=d, saveroot=r,poly_term=[14],
            fitlinetitle=f"Polynomial Regression fitting y=sin(2*pi*x)+eps, M={m}"
        )
    print("="*3,end ="")
    print("Q6: using regularization with different lambda", end="")
    print("="*3)
    numdata = data2['x'].shape[0]
    regu_saveroot = os.path.join(resultroot, "regu_sin_15")
    if not os.path.exists(regu_saveroot):
        os.mkdir(regu_saveroot)
    for idx, wd in enumerate([0, 0.001/numdata, 1/numdata,1000/numdata]):
        print(f"{wd:.6f}")
        r = os.path.join(regu_saveroot, f"l{idx}")
        if not os.path.exists(r):
            os.mkdir(r)
        with open(os.path.join(r, "weight_decacy.txt"), "w+") as f:
            f.write(f"{wd}")
        PolynomialRegressionSample(
            sample=sample2, data=data2, 
            saveroot=r,poly_term=[14],
            weight_decacy=wd,
            fitlinetitle=f"Polynomial Regression fitting y=sin(2*pi*x)+eps, wd={wd:.4f}"
        )

if __name__ == "__main__":
    main()

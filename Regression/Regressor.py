import time
from tqdm import tqdm
import os 
import numpy as np

class HyperPlane():

    def __init__(self, weight:np.ndarray, requird_dir=True) -> None:
        """
        weight : 
            A numpy ndarray with dimension (N+1, 1) 
            for the features with N dimensions.
        requird_dir:
            if Truemay let weight may *(-1) 
            to make weight[1] (i.e. coefficient of x1) > 0
        """
        self.weight = weight.astype(np.float64)
        if requird_dir :
            if self.weight[1][0] < 0:
                self.weight = -self.weight
        self.dim = weight.shape[0] - 1

    def _expand_ones(self, x:np.ndarray)->np.ndarray:

        #expanding one column vector on the first index of
        #the column of x as the constant term.

        return np.hstack(
            [np.ones((x.shape[0], 1)), x]
        ).astype(np.float64)

    def f(self, x:np.ndarray)->np.ndarray:
        """
        x :
            A numpy ndarray for a set of $X$
            in row form
        
        will return x_@self.weight such that 
        x_ is x with expanded extra "ones" columns
        if the dimensions of x is N, 
        otherwise return x@self.weight directly.

        f(x) = w0*1 +  w1*x1 + w2*x2 + ... + wn*xn
        """
        
        if x.shape[1] == self.dim+1:
            return x@self.weight
        return self._expand_ones(x)@self.weight

    def makepoints(self, Xn_1:np.ndarray, concatation=False)->np.ndarray:
        """
        Will using the equation:
        x_n = - [(weight[0])*1 +(weight[1])*x_1 + ... + \
            weight[n-1]*x_(n-1) ] /weight[n]
        to make the points.

        Please notice the error of the floating point 
        operation.
        """
        Xn_1_exp = (Xn_1)
        if Xn_1.shape[1] == self.dim-1:
            """
            hasn't expand yet 
            """
            Xn_1_exp = self._expand_ones(Xn_1)
        Xn = -(Xn_1_exp @ (self.weight[:-1]))/self.weight[-1][0]
        if not concatation:
            return Xn
        else:
            if Xn_1.shape[1] == self.dim-1:
                return np.hstack([Xn_1,Xn])
            else:
                return np.hstack([Xn_1[:, 1:], Xn])

    def save_weight(self, savedir, savename, filetypes:list=['npy', 'csv'])->None:
        d = os.path.join(savedir, savename)
        if not os.path.exists(d):
            os.mkdir(d)
        for ft in filetypes:
            if ft == 'npy':
                np.save(os.path.join(d, savename), self.weight)
            elif ft=='csv':
                np.savetxt(os.path.join(d, f"{savename}.csv"), self.weight,delimiter=",")
            else:
                print("Not support yet")
        
class LinearRegressor(HyperPlane):
    
    def __init__(self, feature_dim:int, w_init:np.ndarray=None) -> None:
        if w_init is not None:
            if feature_dim+1 == w_init.shape[0]:
                super().__init__(weight = w_init, requird_dir=False)
            else:
                raise ValueError(f"required : {feature_dim}+1, given weights : {w_init.shape[0]}")
        else:
            super().__init__(
                weight=np.random.randn(feature_dim+1, 1)
            )

    def train(self, x:np.ndarray, Y:np.ndarray, weight_decacy:float=0.0) -> float:
        
        X = x
        if x.shape[1] != self.dim+1:
            X = self._expand_ones(x)
            
        xsqure = X.T@X
        reg_term = xsqure + weight_decacy*np.eye(xsqure.shape[0], dtype=np.float64)
        pinv_X = None
        try:
            pinv_X = np.linalg.inv(reg_term)
        except:
            pinv_X = np.linalg.pinv(reg_term)
        self.weight = pinv_X@(X.T)@Y

        mse = self.mse(X, Y)

        return mse
        
    def mse(self, X, Y)->float:
        pred = self.f(X)
        return np.mean((pred-Y)**2)
        
class PolynomialRegressor(LinearRegressor):
    
    def __init__(self, feature_dim: int, w_init: np.ndarray = None) -> None:
        super().__init__(feature_dim=feature_dim, w_init=w_init)
        
    def f(self, x:np.ndarray)->np.ndarray:

        if x.shape[1] != self.dim+1:
            
            poly_term = np.power(x, np.arange(self.dim+1))
            return poly_term@self.weight
        else:
            return x@self.weight
    
    def train(self, x: np.ndarray, Y: np.ndarray, weight_decacy=0.0) -> float:
        
        X = np.power(x, np.arange(self.dim+1))
        return super().train(X, Y,weight_decacy=weight_decacy)


def k_fold_split_dataset(num, k)->list:
    data_indiecs = np.arange(0, num)
    random_indiecs = np.random.permutation(data_indiecs)
    fold_size = num//k
    folds = np.zeros((k, fold_size)).tolist()
    if num%k == 0:
        folds = random_indiecs.reshape(k, -1).tolist()
    else:
        for i in range(0, num, fold_size+1):
            folds[i//(fold_size+1)]=(random_indiecs[i:i+fold_size+1].tolist())
    
    training_validation_idx = list({} for _ in range(k))
    
    for i in range(len(folds)):
        training_validation_idx[i]['train'] = np.concatenate(
            list(np.array(folds[j]) for j in range(k) if j != i)
        )
        training_validation_idx[i]['valid'] = np.array(folds[i])
    
    return training_validation_idx

def reg_kfold_cv(
    X:np.ndarray, Y:np.ndarray,dim:int, k:int, model_type,
    weight_decacy:float=0.0, record_path=None
)->dict:
    
    ret = {'models':[],'whole data mse':-1}
    if k > 1:
        tv = k_fold_split_dataset(num=X.shape[0], k=k)
        ret['cv training mse']=[]
        ret['cv validation mse']=[]
        for ki, fold in enumerate(tqdm(tv)):
            model = model_type(dim)
            t_mse = model.train(
                x=X[fold['train']],Y=Y[fold['train']],
                weight_decacy=weight_decacy
            )
            ret['cv training mse'].append(t_mse)
            val_mse = model.mse(X=X[fold['valid']],Y=Y[fold['valid']])
            ret['cv validation mse'].append(val_mse)
            ret['models'].append(model)
            if record_path is not None:
                r = os.path.join(record_path, f"{ki}")
                if not os.path.exists(r):
                    os.mkdir(r)
                model.save_weight(r, "w")
                np.savetxt(
                    os.path.join(r,'batch.csv'),
                    np.hstack((X[fold['train']],Y[fold['train']])),
                    delimiter=','
                )
        ret['avg cv training mse'] = np.mean(ret['cv training mse'])
        ret['avg cv val mse'] = np.mean(ret['cv validation mse'])
    wholemodel = model_type(dim)
    whole_mse = wholemodel.train(X,Y,weight_decacy) 
    whole_model_savepath = os.path.join(record_path, "whole")
    if not os.path.exists(whole_model_savepath):
        os.mkdir(whole_model_savepath)
    wholemodel.save_weight(whole_model_savepath, "w")
    ret['models'].append(wholemodel)
    ret['whole data mse'] = whole_mse
    return ret
    
def Data1(datanum=10, val_region=(0,1))->tuple:
    """
    y = 2*x

    0*1 + 2*x1 + (-1)*x2 = 0
    norm vector: (0, 2, -1)
    """
    s_num = int(round((val_region[1]-val_region[0])/0.01))

    x = np.linspace(val_region[0],val_region[1],datanum, dtype=np.float64).reshape(-1,1)
    
    sample_hp = HyperPlane(weight=np.array([[0],[2],[-1]]))
    samplex = np.linspace(val_region[0],val_region[1],s_num).reshape(-1,1)
    sample = {
        'x':samplex,
        'y':sample_hp.makepoints(samplex,concatation=False)
    }
    
    y0 = sample_hp.makepoints(x,concatation=False).astype(np.float64)
    biasedy =y0+np.random.randn(y0.shape[0],y0.shape[1]).astype(np.float64)
    data = {'x':x, 'y':biasedy}

    return sample, data

def Data2(datanum=10, val_region=(0,1))->tuple:
    """
    y =sin(2*pi*x)
    """
    def f(x):
        return np.sin(2*np.pi*x)

    s_num = int(round((val_region[1]-val_region[0])/0.01))

    x = np.linspace(val_region[0],val_region[1],datanum, dtype=np.float64).reshape(-1,1)
    samplex = np.linspace(val_region[0],val_region[1],s_num).reshape(-1,1)
    sampley = f(samplex)
    

    y0 = f(x)
    sample = {'x':samplex, 'y':sampley}
    biasedy = y0+np.random.normal(loc=0, scale=0.04, size=y0.shape).astype(np.float64)
    data = {'x':x, 'y':biasedy}
    return sample, data

if __name__ == "__main__":
    pass

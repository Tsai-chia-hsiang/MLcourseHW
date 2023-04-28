import numpy as np
import os
from tqdm import tqdm
from Dataset import LIBSVM_Dataset
from libsvm.svmutil import *

def makedir(d):
    if not os.path.exists(d):
        os.mkdir(d)
    return d


def generate_data(region=(-100, 100), N=500, gaussian_noise_para=(0.0,1.0)):
    
    """
    y = 2x+eps~Normal(guassian_noise_para)
    """
    print("data information:")
    print(f"N: {N}, xregion:{region[0]}~{region[1]}")
    print(f"N({gaussian_noise_para[0]}, {gaussian_noise_para[1]})")
    
    x = np.linspace(region[0], region[1], N, dtype=np.float64).reshape(-1,1)
    eps = np.random.normal(
        loc=gaussian_noise_para[0], 
        scale=gaussian_noise_para[1],
        size=x.shape
    )
    label = (eps >= 0).astype(np.int16)
    biased = 2*x+ eps
    return {'data':np.hstack((x,biased)),'label':label}

def main(saveroot):
    
    datasaveroot = makedir(os.path.join(saveroot, "data"))
    
    dataset = LIBSVM_Dataset()
    data = dataset.get_data(fromfile=datasaveroot)
    
    """
    dataset = LIBSVM_Dataset(generator=generate_data)
    data = dataset.gen(
        k_fold=5, savepath=datasaveroot,
        region=(-100, 100), N=500, gaussian_noise_para=(0.0,1.0)
    )
    """
    
    for k,v in data.items():
        print(k)
        print("="*10)
        if isinstance(v, list):
            for vi in v:
                for ki, vii in vi.items():
                    print(f"    {ki}")
                print("   ","="*10)
            
        


if __name__ == "__main__":
    saveroot = os.path.join("result")
    if not os.path.exists(saveroot):
        os.mkdir(saveroot)
        
    main(saveroot=saveroot)
import os
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['text.usetex'] = True


def plotting2D(
    title:str,plot_info:dict,
    saveplt:os.PathLike=None,notebook=False
):
    """
    support ```plot``` and ```scatter```
    
    plot_info:
        key: the label

        value: the dataset, is a dict: 
            key: method, value: how to plot it
            key: data, value: the data, a np (N,2) array
            key: label, value: str, it's label
    """
    
    plt.figure(dpi = 800)
    for k, v in plot_info.items():
        
        data = v['data']
        if v['method'] == "plot":
            plt.plot(
                data[:, 0], data[:, 1], label=k,
                linewidth=0.8
            )
        elif v['method'] == "scatter":
            plt.scatter(
                data[:, 0],data[:, 1],s = 1.5, c=v['color'],
                label = k, alpha=0.3
            )
    
    plt.title(title)
    plt.legend()

    if saveplt is not None:
        plt.savefig(saveplt)
    
    if notebook:
        plt.show()    
    else:
        plt.close()

def plotbarchart(dataset:dict, title, saveplt=None, notebook=False):
    
    criteria = {
        'mean':np.mean,
        'max':np.max,
        'min':np.min
    }
    w = 0
    plt.figure(dpi=800)
    for label, data in dataset.items():
        plt.bar(
            data['x']+w, data['y'], width=0.3,
            color=data['color'], label=label,alpha=0.3
        )
        w += 0.3
        if 'criteria' in data:
            v = criteria[data['criteria']](data['y'])
            plt.plot(
                np.concatenate((data['x'],[data['x'].shape[0]])), 
                v*np.ones(data['x'].shape[0]+1),
                c=data['color'], linestyle="--",alpha=0.5,
                label=f"{label} {data['criteria']}:{v:.3f}"
            )
    
    plt.legend()
    plt.title(title)
    
    if saveplt is not None:
        plt.savefig(saveplt)
    
    if notebook:
        plt.show()    
    else:
        plt.close()
  
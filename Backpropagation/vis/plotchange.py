import matplotlib.pyplot as plt
import os
import numpy as np

def plot_changes(
    l:list, name:list, title:str, 
    saveto:os.PathLike=None, show=False
):
    plt.figure(dpi=800)
    for li, namei in zip(l, name):
        xs = np.arange(len(li))
        plt.plot(xs, li, label=namei)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    if saveto is not None:
        plt.savefig(saveto)
    if show:
        plt.show()

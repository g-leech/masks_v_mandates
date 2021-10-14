import datetime

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set(style="ticks", font='DejaVu Serif')

def transform(rs):
    return (1 - rs) * 100

def main_result_posteriors(m, w, ax):
    wred = w.WearingReduction
    wred = transform(wred)
    if type(wred[0]) == np.ndarray:
        wred = [x[0] for x in wred]
    
    sns.kdeplot(wred, label="wearing", shade=True, ax=ax)

    mred = m.MandateReduction 
    mred = transform(mred)
    if type(mred[0]) == np.ndarray:
        mred = [x[0] for x in mred]
    
    sns.kdeplot(mred, label="mandate", color="green", shade=True, ax=ax)
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_xlabel("% R reduction\n(entire population masked)", fontsize=10)
    ax.set_ylabel("Posterior density", fontsize=10)
    ax.set_xlim(-20, 60)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    ax.axes.get_yaxis().set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    #plt.show()

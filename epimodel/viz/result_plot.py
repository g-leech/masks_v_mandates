import datetime

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

sns.set_style("white")

def transform(rs):
    return (1 - rs) * 100

def main_result_posteriors(m, w, ax):
    wred = w.WearingReduction
    wred = transform(wred)
    if type(wred[0]) == np.ndarray:
        wred = [x[0] for x in wred]
    
    sns.kdeplot(wred, label="wearing", shade=True)

    mred = m.MandateReduction 
    mred = transform(mred)
    if type(mred[0]) == np.ndarray:
        mred = [x[0] for x in mred]
    
    sns.kdeplot(mred, label="mandate", color="green", shade=True)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.xlabel("Inferred reduction in R", fontsize=16)
    plt.xlim(-20, 60)
    ax.axes.get_yaxis().set_visible(False)
    plt.legend(fontsize=16, frameon=False)
    #plt.show()

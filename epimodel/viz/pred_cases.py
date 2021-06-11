import calendar

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
import matplotlib.dates as mdates

sns.set(style="ticks", font='DejaVu Serif')

def month_to_str(i):
    return calendar.month_name[i]


def produce_CIs(array):
    m = np.median(array, axis=0)
    li = np.percentile(array, 2.5, axis=0)
    ui = np.percentile(array, 97.5, axis=0)
    uq = np.percentile(array, 75, axis=0)
    lq = np.percentile(array, 25, axis=0)

    return m, li, ui, lq, uq


def get_predictions(trace, r_i):
    c = trace.ExpectedCases[:, r_i]
    nS, nDs = c.shape

    noise = np.repeat(trace.Psi.reshape((nS, 1)), nDs, axis=-1)
    dist = pm.NegativeBinomial.dist(mu=c, alpha=noise)
    output = dist.random()

    means_expected_cases, l, u, _, _ = produce_CIs(output)

    return means_expected_cases, l, u


def plot_predictions(data, means, l, u, ax):
    days = data.Ds
    days_x = np.arange(len(days))

    ax.plot(days_x, means, label="Predicted Cases", zorder=2, color="tab:blue")

    ax.fill_between(days_x, l, u, alpha=0.1, color="tab:blue", linewidth=0)


def plot_actuals(data, full_data, r_i, ax):
    days = data.Ds
    days_x = np.arange(len(days))
    #days_x = days[:-20]
    cases = data.NewCases[r_i, :]

    ax.scatter(
        days_x,
        cases,
        label="Actual Cases",
        marker="o",
        s=10,
        color="tab:red",
        alpha=0.9,
        zorder=3,
    )
    
    
    days_x = np.arange(len(days))[-20:]
    #days_x = days[-20:]
    r = data.Rs[r_i]
    y = full_data[full_data.CountryName == r].ConfirmedCases.diff().tail(20)
    ax.scatter(
        days_x,
        y,
        label="Heldout Cases",
        marker="o",
        s=10,
        color="tab:red",
        alpha=0.9,
        zorder=3,
        facecolors='none', 
        edgecolors='r',
        linewidths=1
    )


def epicurve_plot(data, oxcgrt, Ds, trace, region, ax=None, leg=False):
    if not ax:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=500)
    
    r_i = list(data.Rs).index(region)
    means, l, u = get_predictions(trace, r_i)

    plot_predictions(data, means, l, u, ax)
    plot_actuals(data, oxcgrt, r_i, ax)

    ax.set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep"])
    ax.set_xticks([3, 33, 63, 93, 123])
    
    ax.set_yscale("log")
    ax.set_title(region, fontsize=12)
    
    if leg:
        ax.legend(fontsize=8, frameon=False)

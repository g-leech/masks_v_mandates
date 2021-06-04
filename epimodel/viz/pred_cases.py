import calendar

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns

import epimodel.pymc3_models.base_model as bm

sns.set_style("white")


def month_to_str(i):
    return calendar.month_name[i]


def get_predictions(trace, r_i):
    c = trace.ExpectedCases[:, r_i]
    nS, nDs = c.shape

    noise = np.repeat(trace.Psi.reshape((nS, 1)), nDs, axis=-1)
    dist = pm.NegativeBinomial.dist(mu=c, alpha=noise)
    output = dist.random()

    means_expected_cases, l, u, _, _ = bm.produce_CIs(output)

    return means_expected_cases, l, u


def plot_predictions(data, means, l, u, ax):
    days = data.Ds
    days_x = np.arange(len(days))

    ax.plot(days_x, means, label="Predicted Cases", zorder=2, color="tab:blue")

    ax.fill_between(days_x, l, u, alpha=0.1, color="tab:blue", linewidth=0)


def plot_actuals(data, full_data, r_i, ax):
    days = data.Ds
    days_x = np.arange(len(days))
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


#     ax.set_yscale("log")
#     #plt.ylim([10 ** 0, 10 ** 6])
#     locs = np.arange(start_d_i, end_d_i, 14)
# #     xlabels = [f"{data.Ds[ts].day}-{month_to_str(data.Ds[ts].month)}" for ts in locs]
# #     plt.xticks(locs, xlabels, rotation=-30, ha="left")
# #     plt.xlim((start_d_i, end_d_i))
#     ax=plt.gca()
#     #bm.add_cms_to_plot(ax, data.ActiveCMs, r_i, start_d_i, end_d_i, data.Ds, cm_plot_style)
#     plt.title(region, fontsize=12)
#     return ax


def epicurve_plot(data, oxcgrt, trace, region, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(8,5), dpi=500)
    r_i = list(data.Rs).index(region)
    means, l, u = get_predictions(trace, r_i)

    plot_predictions(data, means, l, u, ax)
    plot_actuals(data, oxcgrt, r_i, ax)

    ax.set_yscale("log")
    # bm.add_cms_to_plot(ax, data.ActiveCMs, r_i, start_d_i, end_d_i, data.Ds, cm_plot_style)
    ax.set_title(region, fontsize=12)
    ax.legend(fontsize=10)
    #plt.show()

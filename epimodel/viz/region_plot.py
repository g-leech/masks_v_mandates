import argparse
import calendar
import copy
import datetime
import pickle
import re
from datetime import timedelta
from pathlib import Path

import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("white")


locator = mdates.MonthLocator()  # every month
# Specify the format - %b gives us Jan, Feb...
fmt = mdates.DateFormatter("%b")


cols = sns.cubehelix_palette(3, start=0.2, light=0.6, dark=0.1, rot=0.2)


def reprod_plot(
    trace,
    data,
    mobility_data,
    oxcgrt,
    third_party_rts,
    r,
    start_d_i=10,
    cis=True,
    wearing=True,
    second="wearing",
    ax=None
):
    plt.rc("font", size=8)
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)

    r_i = data.Rs.index(r)
    data.Ds = pd.to_datetime(data.Ds)
    Ds = data.Ds
    if not ax:
        plt.figure(figsize=(8, 8), dpi=150, constrained_layout=False)
    
    plt.subplot(4, 3, 1)
    
    if not ax :
        ax = plt.gca()
    
    add_rt_plot(ax, r, r_i, trace, data, start_d_i, cis)
    ax.set_xlim(Ds[0], Ds[-20])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    plt.subplot(4, 3, 2)
    ax = plt.gca()
    ax2 = add_wearing_to_plot(ax, r_i, data)
    ax2.set_xlim(Ds[0], Ds[-20])

    plt.subplot(4, 3, 4)
    ax = plt.gca()
    add_cases_to_plot(ax, r_i, data)
    ax.set_xlim(Ds[0], Ds[-20])

    plt.subplot(4, 3, 5)
    ax = plt.gca()
    mobRs = mobility_data.index.get_level_values("country").unique()
    if r in mobRs:
        add_mobilities_to_plot(ax, mobility_data, r, r_i, Ds)
        ax.set_xlim(Ds[0], Ds[-20])

    if "GrowthNoiseScale" in trace.varnames:
        plt.subplot(4, 3, 6)
        ax = plt.gca()
        add_gnoise_to_plot(ax, trace, r_i, Ds)

    plt.subplot(4, 3, 8)
    ax = plt.gca()
    add_oxcgrt_indices(ax, oxcgrt, r, national=False)
    add_diffs_to_plot(ax, r, oxcgrt, Ds)
    ax.set_xlim(Ds[0], Ds[-20])

#     plt.subplot(4, 3, 8)
#     ax = plt.gca()
#     add_preds(ax, trace, data, r_i)
#     # plt.legend()

    if r in third_party_rts.region.unique():
        plt.subplot(4, 3, 7)
        ax = plt.gca()
        add_third_party_rt(ax, third_party_rts, r, Ds)
        ax.set_xlim(Ds[0], Ds[-20])

#     plt.subplot(4, 3, 10)
#     # add_oxcgrt_indices(ax, oxcgrt, r, national=True)
#     if "r_walk_noise" in trace.varnames:
#         ax = plt.gca()
#         walknoises = trace.r_walk_noise.mean(axis=0)
#         multipliers = np.exp(walknoises)
#         add_random_walks(ax, multipliers, r_i)

#         plt.subplot(4, 3, 11)
#         ax = plt.gca()
#         rt_walks = trace.Rt_walk.mean(axis=0)
#         rt_cms = trace.Rt_cm.mean(axis=0)
#         add_separate_cms_rw(ax, rt_cms, rt_walks, data.Ds, r_i)

    plt.tight_layout()


def add_random_walks(ax, ms, r_i):
    mr = ms[r_i, :]
    ax.plot(mr)
    plt.ylabel("Random walk multiplier", fontsize=10)
    plt.xlabel("week num", fontsize=10)


def add_separate_cms_rw(ax, cms, rws, ds, r_i):
    # plt.plot(ds, cms[r_i, :], alpha=0.5, label="R0 - CMs")
    plt.plot(ds, rws[r_i, :], alpha=0.5, label="RW")

    plt.ylabel("Rt", fontsize=10)
    plt.legend(fontsize=10, frameon=False)
    # ax.set_ylim([0.5, 1.5])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def get_end_ind(data, i):
    if len(np.nonzero(data.NewCases.mask[i, :])[0]) > 0:
        return np.nonzero(data.NewCases.mask[i, :])[0][
            -1
        ]  # np.nonzero(data.NewCases.mask[i, :])[0][0]+3
    else:
        return len(data.Ds)


# def get_rt(trace, data, r_i=None, wearing=True) :
#     nS, nCMs = trace.CM_Alpha.shape
#     nDs = len(data.Ds)
#     ActiveCMs = data.ActiveCMs.copy()
#
#     if wearing:
#         alphas = np.concatenate([trace.CM_Alpha, trace.Wearing_Alpha], axis=1)
#         nWearing = trace.Wearing_Alpha.shape[1]
#         nCMs += nWearing
#
#         if nWearing == 2:
#             wi = data.CMs.index("percent_mc")
#             ws = ActiveCMs[:, wi, :].reshape(len(data.Rs), 1, nDs)
#             ActiveCMs = np.concatenate([ActiveCMs, ws], axis=1)
#
#     else :
#         alphas = np.concatenate([trace.CM_Alpha], axis=1)
#
#     if r_i is not None :
#         ActiveCMRed = np.sum(ActiveCMs[r_i, :, : ].reshape((1, nCMs, nDs)) * alphas.reshape((nS, nCMs, 1)), axis=1)
#         RegionR = trace['RegionR'][:, r_i]
#     else :
#         ActiveCMRed = np.sum(ActiveCMs * alphas, axis=1)
#         RegionR = trace['RegionR']
#
#     return np.exp(np.log(RegionR.reshape((nS, 1))) - ActiveCMRed)


def get_expected_rt(trace, r_i):
    ELogR = trace["ExpectedLogR"][:, r_i]

    return np.exp(ELogR)


def get_rt(trace, data, r_i=None):
    nS, nCMs = trace.CM_Alpha.shape
    nDs = len(data.Ds)
    elogr = trace.ExpectedLogR
    expectedr = np.exp(elogr)

    return expectedr[:, r_i, :].reshape((nS, nDs))


def add_cases_to_plot(ax, r_i, data):
    # wi = data.CMs.index("percent_mc")
    cases = data.NewCases[r_i, :]
    Ds = data.Ds

    # ax2 = ax.twinx()
    ax.plot(Ds, cases, alpha=0.55, label="cases", color="blue")
    ax.set_ylabel("cases", fontsize=10)
    ax.set_xlim([Ds[0], Ds[-1]])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def add_deaths_to_plot(ax, r_i, data):
    deaths = data.NewDeaths[r_i, :]
    Ds = data.Ds
    ax.plot(Ds, deaths, alpha=0.55, label="deaths", color="green")
    ax.set_ylabel("deaths", fontsize=10)
    ax.set_xlim([Ds[0], Ds[-1]])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def add_gnoise_to_plot(ax, trace, r_i, Ds):
    noise = trace.GrowthCasesNoise.mean(axis=0)[r_i, :]
    ax.plot(Ds, noise, alpha=0.55, label="growth noise", color="purple")
    ax.set_ylabel("growth noise", fontsize=10)
    ax.set_ylim([-0.15, 0.15])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def add_rt_plot(ax, r, r_i, trace, data, start_i, cis=True):
    rs = get_rt(trace, data, r_i=r_i)

    end_i = get_end_ind(data, r_i)
    Ds = pd.to_datetime(data.Ds)
    start_d = Ds[start_i]
    end_d = Ds[end_i]
    mns, lu, up, _, _ = produce_CIs(rs)

    plt.title(r, fontsize=12)

    ax.plot(Ds[10:-20], mns[10:-20], color=cols[0], label="R")
    ax.tick_params(axis="y", colors=cols[0])

    if cis:
        plt.fill_between(Ds[10:-20], lu[10:-20], up[10:-20], alpha=0.25, linewidth=0)

    plt.xlim((start_d, end_d))
    plt.ylim([0.5, 1.5])
    plt.ylabel("Estimated $R_t$", fontsize=10)

    ax.plot([start_d, end_d], [1, 1], color=cols[1], linestyle="--")

    #ax3 = add_cms_to_plot(ax, data, r_i, start_i, end_i)


def add_preds(ax, trace, data, r_i):
    lu, m, ul = np.percentile(trace.ExpectedCases[:, r_i, :], [2.5, 50, 97.5], axis=0)
    ax.plot(data.Ds, m)
    ax.fill_between(data.Ds, lu, ul, alpha=0.3)
    ax.scatter(data.Ds, data.NewCases[r_i, :], label="actual new", s=2)
    ax.set_ylabel("Pred. cases", fontsize=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def add_third_party_rt(ax, third_party_rts, r, Ds):
    rts = third_party_rts[third_party_rts.region == r]
    rts.date = pd.to_datetime(rts.date)
    rts = rts[rts.date.isin(Ds)]
    m = rts["mean_minus10"].interpolate()[10:-20]
    #lu = rts.lower_80[10:-20]
    #ul = rts.upper_80[10:-20]
    ds = rts.date.unique()
    plt.xlim(ds[0], ds[-1])
    plt.ylim(0.5, 1.5)
    
    ds = ds[10:-20]

    ax.plot(ds, m)
    plt.axhline(y=1, color="black", linestyle="--")
    ax.set_ylabel("epifor Rt estimate", fontsize=10)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def add_oxcgrt_indices(ax, oxcgrt, r, national=False):
    if national:
        col = "StringencyIndexNational"
        s = "national"
    else:
        col = "StringencyIndex"
        s = ""

    oxr = oxcgrt[oxcgrt.CountryName == r]
    ax.plot(oxr.date, oxr[col])
    ax.set_ylabel(s + " OxCGRT stringency", fontsize=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def get_npi_names(df):
    cs = range(1, 9)
    # ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "H1"]
    npis = []

    for i in cs:
        npi = [c for c in df.columns if f"C{i}" in c]
        npi = [c for c in npi if f"Flag" not in c][0]
        npis += [npi]

    npis += ["H1_Public information campaigns"]

    return npis


def add_diffs(df):
    npis = get_npi_names(df)
    Rs = df.CountryName.unique()
    df = df.set_index("CountryName")

    for c in Rs:
        # df.loc[c, "H6_diff"] = df.loc[c]["H6_Facial Coverings"].diff()

        for npi in npis:
            i = npi[:2]
            df.loc[c, f"{i}_diff"] = df.loc[c][npi].diff()
            if f"{i}_Flag" in df.columns:
                df.loc[c, f"{i}_flag_diff"] = df.loc[c][f"{i}_Flag"].diff()

    return df.reset_index()


def add_diffs_to_plot(ax, r, df, Ds):
    rdf = df[df.CountryName == r]
    diffed = add_diffs(rdf)
    diffcols = [c for c in diffed.columns if "diff" in c and "flag" not in c]
    diffed = diffed[diffcols + ["CountryName", "date"]]

    for i, c in enumerate(diffcols):
        changes = diffed[c].to_numpy().nonzero()[0]
        plot_npi_diff(ax, changes, c, i, Ds)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def plot_npi_diff(ax, changes, col, i, Ds):
    all_heights = np.zeros(len(Ds))
    ax2 = ax.twinx()
    ax2.yaxis.set_ticks([])
    for c in changes:
        ax2.plot(
            [Ds[c], Ds[c]],
            [0, 1],
            "--",
            color="lightgrey",
            linewidth=1,
            zorder=-2,
            alpha=1,
        )
        plot_height = 1 - (0.1 * i)

        if Ds[c] > Ds[0]:
            ax2.text(
                Ds[c],
                plot_height,
                col[:2],  # plot_style[name],
                # fontproperties=fp2,
                # color=plot_style[cm][1],
                size=8,
                va="center",
                ha="center",
                clip_on=True,
                zorder=1,
            )


def produce_CIs(array):
    """
    Produce 95%, 50% Confidence intervals from a Numpy array, taking CIs using the 0th axis.

    :param array: Numpy array from which to compute CIs.
    :return: (median, 2.5 percentile, 97.5 percentile, 25th percentile, 75th percentile) tuple.
    """
    m = np.median(array, axis=0)
    li = np.percentile(array, 2.5, axis=0)
    ui = np.percentile(array, 97.5, axis=0)
    uq = np.percentile(array, 75, axis=0)
    lq = np.percentile(array, 25, axis=0)
    return m, li, ui, lq, uq


cm_plot_style = {
    "C1_School closing": "S",
    "C2_Workplace closing": "W",
    "C4_Restrictions on gatherings_3plus": "G3",
    "C6_Stay at home requirements": "L",
    "C7_Restrictions on internal movement": "I",
    "C4_Restrictions on gatherings_2plus": "G2",
    "C4_Restrictions on gatherings_full": "G4",
}


def add_wearing_to_plot(ax, r_i, data):
    wi = data.CMs.index("percent_mc")
    wearing = data.ActiveCMs[r_i, wi, :]

    # ax2 = ax.twinx()
    ax.plot(data.Ds, wearing, alpha=0.55, label="wearing", color="blue")
    ax.set_ylim([0, 1])
    ax.set_ylabel("% wearing", fontsize=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    return ax


def add_mobilities_to_plot(ax, mob, r, r_i, Ds):
    mob_pub = mob.loc[r, "avg_mobility_no_parks_no_residential"]
    # mob_res = mob.loc[r, "residential_percent_change_from_baseline"]
    rds = pd.to_datetime(mob.loc[r].index)
    ax.plot(rds, mob_pub, alpha=0.55, label="public", color="blue")
    # ax.plot(mob.loc[r].index, mob_res, alpha=0.55, label="residential", color="green")

    ax.set_ylabel("mobility (% max)", fontsize=10)
    ax.set_ylim([0, 1])
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    return ax


def add_cms_to_plot(ax, data, country_indx, min_x, max_x):
    """
    Plotter helper.

    This takes a plot and adds NPI logos on them.

    :param ax: axis to draw
    :param ActiveCMs: Standard ActiveCMs numpy array
    :param country_indx: Country to pull CM data from
    :param min_x: x limit - left
    :param max_x: x limit - right
    :param days: days to plot
    :param plot_style: NPI Plot style
    """
    ActiveCMs = data.ActiveCMs
    Ds = data.Ds
    ax2 = ax.twinx()
    ax2.set_ylim([0, 1])
    # plt.xlim([min_x, max_x])

    CM_names = data.CMs.copy()

    # Remove wearing
    if "percent_mc" in data.CMs:
        i = data.CMs.index("percent_mc")

        if i == len(data.CMs) - 1:
            ActiveCMs = ActiveCMs[:, :-1, :]

        CM_names.remove("percent_mc")

    if "avg_mobility_no_parks_no_residential" in data.CMs:
        i = data.CMs.index("avg_mobility_no_parks_no_residential")
        ActiveCMs = np.delete(ActiveCMs, i, 1)
        CM_names.remove("avg_mobility_no_parks_no_residential")

    CMs = ActiveCMs[country_indx, :, :]
    nCMs, _ = CMs.shape
    CM_changes = np.zeros((nCMs, len(Ds)))
    CM_changes[:, 1:] = CMs[:, 1:] - CMs[:, :-1]
    all_CM_changes = np.sum(CM_changes, axis=0)
    all_heights = np.zeros(all_CM_changes.shape)

    for cm, name in enumerate(CM_names):
        changes = np.nonzero(CM_changes[cm, :])[0].tolist()

        height = 1
        for c in changes:
            close_heights = all_heights[c - 3 : c + 4]
            if len(close_heights) == 7:
                height = np.max(close_heights) + 1
                all_heights[c] = height

            ax2.plot(
                [Ds[c], Ds[c]],
                [0, 1],
                "--",
                color="lightgrey",
                linewidth=1,
                zorder=-2,
                alpha=1,  # 0.5,
            )
            plot_height = 1 - (0.04 * height)

            if c < min_x:
                c_p = min_x
            else:
                c_p = c

            # if CM_changes[cm, c] == 1:
            ax2.text(
                Ds[c_p],
                plot_height,
                name[:2],
                # fontproperties=fp2,
                # color=plot_style[cm][1],
                size=8,
                va="center",
                ha="center",
                clip_on=True,
                zorder=1,
            )

    plt.yticks([])
    return ax2

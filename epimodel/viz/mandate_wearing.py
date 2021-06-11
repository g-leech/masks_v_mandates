import copy
import datetime
import pickle
import re
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick

sns.set(style="ticks", font='DejaVu Serif')

Ds = pd.date_range("2020-05-01", "2020-09-21", freq="D")

PNAS_WIDTH_INCHES = 3.4252


def centre_on_mandate(cdf, s):
    on = s.head(1)
    zeroday = pd.to_datetime(on.index[0])
    l = max(zeroday - timedelta(days=30), Ds[0])
    r = min(zeroday + timedelta(days=30), Ds[-1])
    dates = pd.date_range(l, r)

    y = cdf.loc[dates]["percent_mc"].values
    n = len(dates) // 2
    x = range(-n, len(y) - n, 1)

    return x, y


def centred_mandate_plot_country(cdf, s):
    x, y = centre_on_mandate(cdf, s)
    plt.plot(x, y, alpha=0.3)


def get_mandate_switches(cdf, c):
    return cdf[cdf["H6_Facial Coverings"].diff() == 1]


def messy_centred_mandate_plot(Rs, df):
    for c in Rs:
        cdf = df.loc[c]
        cdf.index = pd.to_datetime(cdf.index)
        switches_on = get_mandate_switches(cdf, c)

        if switches_on["H6_Facial Coverings"].any():
            centred_mandate_plot_country(cdf, switches_on)

    plt.xlabel("Days since mandate")
    plt.ylim(0, 1)
    plt.ylabel("% mask wearing")

    plt.plot([0] * 2, [0, 1], alpha=1, color="black", linestyle="--")
    plt.show()


def country_centred_mandate_plot(Rs, df, tsize=10):
    nCols = 7
    summaries = get_centred_summary(Rs, df)
    Rs = summaries.country.unique()
    nRs = len(Rs)
    nRows = int(np.ceil(nRs/nCols))
    
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, sharex=True, sharey=True, figsize=(10, nRows*1.4), dpi=500)
    i = 0
    for row in axes:
        for ax in row:
            if len(Rs) <= i :
                break
            c = Rs[i]
            cs = summaries[summaries.country == c]
            ax.plot(cs.day, cs.percent_mc * 100)# label="median wearing")
            
            if len(c) > 20:
                tsize = 10
            ax.set_title(c, fontsize=tsize)
            i += 1

            ax.axvline(x=0, color="black", linestyle="--")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            ax.set_ylim(0, 100)
            ax.set_xlim(-20, 20)
    
    fig.text(0.5, -0.02, 'Days since mandate', ha='center', fontsize=16)
    fig.text(-0.02, 0.5, "% mask wearing", va='center', fontsize=16, rotation='vertical')
    plt.tight_layout(pad=0.4)


def centred_mandate_plot(Rs, df):
    fig, ax = plt.subplots(figsize=(8,5), dpi=500)
    summaries = get_centred_summary(Rs, df)
    means = summaries.groupby("day").median() * 100
    lu = summaries.groupby("day").quantile(0.25) * 100
    hi = summaries.groupby("day").quantile(0.75) * 100

    ax.plot(means, label="median wearing")
    ax.fill_between(
        summaries.day.unique(),
        lu["percent_mc"],
        hi["percent_mc"],
        alpha=0.2,
        label="50% CI",
    )

    plt.xlabel("Days since mandate", fontsize=16)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.ylabel("% mask wearing", fontsize=16)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.legend(loc="lower right", fontsize=14, frameon=False)
    plt.ylim(0, 100)
    plt.show()


def get_centred_summary(Rs, df):
    pairs = pd.DataFrame()
    for c in Rs:
        cdf = df.loc[c]
        cdf.index = pd.to_datetime(cdf.index)
        switches_on = get_mandate_switches(cdf, c)

        if switches_on["H6_Facial Coverings"].any():
            x, y = centre_on_mandate(cdf, switches_on)
            cxy = pd.DataFrame()
            cxy["day"] = x
            cxy["percent_mc"] = y
            cxy["country"] = c
            pairs = pd.concat([pairs, cxy])

    return pairs


def mandate_barplot(df):
    plt.figure(figsize=(3, 3))
    ms = (
        df[["percent_mc", "H6_Facial Coverings"]]
        .groupby("H6_Facial Coverings")
        .median()
    )
    errs = (
        df[["percent_mc", "H6_Facial Coverings"]]
        .groupby("H6_Facial Coverings")
        .std()["percent_mc"]
    )
    plt.bar(
        ms.index,
        ms.percent_mc,
        yerr=errs,
        align="center",
        alpha=0.5,
        ecolor="black",
        width=0.3,
    )
    plt.legend(loc="lower right",frameon=False)
    plt.xlabel("Mandate indicator")
    plt.ylabel("% wearing")
    plt.xticks([0, 1])
    plt.ylim(0, 1)
    plt.show()


def mandate_distplot(df, col="H6_Facial Coverings"):
    sns.distplot(df[df[col] == 0]["percent_mc"], label="no mandate", kde=False)
    sns.distplot(
        df[df[col] == 1]["percent_mc"],
        label="mandate",
        kde=False,
        hist_kws={"alpha": 0.35},
    )
    # sns.distplot(df[df["H6_Facial Coverings_3plus"] == 1]["percent_mc"], label="full mandate", kde=False, hist_kws={"alpha": 0.35})
    plt.legend(frameon=False)
    plt.title(col)
    plt.show()


def original_stringency_plot(df):
    ms = df.groupby("H6_Facial Coverings").median()
    plt.plot(ms, label="median wearing")

    lu = df.groupby("H6_Facial Coverings").quantile(0.25)
    hi = df.groupby("H6_Facial Coverings").quantile(0.75)
    plt.fill_between(lu.index, lu["percent_mc"].tolist(), hi["percent_mc"], alpha=0.2)

    plt.legend(loc="lower right",frameon=False)
    plt.xlabel("Mandate stringency")
    plt.ylabel("% wearing")
    plt.show()

    
    
def post_mandate_rise(Rs, df):
    ranges = []
    d = get_centred_summary(Rs, df)
    for c in d.country.unique() :
        last = d[(d.country == c) & (d.day > 23)]
        first = d[(d.country == c) & (d.day < 7)]
        if len(last) :
            last = last.percent_mc.mean()
            first = first.percent_mc.mean()
            ranges.append( last - first )
    
    return ranges
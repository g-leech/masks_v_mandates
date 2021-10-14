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


# After processing, "H6_Facial Coverings" is a binary flag, 1 if mandate is enforced, "H6 level 2+"
def get_mandate_switches(cdf, col="H6_Facial Coverings"):
    return cdf[cdf[col].diff() == 1]


def messy_centred_mandate_plot(Rs, df):
    for c in Rs:
        cdf = df.loc[c]
        cdf.index = pd.to_datetime(cdf.index)
        switches_on = get_mandate_switches(cdf)

        if switches_on["H6_Facial Coverings"].any():
            centred_mandate_plot_country(cdf, switches_on)

    plt.xlabel("Days since mandate")
    plt.ylim(0, 1)
    plt.ylabel("% mask wearing")

    plt.plot([0] * 2, [0, 1], alpha=1, color="black", linestyle="--")
    plt.show()
    
    

def add_uk_nations(Rs, df):
    Rs2 = Rs.copy()
    df2 = df.copy()
    df2 = df2.drop("United Kingdom")
    latest_oxcgrt = pd.read_csv("../data/raw/OxCGRT_latest_oct21.csv")
    uk = latest_oxcgrt[latest_oxcgrt["CountryName"] == "United Kingdom"]
    uk = uk[~uk.RegionName.isna()]
    nations = list(uk.RegionName.unique())
    
    for c in nations:
        cdf = uk[uk.RegionName == c]
        cdf["H6_Facial Coverings"] = (cdf["H6_Facial Coverings"] >= 2) & (cdf["H6_Flag"] == 1) 
        cdf["H6_Facial Coverings_3plus"] = (cdf["H6_Facial Coverings"] >= 3) & (cdf["H6_Flag"] == 1) 
        cdf["Date"] = pd.to_datetime(cdf["Date"], format="%Y%m%d")
        cdf = cdf.set_index(["RegionName", "Date"])
        df2 = pd.concat([df2, cdf])

    Rs2.remove("United Kingdom")
    Rs2 += nations
    
    df2 = add_uk_wearing(df2)
    
    return Rs2, df2


def add_uk_wearing(df):
    uk_wearing = pd.read_csv("../data/raw/umd/umd_uk_regions.csv")
    uk_wearing = uk_wearing[~uk_wearing.region.isna()]
    uk_wearing["date"] = pd.to_datetime(uk_wearing["survey_date"], format="%Y%m%d")
    uk_wearing = uk_wearing[["percent_mc", "region", "date"]]
    regions = uk_wearing.region.unique()
    other_rs = ["Scotland", "Wales", "Northern Ireland"]
    england_rs = [r for r in regions if r not in other_rs]
    england = uk_wearing[uk_wearing.region.isin(england_rs)]

    ruk = uk_wearing[uk_wearing.region.isin(other_rs)]
    ruk["country"] = ruk["region"]
    ruk = ruk.drop("region", axis=1)
    ruk = ruk.set_index(["country", "date"])

    england = england.groupby("date").mean()
    england["country"] = "England"
    england = england.reset_index().set_index(["country", "date"])
    uk = pd.concat([england, ruk])
    df = df.join(uk, rsuffix="2")
    nations = other_rs + ["England"]
    
    for r in nations:
        df.loc[r]["percent_mc"] = df.loc[r]["percent_mc2"]
    return df


def get_mandates_by_date(mw_plot_Rs, mw_plot_df) :
    mandate_dates = []

    for c in mw_plot_Rs:
        cdf = mw_plot_df.loc[c]
        cdf.index = pd.to_datetime(cdf.index)
        switches_on = cdf[cdf["H6_Facial Coverings"].diff() > 0]
        if switches_on["H6_Facial Coverings"].any():
            mandate_dates.append(str(switches_on.index[0])[:10] + " " + c)

    mandate_dates = sorted(mandate_dates)
    mandates_by_date = [ l[1] for l in [s.split(" ", 1) for s in mandate_dates]]
    
    return mandates_by_date


def country_centred_mandate_plot(Rs, df, tsize=14, uk=True):
    nCols = 5
    if uk:
        mw_plot_Rs, mw_plot_df = add_uk_nations(Rs, df)
    else:
        mw_plot_Rs, mw_plot_df = Rs, df
    
    mandate_rs_by_date = get_mandates_by_date(mw_plot_Rs, mw_plot_df)
    summaries = get_centred_summary(mw_plot_Rs, mw_plot_df)
    Rs = mandate_rs_by_date #summaries.country.unique()
    nRs = len(Rs)
    nRows = int(np.ceil(nRs/nCols))
    
    fig, axes = plt.subplots(nrows=nRows, ncols=nCols, sharex=True, sharey=True, figsize=(10, nRows*1.4), dpi=350)
    i = 0
    for row in axes:
        for ax in row:
            if len(Rs) <= i :
                break
            c = Rs[i]
            cs = summaries[summaries.country == c]
            ax.plot(cs.day, cs.percent_mc * 100)# label="median wearing")
            
            if len(c) > 16:
                tsize = 10
            ax.set_title(c, fontsize=tsize)
            i += 1

            ax.axvline(x=0, color="black", linestyle="--")
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
            ax.set_ylim(0, 100)
            ax.set_xlim(-20, 20)
    
    fig.text(0.5, -0.02, 'Days since mandate', ha='center', fontsize=16)
    fig.text(-0.02, 0.5, "% mask wearing", va='center', fontsize=16, rotation='vertical')
    plt.tight_layout(pad=0.6)


def centred_mandate_plot(Rs, df, uk_breakup=True):
    fig, ax = plt.subplots(figsize=(8,5), dpi=250)
    
    if uk_breakup:
        Rs, df = add_uk_nations(Rs, df)
    
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
        label="50% interval",
    )

    plt.xlabel("Days since mandate", fontsize=16)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.ylabel("% mask wearing", fontsize=16)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.legend(loc="lower right", fontsize=14, frameon=False)
    plt.ylim(0, 100)


def get_centred_summary(Rs, df, col="H6_Facial Coverings"):
    pairs = pd.DataFrame()
    for c in Rs:
        cdf = df.loc[c]
        cdf.index = pd.to_datetime(cdf.index)
        switches_on = get_mandate_switches(cdf, col)

        if switches_on[col].any():
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



def centred_announcement_plot(Rs, df, announce_lead=5):
    fig, ax = plt.subplots(figsize=(8,5), dpi=250)
    
    summaries = get_centred_summary(Rs, df)
    summaries["day"] += announce_lead
    means = summaries.groupby("day").median() * 100
    lu = summaries.groupby("day").quantile(0.25) * 100
    hi = summaries.groupby("day").quantile(0.75) * 100

    ax.plot(means, label="median wearing")
    ax.fill_between(
        summaries.day.unique(),
        lu["percent_mc"],
        hi["percent_mc"],
        alpha=0.2,
        label="50% interval",
    )

    plt.xlabel("Days since announcement", fontsize=16)
    plt.axvline(x=0, color="black", linestyle="--")
    plt.ylabel("% mask wearing", fontsize=16)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.legend(loc="lower right", fontsize=14, frameon=False)
    plt.ylim(0, 100)
    plt.show()
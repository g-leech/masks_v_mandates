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
from matplotlib.ticker import FuncFormatter

sns.set_style("white")

locator = mdates.MonthLocator()  # every month
fmt = mdates.DateFormatter("%b")


def millions(x, pos):
    "The two args are the value and tick position"
    return "%1.1fM" % (x * 1e-6)


yformatter = FuncFormatter(millions)


def avg_plot(df, third_party_rts, Ds, alpha=0.05):
    Ds = pd.to_datetime(Ds)
    plt.subplot(3, 2, 1)
    c = df.reset_index().groupby("date").median().ConfirmedCases
    plt.plot(Ds, c)
    lu = (
        df.reset_index()[["ConfirmedCases", "date"]].groupby("date").quantile(alpha / 2)
    )
    hi = (
        df.reset_index()[["ConfirmedCases", "date"]]
        .groupby("date")
        .quantile(1 - alpha / 2)
    )
    plt.fill_between(Ds, lu.ConfirmedCases, hi.ConfirmedCases, alpha=0.1)
    plt.title("Average cumulative cases")
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)
    # ax.yaxis.set_major_formatter(yformatter)

    plt.subplot(3, 2, 2)
    third_party_rts["date"] = pd.to_datetime(third_party_rts["date"])
    tprt = third_party_rts[third_party_rts["date"].isin(Ds)]
    m = tprt.groupby("date").median()["mean"]
    plt.plot(Ds, m)

    lu = tprt[["mean", "date"]].groupby("date").quantile(alpha / 2)
    hi = tprt[["mean", "date"]].groupby("date").quantile(1 - alpha / 2)
    plt.fill_between(Ds, lu["mean"], hi["mean"], alpha=0.1)

    plt.title("Average Rt")
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    plt.subplot(3, 2, 3)
    w = df.reset_index().groupby("date").median().percent_mc
    plt.plot(Ds, w)
    lu = df.reset_index()[["percent_mc", "date"]].groupby("date").quantile(alpha / 2)
    hi = (
        df.reset_index()[["percent_mc", "date"]].groupby("date").quantile(1 - alpha / 2)
    )
    plt.fill_between(Ds, lu.percent_mc, hi.percent_mc, alpha=0.1)
    plt.title("Average wearing %")
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    plt.subplot(3, 2, 4)
    mobil = (
        df.reset_index()
        .groupby("date")
        .median()["avg_mobility_no_parks_no_residential"]
    )
    plt.plot(Ds, mobil)

    lu = (
        df.reset_index()[["avg_mobility_no_parks_no_residential", "date"]]
        .groupby("date")
        .quantile(alpha / 2)
    )
    hi = (
        df.reset_index()[["avg_mobility_no_parks_no_residential", "date"]]
        .groupby("date")
        .quantile(1 - alpha / 2)
    )
    plt.fill_between(
        Ds,
        lu["avg_mobility_no_parks_no_residential"],
        hi["avg_mobility_no_parks_no_residential"],
        alpha=0.1,
    )

    plt.title("Average mobility")
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    plt.subplot(3, 2, 5)
    mandate = df["H6_Facial Coverings"].reset_index().groupby("date").mean()
    plt.plot(Ds, mandate)

    plt.title("% mandates on")
    plt.ylim(0, 1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

    plt.tight_layout()
    plt.subplots_adjust(top=1.3)
    # plt.savefig("average_trends.pdf", dpi=1000)

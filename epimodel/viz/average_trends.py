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
from matplotlib.ticker import FormatStrFormatter

sns.set(style="ticks", font='DejaVu Serif')

locator = mdates.MonthLocator()  # every month
fmt = mdates.DateFormatter("%b")


def millions(x, pos):
    "The two args are the value and tick position"
    return "%1.1fM" % (x * 1e-6)


yformatter = FuncFormatter(millions)


def plot_avg_daily_new(df, Ds, ax, alpha=0.05):
    #plt.subplot(2, 5, 1)
    # new cases
    df2 = df.reset_index()
    df2["new"] = df2.groupby("country").ConfirmedCases.diff()
    df2["new"] = np.where(df2.new < 0, 0, df2.new)
    df2["new"] = df2["new"].rolling(7).mean()
    c = df2.groupby("date").median().new

    ax.plot(Ds, c)


    lu = (
        df2.reset_index()[["new", "date"]].groupby("date").quantile(0.25)
    )
    hi = (
        df2.reset_index()[["new", "date"]]
        .groupby("date")
        .quantile(0.75)
    )
    ax.fill_between(Ds, lu.new, hi.new, alpha=0.1)
    ax.set_title("A", loc="left", fontweight="bold")
    ax.set_ylabel("Average new daily cases", fontsize=10)
    
    #ax.axes.get_xaxis().set_visible(False)
    #ylabels = ['{:,.0f}'.format(x) + 'k' for x in ax.axes.get_yticks()/1000]
    #ax.set_yticklabels(ylabels)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def plot_mob(df, Ds, ax, alpha=0.5):
    #plt.subplot(2, 5, 7)
    mobil = (
        df.reset_index()
        .groupby("date")
        .median()["avg_mobility_no_parks_no_residential"]
    )
    ax.plot(Ds, mobil * 100)

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
    ax.fill_between(
        Ds,
        lu["avg_mobility_no_parks_no_residential"] * 100,
        hi["avg_mobility_no_parks_no_residential"] * 100,
        alpha=0.1,
    )
    ax.set_title("D", loc="left", fontweight="bold")
    ax.set_ylabel("Average mobility reduction", fontsize=10)
    ax.set_ylim(0, 100)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def plot_cum_cases(df):
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
    plt.title("A", loc="left", fontweight="bold")
    plt.ylabel("Average cumulative cases", fontsize=10)
    ax = plt.gca()
    #ax.axes.get_xaxis().set_visible(False)
    ylabels = ['{:,.0f}'.format(x) + 'k' for x in ax.axes.get_yticks()/1000]
    ax.set_yticklabels(ylabels)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)

def plot_mandates(df, Ds, ax):
    #plt.subplot(2, 5, 2)

    mandate = df["H6_Facial Coverings"].reset_index().groupby("date").mean()
    ax.plot(Ds, mandate * 100)
    ax.set_title("B", loc="left", fontweight="bold")
    ax.set_ylabel("% mandates on", fontsize=10)
    ax.set_ylim(0, 100)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    plt.setp(ax.get_xticklabels(), fontsize=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)
    #ax.axes.get_xaxis().set_visible(False)
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))




def plot_wearing(df, Ds, ax, alpha=0.5):
    #plt.subplot(2, 5, 6)
    w = df.reset_index().groupby("date").median().percent_mc
    ax.plot(Ds, w* 100)
    lu = df.reset_index()[["percent_mc", "date"]].groupby("date").quantile(alpha / 2)
    hi = (
        df.reset_index()[["percent_mc", "date"]].groupby("date").quantile(1 - alpha / 2)
    )
    ax.fill_between(Ds, lu.percent_mc * 100, hi.percent_mc* 100, alpha=0.1)
    ax.set_title("C", loc="left", fontweight="bold")
    ax.set_ylabel("Average wearing %", fontsize=10)
    ax.set_ylim(0, 100)
    
    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.setp(ax.get_yticklabels(), fontsize=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(fmt)


def plot_rts(third_party_rts, Ds):
    #plt.subplot(2, 5, 2)
    third_party_rts["date"] = pd.to_datetime(third_party_rts["date"])
    tprt = third_party_rts[third_party_rts["date"].isin(Ds)]
    m = tprt.groupby("date").median()["mean"]
    plt.plot(Ds, m)

    lu = tprt[["mean", "date"]].groupby("date").quantile(alpha / 2)
    hi = tprt[["mean", "date"]].groupby("date").quantile(1 - alpha / 2)
    plt.fill_between(Ds, lu["mean"], hi["mean"], alpha=0.1)

    plt.title("Average Rt", fontsize=10, loc="left")
    ax = plt.gca()
    #ax.axes.get_xaxis().set_visible(False)


def plot_validation():
    return

def avg_plot(df, Ds, alpha=0.05):
    Ds = pd.to_datetime(Ds)
    plot_avg_daily_new(df, Ds)
    plot_wearing(df, Ds)
    plot_mob(df, Ds)
    plot_mandates(df, Ds)
    #plot_validation()
    

    plt.tight_layout(pad=1.5)
    plt.subplots_adjust(top=0.9)#, right=1)
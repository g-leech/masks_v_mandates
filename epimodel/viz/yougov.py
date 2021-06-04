import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import os

pd.options.mode.chained_assignment = None

sns.set_style("white")

import sys; 
sys.path.append("../scripts")
sys.path.append("../epimodel/")
from city_mandate_data import *
import preprocessing.mandate_feature_gen as preproc

DATA_IN = "../data/raw/"



national_mandates_enforced = {
    'France' : ["2020-05-11", "2020-07-20"],
    'Australia' : None,
     'Brazil': "2020-07-10",
    # Very gradual. Most provinces by Oct
     'Canada': None,
     'China' : None,
    # Public transport from 22 Aug, shops from 29th Oct
     'Denmark' : ["2020-08-22", "2020-10-29"],
     'Finland' : None,
    # Final state...
     'Germany' : "2020-04-27",
     'Hong Kong': "2020-07-23",
     'India' : None,
     'indonesia' : "2020-04-05",
     'Italy' : ["2020-05-04", "2020-10-08"],
     'Japan' : None,
     'Malaysia' : "2020-08-01",
     'Mexico': None, #"2020-05-20",
     'Norway': None,
     'Philippines' : "2020-04-02",
     'Saudi Arabia': "2020-05-30",
     'Singapore' : "2020-04-14",
     'Spain' : "2020-05-04",
     'Sweden' : None,
     'Taiwan' : "2020-03-31",
     'Thailand' : "2020-03-25",
     'united-arab-emirates': "2020-04-28",
     # Public transport June
     'united-kingdom' : ["2020-06-04", "2020-07-14"],
     'united-states': None,
     'Vietnam' : "2020-03-16"
}


national_mandates_announced = {
    'france' : ["2020-05-05", "2020-07-16"],
    'australia' : None,
     'brazil': "2020-07-02",
     'canada': None,
     'china' : None,
     'denmark' : ["2020-08-15", "2020-10-23"],
     'finland' : None,
    # Not federal. Take Thuringia as the "announcement"
     'germany' : "2020-04-06",
     'hong-kong': "2020-07-22",
     'india' : None,
     'indonesia' : "2020-04-05",
     'italy' : ["2020-04-28", "2020-10-08"],
     'japan' : None,
     'malaysia' : "2020-07-30",
    # Not federal
     'mexico': None, #"2020-05-20",
     'norway': None,
     'philippines' : "2020-04-02",
     'saudi-arabia': "2020-05-30",
     'singapore' : "2020-04-14",
     'spain' : "2020-05-02",
     'sweden' : None,
     'taiwan' : "2020-04-01",
     'thailand' : "2020-03-25",
     'united-arab-emirates': "2020-04-28",
     # Public transport June
     'united-kingdom' : ["2020-06-15", "2020-07-24"],
     'united-states': None,
     'vietnam' : "2020-03-16"
}

def do_mandate_plot(df, country, announced, enforced, title, ax) : 
    sns.lineplot(x="date", y="pct_wearing_public", data=df, marker='o')
    plt.title(f'{country}'.capitalize())
    
    locator = mdates.MonthLocator()  # every month
    # Specify the format - %b gives us Jan, Feb...
    fmt = mdates.DateFormatter('%b')
    X = ax.xaxis
    X.set_major_locator(locator)
    # Specify formatter
    X.set_major_formatter(fmt)
    
    #announcements = announced[country.lower()]
    mandates = enforced[country.lower()]

    if type(mandates) == list :
        for m in mandates :
            plt.axvline(x=pd.to_datetime([m]), color="black", linestyle="--", label="mandate")
    else :
        x = pd.to_datetime([mandates])
        for i in x :
            plt.axvline(x=i, color="black", linestyle="--", label="mandate")
        
#     if type(announcements) == list :
#         for a in announcements :
#             x = pd.to_datetime([a]*len(vert))
#             plt.plot(x, vert, label="announced")
#     else :
#         x = pd.to_datetime([announcements]*len(vert))
#         plt.plot(x, vert, label="announced")
    plt.ylim(0,100)
    plt.ylabel("")#("YouGov % wearing estimate")
    plt.xlabel("2020")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.show()

    
def load_yg() :
    path = DATA_IN + "yougov-chart-mask-pct.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"UAE": "united-arab-emirates", "UK": "united-kingdom", "USA": "united-states"})
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.date
    
    return df


def plot_mandates(ax, country):
    mandates = national_mandates_enforced[country]

    if type(mandates) == list :
        for m in mandates :
            ax.axvline(x=pd.to_datetime([m]), color="black", linestyle="--")
    else :
        x = pd.to_datetime([mandates])
        for i in x :
            ax.axvline(x=i, color="black", linestyle="--")

            
def plot_earliest_mandates_against_wearing():
    df = load_yg()
    yg_Rs = [k for k,v in national_mandates_enforced.items() if v is not None]
    yg_Rs = [i for i in yg_Rs if i in df.columns]
    nRs = len(yg_Rs)
    fig, axes = plt.subplots(nrows=round(nRs/3), ncols=3, sharex=True, sharey=True, figsize=(7, 10), dpi=500)
    
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)
    mpl.rcParams['xtick.labelsize'] = 10 
    mpl.rcParams['ytick.labelsize'] = 10 
    
    i = 0
    for row in axes:
        for ax in row:
            #if len(yg_Rs) == i: 
                #break
            country = yg_Rs[i]
            df2 = df[["DateTime", country]] 
            df2.columns = ["date", "pct_wearing_public"]
            sns.lineplot(x="date", y="pct_wearing_public", data=df2, marker='o', ax=ax)
            ax.set_title(f'{country}'.title(), fontsize=12)
            ax.set_ylabel("% wearing", fontsize=12)
            
            plot_mandates(ax, country)

            locator = mdates.MonthLocator((3,6,9,12)) 
            # Specify the format - %b gives us Jan, Feb...
            fmt = mdates.DateFormatter('%b')
            X = ax.xaxis
            X.set_major_locator(locator)
            # Specify formatter
            X.set_major_formatter(fmt)
            ax.set_xlabel("2020", fontsize=14)
            
            plt.ylim(0,100)
            
            i += 1
    plt.tight_layout()
    plt.show()
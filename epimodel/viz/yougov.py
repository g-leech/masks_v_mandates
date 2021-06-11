import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import os

pd.options.mode.chained_assignment = None

sns.set(style="ticks", font='DejaVu Serif')

import sys; 
sys.path.append("../scripts")
sys.path.append("../epimodel/")

DATA_IN = "../data/raw/"



national_mandates_enforced = {
    'france' : ["2020-05-11", "2020-07-20"],
    'australia' : None,
     'brazil': "2020-07-10",
    # Very gradual. Most provinces by Oct
     'canada': None,
     'china' : None,
    # Public transport from 22 Aug, shops from 29th Oct
     'denmark' : ["2020-08-22", "2020-10-29"],
     'finland' : None,
    # Final state...
     'germany' : "2020-04-27",
     'hong-kong': "2020-07-23",
     'india' : None,
     'indonesia' : "2020-04-05",
     'italy' : ["2020-05-04", "2020-10-08"],
     'japan' : None,
     'malaysia' : "2020-08-01",
     'mexico': None, #"2020-05-20",
     'norway': None,
     'philippines' : "2020-04-02",
     'saudi-arabia': "2020-05-30",
     'singapore' : "2020-04-14",
     'spain' : "2020-05-04",
     'sweden' : None,
     'taiwan' : "2020-03-31",
     'thailand' : "2020-03-25",
     'united-arab-emirates': "2020-04-28",
     # Public transport June
     'united-kingdom' : ["2020-06-04", "2020-07-14"],
     'united-states': None,
     'vietnam' : "2020-03-16"
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

            
def up(i):
    return i.title().replace("-", " ")

def plot_earliest_mandates_against_wearing():
    df = load_yg()
    df.columns = list(df.columns[:-4]) + ["United Arab Emirates", "United Kingdom", "United States", "Vietnam"]
    
    yg_Rs = [k for k,v in national_mandates_enforced.items() if v is not None]
    yg_Rs = [i for i in yg_Rs if up(i) in df.columns]
    nRs = len(yg_Rs)
    fig, axes = plt.subplots(nrows=round(nRs/3), ncols=3, sharex=True, sharey=True, figsize=(7, 10), dpi=500)
    mpl.rcParams['xtick.labelsize'] = 10 
    mpl.rcParams['ytick.labelsize'] = 10 
    
    i = 0
    for row in axes:
        for ax in row:
            country = yg_Rs[i]
            df2 = df[["DateTime", up(country)]] 
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
    # plt.show()
    
    
    
    
def get_before_after_means(df, mandates) :
    df = df.set_index("date")
    mandates = pd.to_datetime(mandates)
    
    if type(mandates) == pd.DatetimeIndex:
        before = df[:mandates[0]]
        after1 = df[mandates[0]:mandates[1]]
        after2 = df[mandates[1]:]
        
        # Take the last observation before mandate
        obs_before = before[~before.pct_wearing_public.isnull()] \
                        .tail(1)
        pre_mean = obs_before.pct_wearing_public.mean()
        
        # take max of next 3 weeks
        post_mean1 = after1[:21].pct_wearing_public.max()
        post_mean2 = after2[:21].pct_wearing_public.max()
        effect1 = post_mean1 - pre_mean
        effect2 = post_mean2 - post_mean1
        effect = (effect1 + effect2)/2
    else :
        before = df[:mandates]
        after = df[mandates:]
        # Take the last observation before mandate
        obs_before = before[~before.pct_wearing_public.isnull()] \
                        .tail(1)
        pre_mean = obs_before.pct_wearing_public.mean()
        post_mean = after.pct_wearing_public.mean()
        # TODO: Normalise to prior uptake?
        effect = (post_mean - pre_mean)
    
    #if pre_mean > 70 :
    #    return
    
    if len(before) == 0 :
        return
    
    #print(f"Effect: {round(effect, 1)}%")
    
    return effect

def get_pre_mandate_uptake(df, country) :
    fmt_country = country.lower().replace(" ", "-")
    mandate = national_mandates_announced[fmt_country]
    
    if not mandate :
        return None
    
    if type(mandate) == list :
        mandate = mandate[0]
    
    mandate = pd.to_datetime(mandate).date()
    df2 = df[:mandate]
    pre_obs = df2[~df2[country].isnull()].reset_index()[country]
    return list(pre_obs.tail(1))



def get_earliest_uptake(df, country) :
    df2 = df[[country]]
    return df2[~df2[country].isnull()].head(1)





def get_max_mandate_effect_on_wearing(country, df):
    fmt_country = country.lower().replace(" ", "-") 
    
    if not national_mandates_enforced[fmt_country] :
        return
    
    df2 = df[["DateTime", country]] 
    df2.columns = ["date", "pct_wearing_public"]

    title = ""

    mandates = national_mandates_enforced[fmt_country]
    
    return get_before_after_means(df2, mandates)


def get_before_after_mandate_change():
    DATA_IN = "../data/raw/"
    path = DATA_IN + "yougov-chart-mask-pct.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"UAE": "united-arab-emirates", "UK": "united-kingdom", "USA": "united-states"})
    countries = [c for c in df.columns if c != "DateTime"]
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.date
    df = df.set_index("DateTime")
    
    first_uptakes = [ get_earliest_uptake(df, country) for country in countries ]
    avg = np.mean(first_uptakes)
    print(f"Avg uptake at year start: {avg:.1f}")
    
    uptakes = [ get_pre_mandate_uptake(df, country) \
               for country in countries]
    uptakes = [u for u in uptakes if u is not None]
    uptakes = [u[0] for u in uptakes if u]
    avgBefore = np.mean(uptakes)
    print(f"Avg uptake just before announcement: {avgBefore}")
    
    df = df.reset_index()
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.date
    results = [get_max_mandate_effect_on_wearing(country, df) for country in countries]
    
    avgRise = np.mean([r for r in results if r != None and r != "nan"][:-1])#np.nanmean(results)
    print(f"Avg uptake after mandate: {(avgBefore + avgRise):.1f}")
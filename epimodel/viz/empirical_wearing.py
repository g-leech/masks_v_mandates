import calendar

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns

import epimodel.pymc3_models.base_model as bm

sns.set_style("white")


def exp_reduction(a, x):
    reductions = 1 - np.exp((-1.0) * a * x)
    return reductions.mean()


def relu(x):
    return np.maximum(x, 0)


def ll_red(alphas, x):
    w = alphas * x
    return -np.log(relu(1 - w))


def q2_reduction(alphas, x):
    w = alphas[0] * x + alphas[1] * x ** 2
    reductions = -np.log(relu(1 - w))
    return reductions.mean()


def get_alpha_and_reducer(tr, w_par):
    if w_par == "log_quadratic_2":
        a0 = tr.Wearing_Alpha[0]
        a1 = tr.Wearing_Alpha[1]
        a = [a0, a1]
        r = q2_reduction
    elif w_par == "exp":
        a = tr.Wearing_Alpha
        r = exp_reduction

    return a, r


def get_max_reduction(tr, df, w_par):
    obs_ = []
    a, r = get_alpha_and_reducer(tr, w_par)

    for c in df.reset_index().country.unique():
        cdf = df.loc[c]
        max_ = cdf.percent_mc.max()
        max_reduction_r = r(max_, a)
        actual = max_reduction_r
        obs_.append(actual * 100)

    return obs_


def get_median_reduction(tr, df, w_par):
    obs_ = []
    a, r = get_alpha_and_reducer(tr, w_par)

    for c in df.reset_index().country.unique():
        cdf = df.loc[c]
        median_ = cdf.percent_mc.median()
        med_reduction_r = r(median_, a)
        actual = med_reduction_r
        obs_.append(actual * 100)

    return obs_


def get_min_max_reduction(tr, df, w_par):
    obs_ = []

    for c in df.reset_index().country.unique():
        cdf = df.loc[c]
        max_ = cdf.percent_mc.max()
        min_ = cdf.percent_mc.min()

        if w_par == "log_quadratic_2":
            a0 = tr.Wearing_Alpha[0].mean()
            a1 = tr.Wearing_Alpha[1].mean()
            max_reduction_r = log_quad2_reduction(max_, [a0, a1])
            min_reduction_r = log_quad2_reduction(min_, [a0, a1])
        elif w_par == "exp":
            max_reduction_r = exp_reduction(max_, tr.Wearing_Alpha.mean())
            min_reduction_r = exp_reduction(min_, tr.Wearing_Alpha.mean())

        actual = (
            max_reduction_r - min_reduction_r
        )  # (1 - max_reduction_r) - (1 - min_reduction_r)
        obs_.append(actual * 100)

    return obs_


"""
    df = masks_object.df
    plot_actual_wearing_effect(df, exp_trace, "exp")
"""


def plot_actual_wearing_effect(df, tr, w_par):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=500)
    obs_ = get_min_max_reduction(tr, df, w_par)
    print(np.percentile(obs_, [2.5, 50, 97.5]))
    sns.distplot(obs_, kde=False)

    plt.xlabel("% R reduction from wearing", fontsize=16)
    ax.yaxis.set_ticks([])

    med = np.median(obs_)
    print(med)
    max_ = plt.gca().get_ylim()[1]
    straight = np.arange(0, max_, 0.0001)
    plt.plot(
        [med] * len(straight), straight, color="black", linestyle="--", label="median"
    )
    plt.legend(fontsize=16)
    plt.show()


"""
    df = masks_object.df
    plot_max_wearing_effect(df, exp_trace, "exp")
"""


def plot_max_wearing_effect(df, tr, w_par, ax):
    # fig, ax = plt.subplots(figsize=(8,5), dpi=500)
    obs_ = get_max_reduction(tr, df, w_par)
    print(np.percentile(obs_, [2.5, 50, 97.5]))
    sns.distplot(obs_, kde=True, hist=False)

    plt.xlabel("% R reduction from wearing", fontsize=16)
    plt.gca().yaxis.set_ticks([])
    plt.xlim(0, 30)

    med = np.median(obs_)
    print(med)
    plt.axvline(x=med, color="black", linestyle="--", label="median")

    plt.legend(fontsize=16, frameon=False, loc="upper left")
    plt.show()


def plot_median_wearing_effect(df, tr, w_par, ax):
    # fig, ax = plt.subplots(figsize=(8,5), dpi=500)
    obs_ = get_median_reduction(tr, df, w_par)
    print(np.percentile(obs_, [2.5, 50, 97.5]))
    sns.distplot(obs_, kde=True, hist=False)

    plt.xlabel("% R reduction from wearing", fontsize=16)
    plt.gca().yaxis.set_ticks([])
    plt.xlim(0, 30)

    med = np.median(obs_)
    print(med)
    plt.axvline(x=med, color="black", linestyle="--", label="median")

    plt.legend(fontsize=16, frameon=False, loc="upper left")
    plt.show()

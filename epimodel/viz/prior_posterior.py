import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns

from epimodel.pymc3_distributions.asymmetric_laplace import AsymmetricLaplace

sns.set_style("white")
az.style.use("arviz-white")

def prior_posterior_plot(prior, posterior, t):
    fig, ax2 = plt.subplots(1, 1, figsize=(5,5), dpi=500)

    az.plot_dist(prior, label="prior", color="red", ax=ax2)
    az.plot_dist(posterior, label="posterior", ax=ax2)
    plt.title(t)
    ax2.legend()
    ax2.yaxis.set_ticks([])

    #az.plot_dist(prior, label="prior", color="red", ax=ax1)
    plt.show()


def plot_all_pps(trace):
    n = 5000
    if "GrowthNoiseScale" in trace.varnames:
        # GrowthNoiseScale
        with pm.Model() as m:
            pm.HalfNormal("GrowthNoiseScale", 0.2)
            m.trace = pm.sample(n)

        prior_posterior_plot(
            m.trace.GrowthNoiseScale, trace.GrowthNoiseScale, t="GrowthNoiseScale"
        )

    # Wearing
    if "Wearing_Alpha" in trace.varnames:
        wearing_sigma = 0.2
        with pm.Model() as ma:
            pm.Normal("Wearing_Alpha", mu=0, sigma=wearing_sigma, shape=(1,))
            ma.trace = pm.sample(n)

        prior_posterior_plot(ma.trace.Wearing_Alpha, trace.Wearing_Alpha, t="Wearing_Alpha")

    # R noise scale
    if "HyperRMean" in trace.varnames:
        with pm.Model() as mm:
            R_prior_mean_mean=1.07
            R_prior_mean_scale=0.4
            HyperRMean = pm.TruncatedNormal(
                "HyperRMean", mu=R_prior_mean_mean, sigma=R_prior_mean_scale, lower=0.1
            )
            mm.trace = pm.sample(n)

        prior_posterior_plot(mm.trace.HyperRMean, trace.HyperRMean, t="R0 mean")
        
    if "HyperRVar" in trace.varnames:
        with pm.Model() as mr:
            pm.HalfNormal("HyperRVar", sigma=0.3)
            mr.trace = pm.sample(n)

        prior_posterior_plot(mr.trace.HyperRVar, trace.HyperRVar, t="R scale")

    if "r_walk_noise_scale" in trace.varnames:
        with pm.Model() as m:
            pm.HalfNormal("r_walk_noise_scale", 0.15)
            m.trace = pm.sample(n)

        prior_posterior_plot(
            m.trace.r_walk_noise_scale, trace.r_walk_noise_scale, t="r_walk_noise_scale"
        )

    # case delay
    if "CasesDelayMean" in trace.varnames:
        with pm.Model() as md:
            pm.Normal("CasesDelayMean", 10, 1)
            md.trace = pm.sample(n)

        prior_posterior_plot(
            md.trace.CasesDelayMean, trace.CasesDelayMean, t="Case delay mean"
        )

    if "CasesDelayDisp" in trace.varnames:
        with pm.Model() as mdd:
            pm.Normal("CasesDelayDisp", 5, 1)
            mdd.trace = pm.sample(n)

        prior_posterior_plot(
            mdd.trace.CasesDelayDisp, trace.CasesDelayDisp, t="Case delay spread"
        )

    # GI

    with pm.Model() as mg:
        pm.Normal("GI_mean", 5, 1)
        mg.trace = pm.sample(n)

    prior_posterior_plot(mg.trace.GI_mean, trace.GI_mean, t="GI mean")

    with pm.Model() as mgd:
        pm.Normal("GI_sd", 2, 1)
        mgd.trace = pm.sample(n)

    prior_posterior_plot(mgd.trace.GI_sd, trace.GI_sd, t="GI sd")

    # Psi

    with pm.Model() as mp:
        pm.HalfNormal("Psi", 5)
        mp.trace = pm.sample(n)

    prior_posterior_plot(mp.trace.Psi, trace.Psi, t="Output noise spread")

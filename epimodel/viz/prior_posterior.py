import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
from matplotlib.lines import Line2D

from epimodel.pymc3_distributions.asymmetric_laplace import AsymmetricLaplace

sns.set(style="ticks", font='DejaVu Serif')

PNAS_WIDTH_INCHES = 3.4252

def prior_posterior_plot(prior, posterior, t, ax):
    az.plot_dist(prior, color="red", ax=ax)
    az.plot_dist(posterior, ax=ax)
    ax.set_title(t, fontsize=11)
    ax.yaxis.set_ticks([])


def plot_all_pps(trace, mandate_tr):
    fig, axes = plt.subplots(2, 3, figsize=(7,5), dpi=400)
    
    n = 5000
    
    # R noise scale
    if "HyperRMean" in trace.varnames:
        with pm.Model() as mm:
            R_prior_mean_mean=1.07
            R_prior_mean_scale=0.4
            HyperRMean = pm.TruncatedNormal(
                "HyperRMean", mu=R_prior_mean_mean, sigma=R_prior_mean_scale, lower=0.1
            )
            mm.trace = pm.sample(n)

        prior_posterior_plot(mm.trace.HyperRMean, trace.HyperRMean, t="R0 hyperprior mean", ax=axes[0][0])
        
    if "HyperRVar" in trace.varnames:
        with pm.Model() as mr:
            pm.HalfNormal("HyperRVar", sigma=0.3)
            mr.trace = pm.sample(n)

        prior_posterior_plot(mr.trace.HyperRVar, trace.HyperRVar, t="R0 hyperprior scale", ax=axes[0][1])

    if "r_walk_noise_scale" in trace.varnames:
        with pm.Model() as m:
            pm.HalfNormal("r_walk_noise_scale", 0.15)
            m.trace = pm.sample(n)

        prior_posterior_plot(
            m.trace.r_walk_noise_scale, trace.r_walk_noise_scale, t="r_walk_noise_scale", ax=axes[0][2]
        )

    # GI
    with pm.Model() as mg:
        pm.Normal("GI_mean", 5, 1)
        mg.trace = pm.sample(n)

    prior_posterior_plot(mg.trace.GI_mean, trace.GI_mean, t="GI mean", ax=axes[1][0])

    with pm.Model() as mgd:
        pm.Normal("GI_sd", 2, 1)
        mgd.trace = pm.sample(n)

    prior_posterior_plot(mgd.trace.GI_sd, trace.GI_sd, t="GI sd", ax=axes[1][1])

    # Psi

    with pm.Model() as mp:
        pm.HalfNormal("Psi", 5)
        mp.trace = pm.sample(n)

    prior_posterior_plot(mp.trace.Psi, trace.Psi, t="Output noise spread", ax=axes[1][2])

    # axes[2][0].axis('off')
    # axes[2][1].axis('off')
    # axes[2][2].axis('off')
    prior = Line2D([0], [0], label='prior',color='red')
    post = Line2D([0], [1], label='posterior',color='blue')
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.extend([prior,post])
    axes[0][0].legend(handles=handles, frameon=False, loc="upper right", fontsize=6)
    
    
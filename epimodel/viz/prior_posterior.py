import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm
import seaborn as sns
from matplotlib.lines import Line2D
import theano.tensor as T
from epimodel.pymc3_distributions.asymmetric_laplace import AsymmetricLaplace


sns.set(style="ticks", font='DejaVu Serif')

PNAS_WIDTH_INCHES = 3.4252

def prior_posterior_plot(prior, posterior, t, ax):
    az.plot_dist(prior, color="red", ax=ax)
    az.plot_dist(posterior, ax=ax)
    ax.set_title(t, fontsize=9)
    ax.yaxis.set_ticks([])


def plot_all_pps(trace):
    fig, axes = plt.subplots(3, 3, figsize=(7,8), dpi=400)
    
    n = 6000
    
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
    
    
    with pm.Model() as w:
        wearing_red_mean=1.07
        wearing_red_scale=0.4
        Wearing_Alpha = pm.Normal(
                        "Wearing_Alpha", mu=0, sigma=0.4, shape=(1,)
                    )
        WearingReduction = pm.Deterministic(
            "WearingReduction", T.exp((-1.0) * Wearing_Alpha)
        )

        w.trace = pm.sample(10000)
    
    wred = (1 - w.trace.WearingReduction) * 100
    wred_post = (1 - trace.WearingReduction) * 100
    prior_posterior_plot(wred, wred_post, \
                         t="Mask effect (100% wearing)", ax=axes[2][0])
    axes[2][0].set_xlim(-100, 100)
    
    priors, posts, title = plot_npi_prior_effect(trace, axes[2][1], combined=True)
    prior_posterior_plot(priors, posts, t=title, ax=axes[2][1])
    

    # axes[2][0].axis('off')
    # axes[2][1].axis('off')
    # axes[2][2].axis('off')
    prior = Line2D([0], [0], label='prior',color='red')
    post = Line2D([0], [1], label='posterior',color='blue')
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles.extend([prior,post])
    axes[0][0].legend(handles=handles, frameon=False, loc="upper right", fontsize=6)
    
    axes[2][2].axis('off')
    plt.tight_layout()
    

def plot_npi_prior_effect(trace, ax, combined=False):
    with pm.Model() as npi:
        cm_prior_scale = 10
        cm_prior="skewed"
        nCMs = 9

        CM_Alpha = AsymmetricLaplace(
                            "CM_Alpha",
                            scale=cm_prior_scale,
                            symmetry=0.5,
                            shape=(nCMs,),
                        )
        CMReduction = pm.Deterministic(
                    "CMReduction", (T.exp((-1.0) * CM_Alpha))#.mean(axis=1))
                )

        npi.trace = pm.sample(15000)

    total_npi_effect = npi.trace.CMReduction
    total_npi_effect_post = trace.CMReduction#.mean(axis=1)
    pct_reduction = (1 - total_npi_effect) * 100 
    pct_reduction_post = (1 - total_npi_effect_post) * 100 
    title = "NPI effect (each)"

    if combined :
        pct_reduction_ = pct_reduction.sum(axis=1)
        pct_reduction_post = pct_reduction_post.sum(axis=1)
        title = "NPI effect (total)"
    
    return pct_reduction, pct_reduction_post, title


import sys
sys.path.append("../")
sys.path.append("../../")

import os
import yaml
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
import matplotlib.gridspec as gridspec
plt.rcParams["font.family"] = "Optima"
plt.rcParams["font.weight"] = "light"

import matplotlib.pyplot as plt
import seaborn as sns

import arviz as az
import textwrap
import json
import argparse

cols = sns.color_palette('colorblind')

argparser = argparse.ArgumentParser()
argparser.add_argument('--results_path', dest='results_path', type=str, required=True)
argparser.add_argument('--figure_path', dest='figure_path', type=str, required=True)

args, _ = argparser.parse_known_args()

base_dir = args.results_path
fig_path = args.figure_path

#%%

NPIs = ['School closing',
        'School closing: full',
        'Workplace closing',
        'Workplace closing: full',
        'Restrictions on gatherings: 3+',
        'Stay at home requirements',
        'Restrictions on internal movement',
        'Restrictions on gatherings: 2+',
        'Restrictions on gatherings: full']

def create_grouping(info_dict):
    grouped_reductions = []
    if 'CMReduction' in info_dict.keys():
        grouped_reductions.append('CMReduction')
    if 'MandateReduction' in info_dict.keys():
        grouped_reductions.append('MandateReduction')
    if 'MobilityReduction' in info_dict.keys():
        grouped_reductions.append('MobilityReduction')
    if 'WearingReduction' in info_dict.keys():
        grouped_reductions.append('WearingReduction')

    return grouped_reductions

def combine_reductions(info_dict):
    grouped_reductions = create_grouping(info_dict)

    reduction = np.array(info_dict[grouped_reductions[0]])
    if len(reduction.shape) == 1:
        reduction = reduction.reshape((list(reduction.shape)[0], 1))
    # print(reduction)
    # initialise to first reduction
    all_reductions = reduction
    if len(grouped_reductions) > 1:
        for reduction in grouped_reductions[1:]:
            transformed_reduction = np.array(info_dict[reduction])
            if len(transformed_reduction.shape) == 1:
                transformed_reduction = transformed_reduction.reshape((list(transformed_reduction.shape)[0], 1))
            # print(transformed_reduction)
            all_reductions = np.concatenate((all_reductions, transformed_reduction), axis=1)

    info_dict['All_CM_Reductions'] = all_reductions

    return info_dict

def intervention_prior_labeler(d):
    if float(d['exp_config']['intervention_prior']['scale']) == 20.0:
        return  "AsymmetricLaplace(0, 0.5, 20)"
    elif d['exp_config']['intervention_prior']['type'] == "normal":
        return f"Normal(0, {d['exp_config']['intervention_prior']['scale']}$^2$)"
    elif float(d['exp_config']['intervention_prior']['scale']) == 0.15:
        return "HalfNormal(0, 0.15$^2$)"

def get_all_experiments(path):
    experiments = []
    for subdir, dirs, files in os.walk(f'{path}'):
        for filename in files:
            if filename.endswith('.json'):
                filepath = subdir + os.sep + filename
                print(filepath)
                with open(filepath) as f:
                    try:
                        data = json.load(f)
                        new_exp = combine_reductions(data)
                        experiments.append(new_exp)
                        print(f"Divergences: {np.sum(new_exp['divergences'])/len(new_exp['divergences'])}")
                        summary = new_exp['summary']
                        r_hats = [l[-1] for l in summary]
                        bad_r_hat_inds = [i for i in range(len(r_hats)) if r_hats[i] >= 1.05]
                        print(f'There are {len(bad_r_hat_inds)} parameters with r_hat > 1.05')
                    except Exception as e:
                        print(e)
                        print(f'failed to load {f}')
    print(len(experiments))
    return experiments

def filter_by_exp_tag(experiments, exp_tag):
    filtered = []
    for experiment in experiments:
        if experiment['exp_tag'] == exp_tag:
                    filtered.append(experiment)
    return filtered


all_exp_info = {
    "default": {
        "title": "Default",
        "labeler": "",
        "default_label": "Default"
    },
    # "basic_R_prior_mean": {
    #     "title": "$\\tilde{R}_{0, l}$ Prior: Mean",
    #     "labeler": lambda d: d['exp_config']["basic_R_prior"]["mean"],
    #     "default_label": "Default (1.1)"
    # },
    "basic_R_prior_scale": {
        "title": "$\\tilde{R}_{0, l}$ Prior: variability scale",
        "labeler": lambda d: d['exp_config']["basic_R_prior"]["noise_scale"],
        "default_label": "Default (0.4)"
    },
    "basic_R_hyperprior_mean_mean": {
        "title": "$\\tilde{R}_{0, l}$ Prior: mean mean",
        "labeler": lambda d: d['exp_config']["basic_R_prior"]["mean_mean"],
        "default_label": "Default (1.07)"
    },
    "basic_R_hyperprior_mean_scale": {
        "title": "$\\tilde{R}_{0, l}$ Prior: mean scale",
        "labeler": lambda d: d['exp_config']["basic_R_prior"]["mean_scale"],
        "default_label": "Default (0.2)"
    },
    # "basic_R_hyperprior_var_scale": {
    #     "title": "$\\tilde{R}_{0, l}$ Prior: Variability Scale",
    #     "labeler": lambda d: d['exp_config']["basic_R_prior"]["mean_scale"],
    #     "default_label": "Default (0.3)"
    # },
    "cases_delay_mean": {
        "title": "Mean delay from infection to case reporting",
        "labeler": lambda d: f"{float(d['exp_config']['cases_delay_mean'])} days",
        "default_label": "Default (10.92 days)"
    },
    "gen_int_mean": {
        "title": "Generation interval mean",
        "labeler": lambda d: f"{float(d['exp_config']['gen_int_mean'])} days",
        "default_label": "Default (5.41 days)"
    },
    "r_walk_noise_scale_prior": {
        "title": "Random walk noise scale prior",
        "labeler": lambda d: f"HalfNormal({float(d['exp_config']['r_walk_noise_scale_prior']):.2f})",
        "default_label": "Default (HalfNormal(0.15))"
    },
    "npi_leaveout": {
        "title": "NPI leave-out",
        # "labeler": lambda d: f"{textwrap.fill(corrected_names[d['exp_config']['npis'][0]], 15)}",
        "labeler": lambda d: f"{NPIs[d['exp_config']['npis'][0]]}",
        "default_label": "Default"
    },
    "r_walk_period": {
        "title": "Random walk period",
        "labeler": lambda d: f"{d['exp_config']['r_walk_period']} days",
        "default_label": "Default (7 days)"
    },
    "intervention_prior": {
        "title": "Intervention effect prior",
        "labeler": intervention_prior_labeler,
        "default_label": "AsymmetricLaplace(0, 0.5, 30)"
    },
    "boostrap": {
        "title": "Bootstrap",
        "labeler": lambda d: "Bootstrapped",
        "default_label": "Default"
    },
    # "mandates": {
    #     "title": "Mask Mandates",
    #     "labeler": lambda d: "Mandates",
    #     "default_label": "Wearing"
    # },
    "mask_sigma": {
        "title": "Mandate effect prior scale",
        "labeler": lambda d: f"{d['exp_config']['mask_sigma']}",
        "default_label": "Default (0.08)"
    },
    "wearing_parameterisation": {
        "title": "Wearing parameterisation",
        "labeler": lambda d: f"{d['exp_config']['wearing_parameterisation']}",
        "default_label": "Exponential"
    },
    "wearing_prior_scale": {
        "title": "Wearing effect prior scale",
        "labeler": lambda d: f"{d['exp_config']['wearing_sigma']}",
        "default_label": "Default (0.4)"
    },
    "mobility_sigma": {
        "title": "Mobility effect prior scale",
        "labeler": lambda d: f"{d['exp_config']['mobility_sigma']}",
        "default_label": "Default (0.44)"
    },
    "mobility_leaveout": {
        "title": "Mobility leave-out",
        "labeler": lambda d: f"No mobility",
        "default_label": "Mobility"
    },
    "cases_delay_disp_mean": {
        "title": "Case delay dispersion",
        "labeler": lambda d: f"{d['exp_config']['cases_delay_disp_mean']}",
        "default_label": "Default (5.41)"
    },
    "mob_and_wearing_only": {
        "title": "Mobility and wearing only",
        "labeler": lambda d: f"Mob & Wearing Only",
        "default_label": "Default"
    },
    "mobility_mean": {
        "title": "Mobility effect prior mean",
        "labeler": lambda d: f"{d['exp_config']['mobility_mean']}",
        "default_label": "Default (1.704)"
    },
    "fake_wearing_npi": {
        "title": "Fake wearing covariate",
        "labeler": lambda d: f"Fake Wearing NPI",
        "default_label": "Default"
    },
    "window_of_analysis": {
        "title": "Window of analysis",
        "labeler": lambda d: f"{d['exp_config']['start_date']} to {d['exp_config']['end_date']}",
        "default_label": "Default"
    },
    "mask_leave_on": {
        "title": "Mandate leave-on",
        "labeler": lambda d: f"Mandate Leave-on",
        "default_label": "Default"
    },
    "mask_thresholds": {
        "title": "Number of mandate features",
        "labeler": lambda d: f"1",
        "default_label": "Default (2)"
    }
}

class experiment_type():
    def __init__(self, experiments, exp_info, tag):
        self.exp_info = exp_info
        self.experiments = experiments
        self.exp_info["tag"] = tag

        if "All_CM_Reductions" in list(self.experiments[0].keys()):
            self.experiments.sort(key=lambda x: np.median(np.array(x['All_CM_Reductions'])[:, 0]))

def get_unique_exp_tags(experiments):
    return list(np.unique([exp['exp_tag'] for exp in experiments]))

def make_all_experiment_classes(all_experiments):
    tags = get_unique_exp_tags(all_experiments)
    classes = []
    for tag in tags:
        filtered_exps = filter_by_exp_tag(all_experiments, tag)
        if tag in all_exp_info.keys():
            exp_info = all_exp_info[tag]
            classes.append(experiment_type(filtered_exps, exp_info, tag))
        else:
            print(f"{tag} was skipped")

    return classes


grouped_npis_wearing = {
    'Mask-wearing': {
        'npis': ['percent_mc'],
        'type': 'include',
        'color': cols[0],
        'main': True,
    },
    'Mobility': {
        'npis': ['avg_mobility_no_parks_no_residential'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'School closing': {
        'npis': ['C1_School closing'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'School closing: full': {
        'npis': ['C1_School closing_full', 'C1_School closing'],
        'type': "exclude",
        'color': cols[0],
        'main': True,
    },
    'Workplace closing': {
        'npis': ['C2_Workplace closing'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Workplace closing: full': {
        'npis': ['C2_Workplace closing_full', 'C2_Workplace closing'],
        'type': "exclude",
        'color': cols[0],
        'main': True,
    },
    'Restrictions on gatherings: 3+': {
        'npis': ['C4_Restrictions on gatherings_3plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
        'Restrictions on gatherings; 2+': {
        'npis': ['C4_Restrictions on gatherings_2plus', 'C4_Restrictions on gatherings_3plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Restrictions on gatherings: full': {
        'npis': ['C4_Restrictions on gatherings_full', 'C4_Restrictions on gatherings_3plus', 'C4_Restrictions on gatherings_2plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Stay at home requirements': {
        'npis': ['C6_Stay at home requirements'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Restrictions on internal movement': {
        'npis': ['C7_Restrictions on internal movement'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    }
}


grouped_npis_mob_and_wearing_only = {
    'Mask-wearing': {
        'npis': ['percent_mc'],
        'type': 'include',
        'color': cols[0],
        'main': True,
    },
    'Mobility': {
        'npis': ['avg_mobility_no_parks_no_residential'],
        'type': 'include',
        'color': cols[0],
        'main': True,
    }
}


grouped_npis_wearing_mobility_leaveout = {
    'Mask-wearing': {
        'npis': ['percent_mc'],
        'type': 'include',
        'color': cols[0],
        'main': True,
    },
    'School closing': {
        'npis': ['C1_School closing'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'School closing: full': {
        'npis': ['C1_School closing_full', 'C1_School closing'],
        'type': "exclude",
        'color': cols[0],
        'main': True,
    },
    'Workplace closing': {
        'npis': ['C2_Workplace closing'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Workplace closing: full': {
        'npis': ['C2_Workplace closing_full', 'C2_Workplace closing'],
        'type': "exclude",
        'color': cols[0],
        'main': True,
    },
    'Restrictions on gatherings: 3+': {
        'npis': ['C4_Restrictions on gatherings_3plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
        'Restrictions on gatherings; 2+': {
        'npis': ['C4_Restrictions on gatherings_2plus', 'C4_Restrictions on gatherings_3plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Restrictions on gatherings: full': {
        'npis': ['C4_Restrictions on gatherings_full', 'C4_Restrictions on gatherings_3plus', 'C4_Restrictions on gatherings_2plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Stay at home requirements': {
        'npis': ['C6_Stay at home requirements'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Restrictions on internal movement': {
        'npis': ['C7_Restrictions on internal movement'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    }
}


grouped_npis_mandates = {
    'Mask mandates': {
        'npis': ['H6_Facial Coverings'],
        'type': 'include',
        'color': cols[0],
        'main': True,
    },
    'Mobility': {
        'npis': ['avg_mobility_no_parks_no_residential'],
        'type': 'include',
        'color': cols[0],
        'main': True,
    },
    'School closing': {
        'npis': ['C1_School closing'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'School closing: full': {
        'npis': ['C1_School closing_full', 'C1_School closing'],
        'type': "exclude",
        'color': cols[0],
        'main': True,
    },
    'Workplace closing': {
        'npis': ['C2_Workplace closing'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Workplace closing: full': {
        'npis': ['C2_Workplace closing_full', 'C2_Workplace closing'],
        'type': "exclude",
        'color': cols[0],
        'main': True,
    },
    'Restrictions on gatherings: 3+': {
        'npis': ['C4_Restrictions on gatherings_3plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
        'Restrictions on gatherings; 2+': {
        'npis': ['C4_Restrictions on gatherings_2plus', 'C4_Restrictions on gatherings_3plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Restrictions on gatherings: full': {
        'npis': ['C4_Restrictions on gatherings_full', 'C4_Restrictions on gatherings_3plus', 'C4_Restrictions on gatherings_2plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Stay at home requirements': {
        'npis': ['C6_Stay at home requirements'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Restrictions on internal movement': {
        'npis': ['C7_Restrictions on internal movement'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    }
}


grouped_npis_mandates_mobility_leaveout = {
    'Mask mandates': {
        'npis': ['H6_Facial Coverings'],
        'type': 'include',
        'color': cols[0],
        'main': True,
    },
    'School closing': {
        'npis': ['C1_School closing'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'School closing: full': {
        'npis': ['C1_School closing_full', 'C1_School closing'],
        'type': "exclude",
        'color': cols[0],
        'main': True,
    },
    'Workplace closing': {
        'npis': ['C2_Workplace closing'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Workplace closing: full': {
        'npis': ['C2_Workplace closing_full', 'C2_Workplace closing'],
        'type': "exclude",
        'color': cols[0],
        'main': True,
    },
    'Restrictions on gatherings: 3+': {
        'npis': ['C4_Restrictions on gatherings_3plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
        'Restrictions on gatherings; 2+': {
        'npis': ['C4_Restrictions on gatherings_2plus', 'C4_Restrictions on gatherings_3plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Restrictions on gatherings: full': {
        'npis': ['C4_Restrictions on gatherings_full', 'C4_Restrictions on gatherings_3plus', 'C4_Restrictions on gatherings_2plus'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Stay at home requirements': {
        'npis': ['C6_Stay at home requirements'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    },
    'Restrictions on internal movement': {
        'npis': ['C7_Restrictions on internal movement'],
        'type': 'exclude',
        'color': cols[0],
        'main': True,
    }
}

def combine_npi_samples(grouped_npis, alpha_i_samples, npi_names):
    nS, nCMs_orig = alpha_i_samples.shape
    CMs_new = list(grouped_npis.keys())
    CMs_new_include = np.ones(len(CMs_new), dtype=np.bool)
    nCMs_new = len(CMs_new)

    new_samples = np.zeros((nS, nCMs_new))

    # print(npi_names)

    for cm_i_new, (gnpi, sub_npilist_dict) in enumerate(grouped_npis.items()):
        sub_npilist = sub_npilist_dict["npis"]
        add_type = sub_npilist_dict["type"]
        count = 0
        for cm in sub_npilist:
            if cm in npi_names:
                # print(f'adding {cm} to {CMs_new[cm_i_new]}')
                if count == 0:
                    new_samples[:, cm_i_new] += alpha_i_samples[:, npi_names.index(cm)]
                else:
                    new_samples[:, cm_i_new] += alpha_i_samples[:, npi_names.index(cm)] - 1
                count += 1
            elif add_type == "exclude":
                # print(f'excluding {cm}')
                CMs_new_include[cm_i_new] = False
    CMs_new = np.array(CMs_new)[CMs_new_include].tolist()
    new_samples = new_samples[:, CMs_new_include]

    return new_samples, CMs_new


def add_trace_to_plot(samples, y_off, col, label, alpha, width, npi_comb_dict, cm_names, size=6):
    comb_effects, new_names = combine_npi_samples(npi_comb_dict, samples, cm_names)
    comb_effects = 100*(1-comb_effects)
    npi_order = list(npi_comb_dict.keys())
    nF = len(npi_order)

    y_vals = -np.array([npi_order.index(name) for name in new_names])
    plt.plot([100], [100], color=col, linewidth=1, alpha=alpha, label=label)

    li, lq, m, uq, ui = np.percentile(comb_effects, [2.5, 25, 50, 75, 97.5], axis=0)
    plt.scatter(m, y_vals+y_off, marker="o", color=col, s=size, alpha=alpha, facecolor='white', zorder=3, linewidth=width/2)
    for cm in range(len(new_names)):
        plt.plot([li[cm], ui[cm]], [y_vals[cm]+y_off, y_vals[cm]+y_off], color=col, alpha=alpha*0.25, linewidth=width, zorder=2)
        plt.plot([lq[cm], uq[cm]], [y_vals[cm]+y_off, y_vals[cm]+y_off], color=col, alpha=alpha*0.75, linewidth=width, zorder=2)


def setup_plot(experiment_class, npi_comb_dict, y_ticks = True, xlabel=True, x_lims=(-25, 50), axis=None):
    if axis is None:
        plt.figure(figsize=(4, 6), dpi=400)
    else:
        plt.sca(axis)

    x_min, x_max = x_lims

    npi_order = list(npi_comb_dict.keys())
    plt.plot([0, 0], [1, -(len(npi_order)+2)], "--k", linewidth=0.5)

    xrange = np.array([x_min, x_max])

    for height in range(0, len(npi_order)+2, 2):
        plt.fill_between(xrange, -(height-0.5), -(height+0.5), color="silver", alpha=0.25, linewidth=0)
    xtick_vals = [-25, 0, 25, 50, 75, 100]
    xtick_str = [f"{x:.0f}%" for x in xtick_vals]

    if y_ticks:
        plt.yticks(-np.arange(len(npi_order)), npi_order, fontsize=6)
    else:
        plt.yticks([])

    plt.xticks(xtick_vals, xtick_str, fontsize=8)
    plt.xlim([x_min, x_max])
    plt.ylim([-(len(npi_order) - 0.25), 0.75])

    plt.plot([-100, 100], [-0.5, -0.5], 'k')

    if xlabel:
        plt.xlabel("Reduction in R", fontsize=8)

colors = [*sns.color_palette("colorblind"), *sns.color_palette("dark")]

def plot_experiment_class(experiment_class, npi_comb_dict, x_lims=None, default_res=None, default_names=None, width=1, axis=None):
    default_label = experiment_class.exp_info["default_label"]
    labeler = experiment_class.exp_info["labeler"]

    if 'Mask mandates' in npi_comb_dict.keys():
        title = experiment_class.exp_info["title"] + ' (Mandates model)'
    else:
        title = experiment_class.exp_info["title"] + ' (Wearing model)'

    n_max = 6

    if len(experiment_class.experiments) > n_max:
        print('splitting inds')
        n_plots = int(np.ceil(len(experiment_class.experiments) / n_max))
        all_indices = np.array_split(range(len(experiment_class.experiments)), n_plots)
        for j, indices in enumerate(all_indices):
            setup_plot(experiment_class, npi_comb_dict, x_lims=x_lims, axis=axis)
            print('set up plot')
            y_off = -np.linspace(-0.3, 0.3, len(indices)+1)
            for i, trace in enumerate(experiment_class.experiments[indices[0]:indices[-1]+1]):
                add_trace_to_plot(np.array(trace['All_CM_Reductions']), y_off[i], colors[i], labeler(trace), alpha=1,
                                  width=width, npi_comb_dict=npi_comb_dict, cm_names=trace['cm_names'] )
            if default_res is not None:
                add_trace_to_plot(default_res, y_off[-1], "k", default_label, alpha=1,
                                  width=width, npi_comb_dict=npi_comb_dict, cm_names=default_names)
            print('added traces')
            plt.legend(shadow=True, fancybox=True, loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=6)
            plt.title(title, fontsize=10)
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            if axis is None:
                plt.savefig(f'{fig_path}/Fig_{experiment_class.exp_info["tag"]}_{j}.pdf', bbox_inches='tight')
                print('saved')
    else:
        setup_plot(experiment_class, npi_comb_dict, x_lims=x_lims, axis=axis)
        y_off = -np.linspace(-0.3, 0.3, len(experiment_class.experiments)+1)
        for i, trace in enumerate(experiment_class.experiments):

            add_trace_to_plot(np.array(trace['All_CM_Reductions']), y_off[i], colors[i], labeler(trace), alpha=1,
                                  width=width, npi_comb_dict=npi_comb_dict, cm_names=trace['cm_names'] )

        if default_res is not None:
            add_trace_to_plot(default_res, y_off[-1], "k", default_label, alpha=1,
                              width=width, npi_comb_dict=npi_comb_dict, cm_names=default_names)

        plt.legend(shadow=True, fancybox=True, loc="upper right", bbox_to_anchor=(0.99, 0.99), fontsize=6)
        plt.title(title, fontsize=10)
        if not os.path.exists(fig_path):
            os.mkdir(fig_path)
        if axis is None:
            plt.savefig(f'{fig_path}/Fig_{experiment_class.exp_info["tag"]}.pdf', bbox_inches='tight')


default_dir = base_dir + 'default'
# default_dir = base_dir + 'default/default'
default_experiments = get_all_experiments(default_dir)
default_experiment_classes = make_all_experiment_classes(default_experiments)
default_res = default_experiment_classes[0].experiments[0]
default_alpha = default_res['All_CM_Reductions']
default_names = default_res['cm_names']

#%%

categories = ['basic_R_hyperprior_mean_mean',
              'basic_R_hyperprior_mean_scale',
              'basic_R_prior_scale',
              'cases_delay_mean',
              'gen_int_mean',
              'intervention_prior',
              'npi_leaveout',
              'r_walk_noise_scale_prior',
              'r_walk_period',
              'boostrap',
              'mobility_mean',
              'mobility_sigma',
              'mobility_leaveout',
              'cases_delay_disp_mean',
              'mob_and_wearing_only',
              'window_of_analysis'
]

categories_wearing = [
    'wearing_parameterisation',
                    'wearing_prior_scale',
                    'fake_wearing_npi',
                      ]

categories_mandates = ['mask_sigma',
                       'mask_leave_on',
                       'mask_thresholds']

categories += categories_wearing
categories += categories_mandates


## Seperate sections:

#%%

for category in categories:
    try:
        res_dir = base_dir + category
        all_experiments = get_all_experiments(res_dir)
        tags = get_unique_exp_tags(all_experiments)
        len(all_experiments)
        experiment_classes = make_all_experiment_classes(all_experiments)

        for experiment_class in experiment_classes:
            if experiment_class.exp_info['tag'] == "default":
                continue
            else:
                if 'mandates' in base_dir:
                    if experiment_class.exp_info['tag'] == "mobility_leaveout":
                        grouped_npis_dict = grouped_npis_mandates_mobility_leaveout
                    else:
                        grouped_npis_dict = grouped_npis_mandates
                else:
                    if experiment_class.exp_info['tag'] == "mobility_leaveout":
                        grouped_npis_dict = grouped_npis_wearing_mobility_leaveout
                    elif experiment_class.exp_info['tag'] == "mob_and_wearing_only":
                        grouped_npis_dict = grouped_npis_mob_and_wearing_only
                    else:
                        grouped_npis_dict = grouped_npis_wearing

                plot_experiment_class(experiment_class, grouped_npis_dict,
                                      (-25, 100), default_alpha, default_names)
    except Exception as e:
        print(e)
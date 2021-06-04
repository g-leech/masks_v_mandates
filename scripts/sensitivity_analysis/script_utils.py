"""
:code:`script_utils.py`
Utilities to support the use of command line sensitivity experiments
"""
import os

import numpy as np
import yaml

from epimodel.pymc3_models.mask_models import *


def get_target_accept_from_model_str(model_type_str):
    # default
    return 0.9


def get_tree_depth_from_model_str(model_type_str):
    # default
    return 12


def add_argparse_arguments(argparse):
    """
    add argparse arguments to scripts
    :param argparse: argparse object
    """
    argparse.add_argument(
        "--model_type",
        dest="model_type",
        type=str,
        help="""model""",
    )
    argparse.add_argument(
        "--exp_tag", dest="exp_tag", type=str, help="experiment identification tag"
    )
    argparse.add_argument(
        "--num_chains",
        dest="num_chains",
        type=int,
        help="the number of chains to run in parallel",
    )
    argparse.add_argument(
        "--num_samples",
        dest="num_samples",
        type=int,
        help="the number of samples to draw",
    )

    argparse.add_argument(
        "--num_warmup",
        dest="num_warmup",
        type=int,
        help="the number of warmup samples to draw",
    )

    argparse.add_argument(
        "--model_config",
        dest="model_config",
        type=str,
        help="model config file, which is used for overriding default options",
    )


def load_model_config(model_config_str):
    with open("sensitivity_analysis/model_configs.yaml", "r") as stream:
        try:
            model_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return model_config[model_config_str]


def pprint_mb_dict(d):
    """
    pretty print dictionary
    :param d:
    :return:
    """
    print("Model Build Dict" "----------------")

    for k, v in d.items():
        print(f"    {k}: {v}")


def generate_base_output_dir(model_type, model_config, exp_tag):
    """
    standardise output directory
    :param model_type:
    :param model_config:
    :param exp_tag:
    :return: output directory
    """
    if 'Mandate' in model_type:
        type = 'mandates'
    else:
        type = 'wearing'

    out_path = os.path.join(
        "/mnt/sensitivity_analysis", f"1_non_reopenings_full", type, exp_tag
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    return out_path

def get_data_path(model_type):
    if 'MobilityModel' in model_type:
        print('Using mobility dataset')
        return "../data/modelling_set/master_data_mob_include_us_True_m_w.csv"
    else:
        print('Using non-mobility dataset')
        return "../data/modelling_set/master_data_mob_exclude_us_True_m_w.csv"

def load_keys_from_samples(keys, posterior_samples, summary_dict):
    for k in keys:
        if k in posterior_samples.keys():
            # save to list
            summary_dict[k] = np.asarray(posterior_samples[k]).tolist()
    return summary_dict

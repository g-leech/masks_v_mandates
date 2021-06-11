import sys, os

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# sys.path.append(os.getcwd())  # add current working directory to the path
sys.path.append("..")
import argparse
from datetime import datetime
import pickle
import pymc3 as pm
import json

from epimodel.preprocessing.preprocess_mask_data import Preprocess_masks
from epimodel.pymc3_models.mask_models import (
    MandateMobilityModel,
    RandomWalkMobilityModel,
    RandomWalkModelFixedCaseDelay,
    RandomWalkModel,
    MobilityModel,
    Base_Model,
    CasesOnlyModel,
    CasesDeathsModel,
)
from scripts.sensitivity_analysis.script_utils import *



argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--npis", dest="npis", type=int, help="NPI indices to leave out", nargs="+"
)

add_argparse_arguments(argparser)
args = argparser.parse_args()


if __name__ == "__main__":
    print(f"Running Sensitivity Analysis {__file__} with config:")
    config = load_model_config(args.model_config)
    pprint_mb_dict(config)

    print("Loading data")
    masks_object = Preprocess_masks(get_data_path(args.model_type))
    featurize_kwargs = config["featurize_kwargs"]
    if 'Mandate' in args.model_type:
        print('Using mandates')
        featurize_kwargs['masks'] = 'mandate'

    featurize_kwargs['npi_leaveout_inds'] = []
    for npi in args.npis:
        if npi > masks_object.nCMs:
            print(
                f"You tried to remove NPI index {npi}, but there are only {masks_object.nCMs} npis"
            )
            sys.exit()
        featurize_kwargs['npi_leaveout_inds'].append(npi)

    masks_object.featurize(**featurize_kwargs)
    masks_object.make_preprocessed_object()
    data = masks_object.data

    print("Loading model args")
    model_kwargs = config["model_kwargs"]

    first_day_confirmed = data.Confirmed[:, 0]
    median_init_size = np.median(first_day_confirmed)
    model_kwargs["log_init_mean"] = np.log(median_init_size)
    model_kwargs["log_init_sd"] = np.log(median_init_size)

    if 'Mandate' in args.model_type:
        print('Using MandateMobilityModel')
        with MandateMobilityModel(data) as model:
            model.build_model(**model_kwargs)
    else:
        print('Using RandomWalkMobilityModel')
        with RandomWalkMobilityModel(data) as model:
            model.build_model(**model_kwargs)

    ta = get_target_accept_from_model_str(args.model_type)
    td = get_tree_depth_from_model_str(args.model_type)

    base_outpath = generate_base_output_dir(
        args.model_type, args.model_config, args.exp_tag
    )
    ts_str = datetime.now().strftime("%Y-%m-%d;%H:%M:%S")
    summary_output = os.path.join(base_outpath, f"{ts_str}_summary.json")
    full_output = os.path.join(base_outpath, f"{ts_str}_full.netcdf")

    info_dict = {}
    info_dict["model_config_name"] = args.model_config
    info_dict["model_kwargs"] = model_kwargs
    info_dict["featurize_kwargs"] = featurize_kwargs
    info_dict["start_dt"] = ts_str
    info_dict["exp_tag"] = args.exp_tag
    info_dict["exp_config"] = {"npis": args.npis}
    info_dict["cm_names"] = data.CMs
    info_dict["data_path"] = get_data_path(args.model_type)

    with model:
        model.trace = pm.sample(
            args.num_samples,
            tune=args.num_warmup,
            cores=args.num_chains,
            chains=args.num_chains,
            max_treedepth=td,
            target_accept=ta,
            init="adapt_diag",
        )

        print('Building info dict')
        ReductionNames = []
        if 'Only' not in args.model_type:
            info_dict['CMReduction'] = np.array(model.trace.CMReduction).tolist()
            ReductionNames.append('CMReduction')
        if 'MobilityModel' in args.model_type:
            info_dict['MobilityReduction'] = np.array(model.trace.MobilityReduction).tolist()
            ReductionNames.append('MobilityReduction')
        if 'Mandate' in args.model_type:
            info_dict['MandateReduction'] = np.array(model.trace.MandateReduction).tolist()
            ReductionNames.append('MandateReduction')
        else:
            info_dict['WearingReduction'] = np.array(model.trace.WearingReduction).tolist()
            ReductionNames.append('WearingReduction')
            info_dict['Wearing_Alpha'] = np.array(model.trace.Wearing_Alpha).tolist()

        info_dict['ReductionNames'] = ReductionNames
        info_dict['varnames'] = np.array(model.trace.varnames).tolist()
        info_dict['summary'] = np.array(pm.summary(model.trace, hdi_prob=0.95)).tolist()
        info_dict['ExpectedLogR'] = np.array(model.trace.ExpectedLogR).tolist()
        info_dict['Rt_walk'] = np.array(model.trace.Rt_walk).tolist()
        info_dict['Rt_cm'] = np.array(model.trace.Rt_cm).tolist()
        info_dict['divergences'] = np.array(model.trace["diverging"]).tolist()

    print("Saving json")
    try:
        with open(summary_output, "w") as f:
            json.dump(info_dict, f, ensure_ascii=False, indent=4)
        print(f'file saved in {summary_output}')
    except Exception as e:
        print(e)


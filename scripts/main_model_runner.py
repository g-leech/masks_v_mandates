import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

np.random.seed(123456)

import sys
import argparse
import datetime
import pickle

import pymc3 as pm

from epimodel import EpidemiologicalParameters
from epimodel.preprocessing.preprocess_mask_data import Preprocess_masks
from epimodel.pymc3_models.mask_models import (
    RandomWalkMobilityModel,
    MandateMobilityModel
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--model", dest="model", type=str, help="Model type")
argparser.add_argument("--masks", dest="masks", type=str, help="Which mask feature")
argparser.add_argument(
    "--w_par", dest="w_par", type=str, help="Which wearing parameterisation"
)
argparser.add_argument("--mob", dest="mob", type=str, help="How to include mobility")

# argparser.add_argument('--filter', dest='filtered', type=str, help='How to remove regions')
# argparser.add_argument('--gatherings', dest='gatherings', type=int, help='how many gatherings features')
argparser.add_argument("--tuning", dest="tuning", type=int, help="tuning samples")
argparser.add_argument("--draws", dest="draws", type=int, help="draws")
argparser.add_argument("--chains", dest="chains", type=int, help="chains")
# argparser.add_argument('--hide_ends', dest='hide_ends', type=str)
args, _ = argparser.parse_known_args()

MODEL = args.model
MASKS = args.masks
W_PAR = args.w_par if args.w_par else "exp"
MOBI = args.mob
TUNING = args.tuning if args.tuning else 1000
DRAWS = args.draws if args.draws else 500
CHAINS = args.chains if args.chains else 4
# FILTERED = args.filtered

US = True
SMOOTH = False
GATHERINGS = 3  # args.gatherings if args.gatherings else 3
# MASKING = True # Always true


# prep data object
path = f"data/modelling_set/master_data_mob_{MOBI}_us_{US}_m_w.csv"
print(path)
masks_object = Preprocess_masks(path)
masks_object.featurize(gatherings=GATHERINGS, masks=MASKS, smooth=SMOOTH, mobility=MOBI)
masks_object.make_preprocessed_object()
data = masks_object.data


# model specification
ep = EpidemiologicalParameters()
bd = ep.get_model_build_dict()


def set_init_infections(data, d):
    n_masked_days = 10
    first_day_new = data.NewCases[:, n_masked_days]
    first_day_new = first_day_new[first_day_new.mask == False]
    median_init_size = np.median(first_day_new)
    print(median_init_size)

    if median_init_size == 0:
        median_init_size = 50

    d["log_init_mean"] = np.log(median_init_size)
    d["log_init_sd"] = np.log(median_init_size) 


set_init_infections(data, bd)

bd["wearing_parameterisation"] = W_PAR


if MODEL == "cases":
    del bd["deaths_delay_mean_mean"]
    del bd["deaths_delay_mean_sd"]
    del bd["deaths_delay_disp_mean"]
    del bd["deaths_delay_disp_sd"]


print(bd)


if MASKS == "wearing":
    with RandomWalkMobilityModel(data) as model:
        model.build_model(**bd)
elif MASKS == "mandate":
    with MandateMobilityModel(data) as model:
        model.build_model(**bd)


MASS = "adapt_diag"  # Originally: 'jitter+adapt_diag'


with model:
    model.trace = pm.sample(
        DRAWS,
        tune=TUNING,
        cores=CHAINS,
        chains=CHAINS,
        max_treedepth=12,
        target_accept=0.9,
        init=MASS,
    )


dt = datetime.datetime.now().strftime("%m-%d-%H:%M")

if MASKS == "wearing":
    idstr = f"pickles/{MASKS}_{W_PAR}_{MODEL}_{len(data.Rs)}_{MOBI}_{dt}"
else:
    idstr = f"pickles/{MASKS}_{MODEL}_{len(data.Rs)}_{MOBI}_{dt}"

pickle.dump(model.trace, open(idstr + ".pkl", "wb"))

with open(idstr + "_cols", "w") as f:
    f.write(", ".join(data.CMs))

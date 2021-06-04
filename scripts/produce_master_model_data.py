import sys

sys.path.append("..")
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.signal.conv as C

from epimodel import EpidemiologicalParameters, preprocess_data

np.random.seed(123456)

import argparse
import copy
import datetime
import itertools
import pickle
import re
from datetime import timedelta

import matplotlib.dates as mdates
import pandas as pd
import pymc3 as pm
from sklearn import preprocessing

US = True
MOBI = "include"  # "exclude"
RADER_JOINT = False
GATHERINGS = 3
# MASKING = True # Now always true
# smoothing happens at data init time if at all

assert MOBI in ["exclude", "only", "include"]


mob_cols = [
    "avg_mobility_no_parks_no_residential",
    "residential_percent_change_from_baseline",
]

# Ds = pd.date_range('2020-05-01', '2020-09-01', freq='D')
# Adding 20 days to account for death delay
Ds = pd.date_range("2020-05-01", "2020-09-21", freq="D")


def fill_missing_days(df):
    df = df.set_index(["date", "country"])
    df = df.unstack(fill_value=-1).asfreq("D", fill_value=-1).stack().reset_index()
    df = df.replace(-1, np.nan)

    return interpolate_wearing_fwd_bwd(df)


def interpolate_wearing_fwd_bwd(df):
    regions = df.country.unique()
    cs = []

    for r in regions:
        c = df[df.country == r]
        c = c.set_index("date")
        c = c.interpolate(method="time", limit_direction="both").reset_index()
        cs.append(c)

    return pd.concat(cs)


# moving average
def smooth(x, window_len=7):
    l = window_len
    s = np.r_[x[l - 1 : 0 : -1], x, x[-2 : -l - 1 : -1]]
    w = np.ones(window_len, "d")

    return np.convolve(w / w.sum(), s, mode="valid")


def smooth_rader(df, win=7):
    for r in df.label.unique():
        s = df[df.label == r]
        s["percent_mc"] = smooth(s["percent_mc"], window_len=win)[: -win + 1]
        df[df.label == r] = s

    return df


def joint_shop_work(df, THRESHOLD=2):
    return (df.likely_wear_mask_grocery_shopping <= THRESHOLD) & (
        df.likely_wear_mask_workplace <= THRESHOLD
    )


def mean_shop_work(df, THRESHOLD=2):
    venues = ["likely_wear_mask_grocery_shopping", "likely_wear_mask_workplace"]
    df["percent_mc"] = df[venues].mean(axis=1)

    return df["percent_mc"] <= THRESHOLD


def load_and_clean_rader(THRESHOLD=2, SMOOTH_RADER=True):  # or less
    DATA_IN = "../data/raw/"
    directory = DATA_IN + "rader/sm_cny_data_1_21_21.csv"
    us = pd.read_csv(directory)

    masks = [
        "likely_wear_mask_exercising_outside",
        "likely_wear_mask_grocery_shopping",
        "likely_wear_mask_visit_family_friends",
        "likely_wear_mask_workplace",
    ]
    # weights = ["weight_daily_national_13plus", "weight_state_weekly"]
    us = us[["response_date", "state"] + masks]  # + weights

    codes = pd.read_excel(DATA_IN + "rader/cny_sm_codebook_2_5_21.xls")
    num2name = codes[codes["column"] == "state"][["value", "label"]]
    us = pd.merge(us, num2name, left_on="state", right_on="value").drop(
        ["value", "state"], axis=1
    )
    us["response_date"] = pd.to_datetime(us["response_date"])

    if RADER_JOINT:
        us["percent_mc"] = joint_shop_work(us, THRESHOLD)
    else:
        us["percent_mc"] = mean_shop_work(us, THRESHOLD)

    us = (
        us[["response_date", "label", "percent_mc"]]
        .groupby(["response_date", "label"])
        .mean()
        .reset_index()
    )
    if SMOOTH_RADER:
        us = smooth_rader(us)

    return us


def add_dummy_wearing_us(us, backfill=True):
    rader_start = us.date.iloc[0] - timedelta(days=1)
    fill_days = pd.date_range(Ds[0], rader_start, freq="D")

    Rs = us.country.unique()

    if backfill:
        for s in Rs:
            df = pd.DataFrame(columns=["date", "country", "percent_mc"])
            df.date = fill_days
            df.country = s
            fill = us.set_index(["country", "date"]).loc[s].percent_mc.iloc[0]
            df.percent_mc = fill
            us = pd.concat([df, us])
    # totally random dummy
    else:
        for s in us.country.unique():
            df = pd.DataFrame(columns=["date", "country", "percent_mc"])
            df.date = fill_days
            df.country = s
            df.percent_mc = np.random.random(len(df))
            us = pd.concat([df, us])

    us = us.sort_values(["date", "country"])
    return us


def load_and_clean_wearing():
    wearing = pd.read_csv(
        "../data/raw/umd/umd_national_wearing.csv",
        parse_dates=["survey_date"],
        infer_datetime_format=True,
    ).drop_duplicates()
    wearing = wearing[(wearing.survey_date >= Ds[0]) & (wearing.survey_date <= Ds[-1])]
    cols = ["country", "survey_date", "percent_mc"]
    wearing = wearing[cols]
    cols = ["country", "date", "percent_mc"]
    wearing.columns = cols

    # Append US
    us_wearing = load_and_clean_rader()
    us_wearing.columns = ["date", "country", "percent_mc"]
    us_wearing = us_wearing[cols]
    us_wearing = us_wearing.replace("Georgia", "Georgia-US")
    us_wearing = us_wearing.replace("District of Columbia (DC)", "District of Columbia")
    # Add dummy wearing back to 1st May
    us_wearing = add_dummy_wearing_us(us_wearing, backfill=True)
    wearing = pd.concat([wearing, us_wearing])

    return fill_missing_days(wearing)


def get_npi_names(df):
    cs = [1, 2, 4, 6, 7]
    npis = []
    for i in cs:
        npi = [c for c in df.columns if f"C{i}" in c]
        npi = [c for c in npi if f"Flag" not in c][0]
        npis += [npi]
    npis += ["H6_Facial Coverings"]

    return npis


def add_diffs(df, npis):
    Rs = df.CountryName.unique()
    df = df.set_index("CountryName")

    for c in Rs:
        df.loc[c, "H6_diff"] = df.loc[c]["H6_Facial Coverings"].diff()

        for npi in npis:
            i = npi[:2]
            df.loc[c, f"{i}_diff"] = df.loc[c][npi].diff()
            df.loc[c, f"{i}_flag_diff"] = df.loc[c][f"{i}_Flag"].diff()

    return df.reset_index()


# Measure increases and flag 1 -> 0
def detect_regional_increase_national_off(df, npi):
    return df[(df[f"{npi}_diff"] > 0) & (df[f"{npi}_flag_diff"] < 0)]


# Measure flag 0 -> 1
def detect_national_back_on(df, npi):
    return df[df[f"{npi}_flag_diff"] > 0]
    # & (df[f"{npi}_diff"] < 0) # Decrease only


def get_previous_national_value(start, df, country, npi):
    previousValDate = start - timedelta(days=1)
    c = df[df.CountryName == country]
    previousValDf = c[c.date == previousValDate][npi]
    l = list(previousValDf)

    val = l[0]

    return val


def impute_country_switch(row, df, npi, country_ends):
    country = row["CountryName"]
    start = row["date"]
    code = npi[:2]

    # Want the val of the day before the regional change
    if start == Ds[0]:
        return

    previousVal = get_previous_national_value(start, df, country, npi)

    isChangeImputed = False

    # Only impute once per regional change:
    while not isChangeImputed:
        for _, end in country_ends.iterrows():
            if end["date"] <= start:
                continue

            # `between` is inclusive so trim last day:
            end = end["date"] - timedelta(days=1)
            df.loc[
                (df.CountryName == country) & (df.date.between(start, end)), npi
            ] = previousVal
            df.loc[
                (df.CountryName == country) & (df.date.between(start, end)),
                code + "_Flag",
            ] = 1
            isChangeImputed = True
        break

    # if OXCGRT never returns to national flag,
    # impute to end of our window
    if isChangeImputed == False:
        end = Ds[-1]
        df.loc[
            (df.CountryName == country) & (df.date.between(start, end)), npi
        ] = previousVal
        df.loc[
            (df.CountryName == country) & (df.date.between(start, end)), code + "_Flag"
        ] = 1

    return df


# Find regional NPI increases which obscure national NPIs
# Find dates of dataset returning to national policy
# Fill regional increases with previous level and the flag=1
# Up to next national measurement date
def find_and_impute_npi(df, npi, diffed):
    code = npi[:2]
    diff_col = code + "_diff"
    flag_diff = code + "_flag_diff"

    imputation_starts = detect_regional_increase_national_off(diffed, code)
    imputation_ends = detect_national_back_on(diffed, code)

    for i, row in imputation_starts.iterrows():
        country = row["CountryName"]
        country_ends = imputation_ends[imputation_ends.CountryName == country]
        df = impute_country_switch(row, df, npi, country_ends)

    return df


def fix_regional_overwrite(oxcgrt, npis):
    npis = get_npi_names(oxcgrt)
    diffed = add_diffs(oxcgrt, npis)

    for npi in npis:
        oxcgrt = find_and_impute_npi(oxcgrt, npi, diffed)

    check_imputation(oxcgrt)

    return oxcgrt


def load_oxcgrt(use_us=True):
    OXCGRT_PATH = "../data/raw/OxCGRT_latest.csv"
    oxcgrt = pd.read_csv(OXCGRT_PATH, parse_dates=["Date"], low_memory=False)
    # Drop regional data
    nat = oxcgrt[oxcgrt.Jurisdiction == "NAT_TOTAL"]

    # Add US states
    if use_us:
        states = oxcgrt[
            (oxcgrt.CountryName == "United States")
            & (oxcgrt.Jurisdiction == "STATE_TOTAL")
        ]
        # Drop GEO to prevent name collision
        nat = nat[nat.CountryName != "Georgia"]
        states.CountryName = states.RegionName
        states = states.replace("Georgia", "Georgia-US")
        nat = pd.concat([nat, states])

    i = list(nat.columns).index("Date")
    nat.columns = list(nat.columns[:i]) + ["date"] + list(nat.columns[i + 1 :])

    return nat[nat.date.isin(Ds)]


def clean_oxcgrt(oxcgrt, npis):
    oxcgrt = fix_regional_overwrite(oxcgrt, npis)

    # Threshold and filter to _Flag == 1
    gatherings3 = GATHERINGS == 3

    return threshold_oxcgrt(oxcgrt, gatherings_3=gatherings3)


# TODO: Assumption for now: whenever a country *increases* a policy from A to B but flips national off, the national policy continues at level A until next Flag=1 change

# Relying on the above imputation to ensure that the national flags are sensible
def threshold(df, col, t):
    NATIONAL = 1
    code = col[:2]

    return (df[col] >= t) & (df[f"{code}_Flag"] == NATIONAL)


def threshold_oxcgrt(df, gatherings_3=False):
    df["C1_School closing_full"] = threshold(df, "C1_School closing", 3)  # note order
    df["C1_School closing"] = threshold(df, "C1_School closing", 2)
    df["C2_Workplace closing_full"] = threshold(df, "C2_Workplace closing", 3)
    df["C2_Workplace closing"] = threshold(df, "C2_Workplace closing", 2)

    df["C4_Restrictions on gatherings_3plus"] = threshold(
        df, "C4_Restrictions on gatherings", 3
    )

    if gatherings_3:
        df["C4_Restrictions on gatherings_2plus"] = threshold(
            df, "C4_Restrictions on gatherings", 2
        )
        df["C4_Restrictions on gatherings_full"] = threshold(
            df, "C4_Restrictions on gatherings", 4
        )

    df["C6_Stay at home requirements"] = threshold(
        df, "C6_Stay at home requirements", 2
    )
    df["C7_Restrictions on internal movement"] = threshold(
        df, "C7_Restrictions on internal movement", 2
    )

    if "H6_Facial Coverings" in npi_cols:
        df["H6_Facial Coverings_3plus"] = threshold(df, "H6_Facial Coverings", 3)
        df["H6_Facial Coverings"] = threshold(df, "H6_Facial Coverings", 2)

    i = npi_cols.index("C4_Restrictions on gatherings")
    npi_cols[i] = "C4_Restrictions on gatherings_3plus"

    float_cols = ["percent_mc"] + mob_cols
    oxcols = [c for c in npi_cols_minus_mob if c not in float_cols]
    df[oxcols] = df[oxcols].astype(int)

    check_thresholding(df)

    return df


# TODO: Justify ignoring
# IGNORE_NPIS = ["C3_Cancel public events", "C5_Close public transport", "C8_International travel controls"]


def join_ox_umd(oxcgrt, wearing, npi_cols):
    join = oxcgrt.merge(
        wearing,
        right_on=["country", "date"],
        left_on=["CountryName", "date"],
        suffixes=("", "_"),
    )  # , \
    # how='left')

    return join[npi_cols + ["country", "date", "ConfirmedCases", "ConfirmedDeaths"]]


def load_and_clean_mob(wearing, normalise=True, use_us=True, SMOOTH_MOB=False):
    mob = join_mob(use_us)

    if normalise:
        Rs = mob.country_region.unique()
        mob = mob.set_index("country_region", "date")
        minmax = preprocessing.MinMaxScaler()

        for c in mob_cols:
            for r in Rs:
                mob.loc[r, c] = -1.0 * mob.loc[r, c] / 100
                mob.loc[r, c] = mob.loc[r, c].interpolate()
                if SMOOTH_MOB:
                    l = 7
                    mob.loc[r, c] = smooth(mob.loc[r, c], l)[: -l + 1]

        mob = mob.reset_index()

    join = mob.merge(
        wearing, right_on=["country", "date"], left_on=["country_region", "date"]
    )

    cs = mob_cols + ["date", "country", "percent_mc"]

    return join[cs]


def add_cgrt_responses(mobwearing, oxcgrt):
    cs = ["CountryName", "date", "ConfirmedCases", "ConfirmedDeaths"]
    oxcgrt = oxcgrt[cs]

    join = oxcgrt.merge(
        mobwearing, left_on=["CountryName", "date"], right_on=["country", "date"]
    )

    join = join.drop("CountryName", axis=1)

    return join


## Mobility


def load_global_mob():
    path = "../data/raw/Global_Mobility_Report.csv"
    df = pd.read_csv(path, parse_dates=["date"])

    # Filter to national
    national_row = lambda df: df[
        df.sub_region_1.isna() & df.sub_region_2.isna() & df.metro_area.isna()
    ]
    nat = national_row(df)
    nat = nat.drop("parks_percent_change_from_baseline", axis=1)
    nat = nat.drop(
        [
            "country_region_code",
            "sub_region_1",
            "sub_region_2",
            "metro_area",
            "iso_3166_2_code",
            "census_fips_code",
            "place_id",
        ],
        axis=1,
    )

    subs = {"Czechia": "Czech Republic"}
    nat["country_region"] = nat["country_region"].replace(subs)

    return nat


def get_us_states():
    return list(load_us_mob().country_region.unique())


def load_us_mob():
    path = "../data/raw/2020_US_Region_Mobility_Report.csv"
    us = pd.read_csv(path, parse_dates=["date"])
    us = us.replace("Georgia", "Georgia-US")
    states_plus_counties = us[~us.sub_region_1.isna()]
    states = states_plus_counties[states_plus_counties.sub_region_2.isna()]
    keep = [
        "sub_region_1",
        "date",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    states = states[keep]
    states.columns = ["country_region"] + keep[1:]

    return states


def join_mob(use_us=True):
    nat = load_global_mob()

    if use_us:
        states = load_us_mob()
        nat = pd.concat([nat, states])

    ind = nat.set_index(["country_region", "date"])
    res = ind.residential_percent_change_from_baseline

    ind = ind.drop("residential_percent_change_from_baseline", axis=1)
    ind["avg_mobility_no_parks_no_residential"] = ind.mean(axis=1)
    to_avg = [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
    ]
    ind = ind.drop(to_avg, axis=1)

    return ind.join(res).reset_index()


def build_npi_list(mob):
    npi_cols = []

    if mob != "only":
        npi_cols = [
            "C1_School closing",
            "C2_Workplace closing",
            "C4_Restrictions on gatherings",
            "C6_Stay at home requirements",
            "C7_Restrictions on internal movement",
        ]
    npi_cols += [
        "C4_Restrictions on gatherings_2plus",
        "C4_Restrictions on gatherings_full",
    ]
    npi_cols += ["C1_School closing_full", "C2_Workplace closing_full"]

    npi_cols += ["H6_Facial Coverings", "H6_Facial Coverings_3plus"]
    npi_cols += ["percent_mc"]

    npi_cols_minus_mob = npi_cols

    if mob != "exclude":
        npi_cols += mob_cols

    return npi_cols, npi_cols_minus_mob


def check_imputation(df):
    bswitch = df[(df.CountryName == "Belgium") & (df.date == "2020-07-24")]
    assert bswitch["H6_Facial Coverings"].tolist() == [2.0]
    bswitch = df[(df.CountryName == "Belgium") & (df.date == "2020-07-25")]
    assert bswitch["H6_Facial Coverings"].tolist() == [2.0]
    bswitch = df[(df.CountryName == "Belgium") & (df.date == "2020-07-29")]
    assert bswitch["H6_Facial Coverings"].tolist() == [3.0]
    aswitch = df[(df.CountryName == "Afghanistan") & (df.date == "2020-06-07")]
    assert aswitch["H6_Facial Coverings"].tolist() == [3.0]
    cswitch = df[(df.CountryName == "Connecticut") & (df.date == "2020-07-24")]
    assert aswitch["H6_Facial Coverings"].tolist() == [3.0]
    uswitch = df[(df.CountryName == "United Kingdom") & (df.date == "2020-09-14")]
    assert uswitch["H6_Facial Coverings"].tolist() == [2.0]
    fswitch = df[(df.CountryName == "France") & (df.date == "2020-08-03")]
    assert fswitch["H6_Facial Coverings"].tolist() == [3.0]


def check_thresholding(df):
    bswitch = df[(df.CountryName == "Belgium") & (df.date == "2020-07-24")]
    assert bswitch["H6_Facial Coverings"].tolist() == [1]
    bswitch = df[(df.CountryName == "Belgium") & (df.date == "2020-07-25")]
    assert bswitch["H6_Facial Coverings"].tolist() == [1]
    bswitch = df[(df.CountryName == "Belgium") & (df.date == "2020-07-29")]
    assert bswitch["H6_Facial Coverings"].tolist() == [1]
    assert bswitch["H6_Facial Coverings_3plus"].tolist() == [1]
    aswitch = df[(df.CountryName == "Afghanistan") & (df.date == "2020-06-07")]
    assert aswitch["H6_Facial Coverings"].tolist() == [1]
    assert aswitch["H6_Facial Coverings_3plus"].tolist() == [1]
    cswitch = df[(df.CountryName == "Connecticut") & (df.date == "2020-07-24")]
    assert aswitch["H6_Facial Coverings"].tolist() == [1]
    fswitch = df[(df.CountryName == "France") & (df.date == "2020-08-03")]
    assert fswitch["H6_Facial Coverings"].tolist() == [1]
    assert fswitch["H6_Facial Coverings_3plus"].tolist() == [1]


if __name__ == "__main__":
    npi_cols, npi_cols_minus_mob = build_npi_list(MOBI)
    print(npi_cols)

    print(MOBI)

    wearing = load_and_clean_wearing()
    oxcgrt = load_oxcgrt(use_us=US)

    if MOBI in ["only", "include"]:
        join = load_and_clean_mob(wearing, use_us=US)
        join = add_cgrt_responses(join, oxcgrt)

    if MOBI in ["exclude"]:
        oxcgrt = clean_oxcgrt(oxcgrt, npi_cols_minus_mob)
        join = join_ox_umd(oxcgrt, wearing, npi_cols_minus_mob)

    if MOBI in ["include"]:
        oxcgrt = clean_oxcgrt(oxcgrt, npi_cols_minus_mob)
        joinRs = join.country.unique()
        join = join_ox_umd(oxcgrt, join, npi_cols)

    print(join.country.unique().shape[0])

    # date, region, npi cols, mandates, mobility, wearing, cases, deaths
    path = f"../data/modelling_set/master_data_mob_{MOBI}_us_{US}_m_w.csv"
    print("Output to", path)
    join.to_csv(path, index=False)

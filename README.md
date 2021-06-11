# Code and data for Leech et al (2021), "Mass mask-wearing is associated with reduced COVID-19 transmission"

by [Charlie Rogers-Smith](
https://github.com/CRogers-Smith) and [Gavin Leech](https://gleech.org), based on the work of [Brauner et al](https://github.com/epidemics/COVIDNPIs/).


## Linux

### Installation

1. `git clone https://github.com/g-leech/masks_v_mandates.git`
2. Get [Poetry](https://python-poetry.org/docs/#installation).

```
cd masks_v_mandates
poetry install
cd data/raw
wget https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv
wget https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip
unzip Region_Mobility_Report_CSVs.zip
rm Region_Mobility_Report_CSVs.zip
wget https://github.com/OxCGRT/covid-policy-tracker/blob/master/data/OxCGRT_latest.csv
python ../../scripts/pull_from_umd_api.py
cd ../..
```


### Run

1. `poetry shell`
2. `python scripts/produce_master_model_data.py`
3. `mkdir pickles`
5. `python3 scripts/main_model_runner.py --model="cases" --masks="wearing" --mob="include"`
6. `python3 scripts/main_model_runner.py --model="cases" --masks="mandate" --mob="include"`

After about 6 hours, this will produce two posterior traces in `pickles` which can be loaded and visualised with [model_check_and_viz](https://github.com/g-leech/masks-npis/blob/sensitivity_analysis/notebooks/model_check_and_viz.ipynb), changing the filenames in `Load pickles`.


## Datasets

The data used is from the following sources.

* National nonpharmaceutical interventions: [OxCGRT](https://github.com/OxCGRT/covid-policy-tracker/).
* National confirmed case counts: [Johns Hopkins CSSE via OxCGRT](https://github.com/CSSEGISandData/COVID-19).
* World mask wearing survey: [UMD / Facebook](https://gisumd.github.io/COVID-19-API-Documentation/)
* [US mask wearing survey](https://github.com/g-leech/masks_v_mandates/tree/main/data/raw/rader): [Rader (2021)](https://www.sciencedirect.com/science/article/pii/S2589750020302934)
* National mobility index: [Google Mobility Reports](https://www.google.com/covid19/mobility/)


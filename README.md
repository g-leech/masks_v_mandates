# Code and data for Leech et al (2021), "Mass mask-wearing is associated with reduced COVID-19 transmission"

### Installation

1. `git clone https://github.com/g-leech/masks_v_mandates.git`
2. Get [Poetry](https://python-poetry.org/docs/#installation).
3. `cd masks_v_mandates`
4. `poetry install`
5. `cd data/raw`
5. `wget https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv`
6. `cd ../..`

### Run

1. `poetry shell`
2. `cd scripts`
3. `python produce_master_model_data.py`
2. `cd ../notebooks`
2. `cd notebooks`
3. `python3 wearing.py --model="cases" --masks="wearing" --mob="include"`
4. `python3 wearing.py --model="cases" --masks="mandate" --mob="include"`

After about 6 hours, this will produce two posterior traces in `notebooks/pickles` which can be loaded and visualised with [model_check_and_viz](https://github.com/g-leech/masks-npis/blob/sensitivity_analysis/notebooks/model_check_and_viz.ipynb), changing the filenames in `Load pickles`.


### Datasets

The data used is from the following sources.

* [National nonpharmaceutical interventions](https://github.com/g-leech/masks_v_mandates/blob/main/data/raw/OxCGRT_latest.csv): [OxCGRT](https://github.com/OxCGRT/covid-policy-tracker/).
* National confirmed case counts: [Johns Hopkins CSSE via OxCGRT](https://github.com/CSSEGISandData/COVID-19).
* [World mask wearing survey](https://github.com/g-leech/masks_v_mandates/tree/main/data/raw/umd): [UMD / Facebook]
* [US mask wearing survey](https://github.com/g-leech/masks_v_mandates/tree/main/data/raw/rader): [Rader (2021)](https://www.sciencedirect.com/science/article/pii/S2589750020302934)
* National mobility index: [Google Mobility Reports](https://www.google.com/covid19/mobility/)


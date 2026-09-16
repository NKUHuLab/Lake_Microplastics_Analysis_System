# Lake Microplastics Analysis System
This repository contains the source code, data processing pipelines, and optimization engines used in the study: "Intensive Aquaculture Contributes to Microplastic Pollution and Threatens Species in Lakes".
________________________________________
# 1. System Requirements
Operating Systems
•	Windows: Tested on Windows 10 and 11 (64-bit).
•	Linux: Tested on Ubuntu 20.04 and 22.04.
•	macOS: Tested on macOS Monterey (12.0) and later.
# Software Dependencies
The system requires both Python and R environments. The versions below describe the original workflow; the self-contained revision pathway calculation requires only NumPy and pandas and was checked with NumPy 2.3.5 and pandas 3.0.1.
Python (Version 3.9.12 tested)
•	scikit-learn (1.0.2)
•	pandas (1.4.2)
•	numpy (1.21.5)
•	geopandas (0.10.2)
•	shap (0.40.0)
•	geatpy (2.7.0) — Used for Genetic Algorithm optimization
•	tensorflow (2.8.0)
•	xgboost (1.5.1) / lightgbm (3.3.2)
R (Version 4.2.1 tested)
•	tidyverse (1.3.1)
•	glmmTMB (1.1.3) — For Zero-Inflated Mixed Models
•	segmented (1.6.0) — For breakpoint analysis
•	plspm (0.5.0) — For Path Modeling
•	patchwork (1.1.1)
# Hardware Requirements
•	Minimum: 8GB RAM, 4-core CPU.
•	Recommended: 16GB+ RAM (for handling large global lake shapefiles).
•	No non-standard hardware (e.g., GPU) is required, though geatpy can benefit from multi-core CPUs.
________________________________________
# 2. Installation Guide
Python Setup
1.	Clone the repository: git clone https://github.com/NKUHuLab/Lake_Microplastics_Analysis_System
2.	Create a virtual environment and install dependencies:
Bash
cd Lake_Microplastics_Analysis_System
pip install -r requirements.txt
R Setup
Open R/RStudio and run:
R
install.packages(c("tidyverse", "glmmTMB", "segmented", "plspm", "patchwork"))
Typical Install Time
•	~10–15 minutes on a standard desktop computer with a stable internet connection.
________________________________________
# 3. Demo (Quick Start)

The existing feature demonstration dataset is [Demo_dataset.csv](Demo_dataset.csv). It illustrates the input structure; full model training and global prediction require the original complete inputs.

For a runnable revision example with all its numerical inputs included, execute from the repository root (one directory above this README):

```sh
pip install -r analysis/modules/pathway_attribution/requirements.txt
python analysis/modules/pathway_attribution/code/run_all.py
```

This reproduces 10,000 literature-informed Monte Carlo draws and checks agreement with the frozen final pathway results. Outputs are written to `analysis/modules/pathway_attribution/outputs/`. See the [pathway calculation guide](../analysis/modules/pathway_attribution/README.md) for inputs, methods and interpretation.

________________________________________
# 4. Instructions for Use
Configuration

The default analysis root is the repository directory. Set `LAKE_MP_ROOT` to use another workspace; input and output paths are defined in [Utils/config.py](Utils/config.py). The original Python scripts import `config` from `Utils`. For example, in PowerShell from the repository root (one directory above this README), after supplying the configured inputs:

```powershell
$env:PYTHONPATH = (Resolve-Path './Lake_Microplastics_Analysis_System/Utils').Path
python ./Lake_Microplastics_Analysis_System/01_Main_Analysis_Pipeline/01_Train_RandomForest.py
```

The original workflow remains in `Lake_Microplastics_Analysis_System/`:

| Step | Script or directory | Function |
| --- | --- | --- |
| 1 | `01_Main_Analysis_Pipeline/01_Train_RandomForest.py` and `05_Global_Prediction.py` | Model fitting and global MP prediction |
| 2 | `01_Main_Analysis_Pipeline/03_SHAP_Global_Analysis.py` and `08_SHAP_Clustering_Analysis.py` | Feature attribution and clustering |
| 3 | `03_Bio_Validation_MPB/02_Analysis_Step2.R` | Organismal ingestion analysis |
| 4 | `02_Biodiversity_Risk_IUCN/` | Species–lake exposure analyses |
| 5 | `01_Main_Analysis_Pipeline/14_GA_Optimization_Engine.ipynb` | Mitigation scenario optimization |

Selected revision analyses are in [analysis/modules](../analysis/modules/README.md), including grouped validation, hydrological moderation, biological sensitivity and trait controls, OSM proxies, training-subset sensitivity and literature-informed pathway calculations. Their original input requirements remain in place; selected publication products do not replace the full analysis samples.

The standalone two-box equations and residence-time scenario can be run with:

```sh
python analysis/modules/hydrology_gam_and_box_model/box_model_core.py
```

# How to use on your own data

1. Use the predictor names in `Utils/config.py` and the existing demonstration dataset to prepare your inputs.
2. Supply the configured `data/train/train_data.csv`, including the predictors and `ln` response, before training. Global prediction also requires the configured feature and spatial files. The original training script does not accept a `--input` argument.

# 5. Selected Data

See the [dataset index](../data/Dataset_index.csv) and [data guide](../data/README.md) for selected lake-level OSM proxies, box-model parameter evidence, costs, implementation and related timing, checked pathway sources, and lake predictions with uncertainty. The [SHP archive](../data/geospatial/lake_prediction_exposure_high.zip) contains selected lake polygons with uncertainty and derived aggregate exposure attributes, together with its field dictionary and selection criteria. All numerical inputs used in the revised pathway calculation are included under `analysis/modules/pathway_attribution/`.

The water and biota supplementary datasets and the existing feature demo are not duplicated in this update. Model reliability classes and source-verification scopes are described separately in the data guide. Data curation is ongoing.

________________________________________
# License
The software is licensed under the [MIT License](License). Data reuse follows the [dataset-specific conditions](../data/README.md), including OSM, HydroLAKES and IUCN attribution and applicable restrictions.
# Contact
For any issues or questions, please contact Xiangang Hu (huxiangang@nankai.edu.cn).


See [methods, explicit assumptions and code corrections](../docs/methods_and_validation.md) for the current validation boundary and analyses requiring a fresh full-data run.

Global Lake Microplastics Analysis & Mitigation Framework
This repository contains the source code and analytical pipeline for the study: "Intensive Aquaculture Contributes to Microplastic Pollution and Threatens Species in Lakes".


System Requirements
Python Environment
Python Version: 3.8+

Key Libraries:

scikit-learn (Machine Learning)

pandas, numpy (Data Manipulation)

shap (Model Interpretability)

geatpy (Genetic Algorithms for Optimization)

matplotlib, seaborn (Visualization)

joblib (Model Persistence)

R Environment
R Version: 4.0+

Key Libraries:

tidyverse (Data Wrangling)

nlme (Robust Linear Mixed-Effects Models)

ggeffects (Marginal Effects)

patchwork (Plot Composition)


Setup & Configuration
Data Placement: Ensure source datasets (training data, shapefiles) are placed in the directory structure defined in Utils/config.py.

Path Configuration: Open Utils/config.py and set the PROJECT_ROOT variable to your local directory:

Python
# Utils/config.py
PROJECT_ROOT = r"/path/to/your/project"
This file controls all input/output paths and feature groupings.

🔬 Analysis Modules
Module 1: Machine Learning & Global Prediction
Training: 01_Train_RandomForest.py trains the core Random Forest Regressor. It performs 10-fold Cross-Validation (ShuffleSplit) and a Permutation Test (n=100) to validate statistical significance.

Optimization: 14_GA_Optimization_Engine.ipynb uses the Geatpy library to perform evolutionary optimization. It minimizes MP concentrations by adjusting watershed features (e.g., wastewater treatment, fishery intensity) within defined constraints.

Module 2: Ecological Risk Assessment
Integrates IUCN Red List spatial data with predicted MP concentrations.

Calculates the Accumulated Category Risk (ACR) and Calibrated Normalized Exposure Index (CNEI) for threatened species.

Module 3: Biological Validation (In Vivo)
Script: 03_Analysis_Step2.R

Methodology: Uses a Linear Mixed-Effects Model (nlme::lme) to validate the correlation between predicted environmental MP concentrations and measured organismal burdens (items/g and items/ind).

Controls: Models include random effects for Study_ID and fixed effects for Trophic_level, Origin, and Habitat_dependency.

Module 4: Visualization
Generates high-resolution maps and statistical plots found in the manuscript.

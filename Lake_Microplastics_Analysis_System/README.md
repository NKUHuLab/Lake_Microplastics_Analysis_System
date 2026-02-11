# Lake Microplastics Analysis System (v1.0)
This repository contains the source code, data processing pipelines, and optimization engines used in the study: "Intensive Aquaculture Contributes to Microplastic Pollution and Threatens Species in Lakes".
________________________________________
# 1. System Requirements
Operating Systems
•	Windows: Tested on Windows 10 and 11 (64-bit).
•	Linux: Tested on Ubuntu 20.04 and 22.04.
•	macOS: Tested on macOS Monterey (12.0) and later.
# Software Dependencies
The system requires both Python and R environments.
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
We have provided a subset of the global dataset (240 lakes) in Data/Demo/ to verify the installation.
Instructions
1.	Run the prediction demo:
Bash
python 01_Train_RandomForest.py 
2.	Run the optimization demo:
Bash
python 14_GA_Optimization_Engine.py

Expected Run Time
•	~2–3 minutes on a normal desktop.
________________________________________
# 4. Instructions for Use
Configuration
Before running the full analysis, update the root path in Utils/config.py

To reproduce the quantitative results and figures in the manuscript, execute scripts in the following order:
Step	Script	Function	Target Result
1	01_Train_RandomForest.py	Global MP abundance prediction	Fig. 1A, Fig. S2-S4
2	02_SHAP_Clustering.ipynb	Feature attribution & Mode ID	Fig. 1B-C, Fig. S8
3	03_Analysis_Step2.R	In vivo validation (ZINB-GLMM)	Fig. 4, Table S2
4	10_Risk_Assessment.py	IUCN species exposure calculation	Fig. 4A-D
5	14_GA_Optimization.py	Mitigation scenario modeling	Fig. 5, Fig. 6
# How to use on your own data
1.	Format your lake features (hydrology, land use, fishery intensity) according to the template in Data/Template_Features.csv.
2.	Place the file in Input/ and run 01_Train_RandomForest.py --input YourData.csv.
________________________________________
# License
This project is licensed under the MIT License.
# Contact
For any issues or questions, please contact Xiangang Hu (huxiangang@nankai.edu.cn).


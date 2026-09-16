# Selected revision code

Only core revision analyses are included. Existing original modules remain in their original directory; the additions do not silently overwrite their scientific results.

| Directory | Purpose | Input boundary |
| --- | --- | --- |
| `r1c4` | Six-method within-component allocation and literature-informed Monte Carlo scenarios | All numerical runtime inputs included; `code/run_all.py` checks frozen final draws. |
| `hydrology_gam_and_box_model` | Hydrological moderation, GAM display, grouped interactions and marginal sensitivity | Statistical scripts need the original training table through `LAKE_MP_TRAIN_DATA`. `box_model_core.py` is a standalone extraction of the actual two-box equations and 1,200-draw residence-time scenario; curated box evidence is not fitted automatically. |
| `grouped_cross_validation` | Dependence-aware validation | Requires original observations, predictor table and study/site/lake identifiers specified in the script. |
| `biota_zinb_and_traits` | Biological method/trophic sensitivity and trait controls | Requires the original biological analysis tables and joins; these are not supplied by the selected spatial exposure table. |
| `osm_fishery_proxy` | Independent fishery proxy analysis | Requires the frozen proxy-analysis inputs and `LAKE_MP_WORLD_SHP` / `LAKE_MP_LAKES_SHP`. The selected 6,066-lake OSM product is a separate publication subset, not a replacement for the original analysis sample. |
| `training_sensitivity` | Geographic, recent-period and methodological subsets | Requires the original full training table and associated metadata. |

Python and R scripts expose their paths near the beginning. Revision scripts use `LAKE_MP_REPO_ROOT`, `LAKE_MP_DATA_ROOT`, `LAKE_MP_OUTPUT_DIR` or the analysis-specific overrides present in each script. R scripts should be run from the repository root unless paths are explicitly overridden. Imports in each script define additional dependencies; the lightweight R1C4 requirements do not install the entire original workflow.

The full raw species-range intersections and the area-overlap sensitivity rerun are not supplied in this selected release. The selected spatial product excludes lakes with known invalid overlap flags, but does not recalculate or validate the full Fig. S17 sensitivity analysis.

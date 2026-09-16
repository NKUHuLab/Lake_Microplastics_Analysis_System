# Literature-informed pathway scenario analysis

Run `python analysis/revision/r1c4/code/run_all.py` from the repository root after installing this directory's `requirements.txt` (NumPy and pandas).

## Inputs and calculation

- `data/numeric_inputs.csv`: all 185 admitted numerical quantities actually used by the frozen final calculation. Values, source-context locators, units, conversions, screening decisions and verification fields are retained. Admission to the scenario calculation does not mean every value was independently rechecked against a primary PDF.
- `data/verified_source_context.csv`: retained checked source context; `data/comparison_group_registry.csv`: comparable-parameter grouping rules.
- `inputs/original_feature_method_percentages.csv`: original six-method feature scores. Their equal-weight average is normalized within each component to allocate fixed baselines: non-fishery watershed 15.8, fishery 13.0, atmosphere 9.8, hydrology 35.2. The six methods do not re-estimate these component totals.
- Other input CSVs preserve the previous feature allocation, protected source context, the legacy residual sequence and the earlier trial comparison required by the original checks. `reference_final_group_draws.csv` is the frozen final result used for the reproduction check, distinct from the earlier trial reference.
- The screening-corpus count in `model_config.json` is provenance metadata. The complete extraction corpus is not a runtime input and is not redistributed here.

The sampler selects document-title groups, parameter strata and records with equal probabilities at their respective levels. Positive ranges use log-uniform draws and point values remain fixed. Multipliers are referenced within the defined groups; eligible comparable parameter families use shared reference values. Applicable multipliers are multiplied for each feature, applied to baseline allocations and normalized across components. The direct fishery multiplier remains one because no direct-release stream was admitted; its normalized interval arises from variation in other terms. The residual score retains the specified Uniform(0,15) sequence.

Results are conditional relative attribution scenarios, not measurements of physical input fractions. Source contexts and units must be interpreted at the recorded measurement scope; heterogeneous quantities are not pooled as absolute fluxes. No filtering by the direction of the resulting fishery contribution is applied to the 185 runtime inputs.

## Outputs and checks

The generated `outputs/` directory is not versioned. It contains pathway draws and summaries, feature and source participation records, perturbation/window analyses and `run_checks.json`. `reproduction_check.json` reports agreement with the frozen final 10,000 draws (absolute tolerance 1e-12).

The packaged calculation was checked with NumPy 2.3.5 and pandas 3.0.1. All 60,000 group-contribution values agreed with the saved final draws within 2.14e-14 percentage points. This check covers numerical reproduction of the frozen configuration, not independent source verification.

The final fishery median is 15.4655%, with a central 95% scenario interval of 6.5359–23.2258%; the frequency of ranking first among watershed subfeatures is 62.49%. These statements describe this frozen configuration. The general curated evidence files under `data/evidence/` are contextual datasets and do not replace these numerical inputs.

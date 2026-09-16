# Methods, explicit assumptions and code corrections

## Explicit assumptions

| Analysis | Assumption or fixed setting | Interpretation |
| --- | --- | --- |
| Pathway scenarios | Component baselines: non-fishery watershed 15.8, fishery 13.0, atmospheric 9.8 and hydrological 35.2 | Prespecified relative scores. The six-method feature average allocates these scores within components; it does not estimate the component totals. |
| Pathway scenarios | Other score Uniform(0,15); direct fishery multiplier fixed at one | Scenario assumptions. The absence of an admitted fishery-release stream is not evidence that fishery-release uncertainty is zero. |
| Pathway scenarios | Equal-probability selection at document, stratum and record levels; log-uniform positive ranges | Sampling design and scenario envelopes, not an empirical posterior or physical flux confidence intervals. |
| Two-box calculation | H=10 m, active sediment depth=0.10 m, unit source, and specified kinetic ranges | Representative mechanistic scenario, not lake-specific fitted geometry or rate estimates. |
| Model reliability | Equal-weight combination of normalized novelty and tree dispersion, then quartile classes | Heuristic relative reliability classification, not an externally calibrated probability. |
| Random forest | Prespecified tree count, depth, feature subsampling and random seeds | Estimator settings and reproducibility controls. No evidence of global optimality is supplied by fixing these values. |

These settings remain explicit. Removing or changing them defines a different analysis and requires recomputing and reassessing the corresponding manuscript results. Frozen output comparisons are regression checks; they do not force the model to fit observations or establish source validity.

## Corrections in this update

- Removed the per-lake t test that treated correlated trees as independent replicates, and removed the corresponding p-value fields from the selected CSV and SHP. Descriptive tree dispersion is retained under its actual interpretation.
- Corrected out-of-bag sample selection, removed hardcoded partial-correlation output, and separated Pearson correlation from the test-set coefficient of determination in the OSM/validation code.
- Moved missing-predictor imputation inside cross-validation folds. Final prediction uses training-set imputation statistics. The repeated ShuffleSplit evaluation is described as random holdout validation, not disjoint ten-fold cross-validation.
- Aligned the separate random-forest validation configuration with the main training script. Corrected missing helper imports and seeded geographic subset selection.
- Unweighted sums of concentrations and changes in geometric mean concentration are described as diagnostics. They are not volume-weighted lake reservoirs; ranges from resampling forest trees are not sampling confidence intervals for total stock. The fixed 5.52e14 reservoir multiplier has been removed from that proxy calculation.
- Removed the legacy model-comparison entry point with an undefined model registry and the legacy biological Python plot with an observed-mean-calibrated intercept and hardcoded slope annotation. No replacement historical estimators or fitted intercepts have been invented. The R biological models remain available.
- Corrected the restored unit for PATH-10661. Its 5–26 g/m2 numerical range was already consistent with 50–260 kg/ha; no numerical input row was removed on the basis of this transcription issue.

## Validation boundary

Renaming folders and the pathway unit-metadata correction do not alter the frozen pathway draws. Code-level regression tests check fold-specific imputation, valid out-of-bag aggregation, statistical-field removal, geographic sampling reproducibility and the pathway calculation. The selected prediction and exposure values are retained; their removed p-value fields must not be reused.

The full historical training, OSM and biological analyses have not been rerun as part of this code correction. Previously reported validation metrics and figures affected by the corrected code must be recomputed before they can be described as reproduced by this release. A code correction alone does not update manuscript results, and this audit is not a finding about research intent or an exhaustive primary-literature verification.

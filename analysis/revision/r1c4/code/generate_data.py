"""Rebuild table admission, v7-centered resampling and every plotted statistic."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import table_evidence
import scenario_sampler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs"
CFG = json.loads(Path(__file__).with_name("model_config.json").read_text(encoding="utf-8"))
TOP = ["Total_Watershed", "Hydrological", "Atmospheric", "F_other"]


def save(frame, name):
    frame.to_csv(DATA / name, index=False, encoding="utf-8-sig")


def summarize(frame):
    rows = []
    for term in frame:
        x = frame[term].to_numpy()
        lo, med, hi = np.quantile(x, [.025, .5, .975])
        block = np.array([np.quantile(v, [.025, .5, .975]) for v in np.array_split(x, 20)])
        se = block.std(axis=0, ddof=1)/np.sqrt(20)
        rows.append(dict(term=term, mean=x.mean(), median=med, lower=lo, upper=hi,
                         width=hi-lo, mcse_lower=se[0], mcse_median=se[1], mcse_upper=se[2]))
    return pd.DataFrame(rows)


def normalize(raw):
    p = raw.div(raw.sum(axis=1), axis=0)*100
    p["Total_Watershed"] = p.WS_nonfish+p.Fishery
    return p


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    admitted = pd.read_csv(ROOT / "data/numeric_inputs.csv").fillna("")
    table_evidence.OUT.mkdir(parents=True, exist_ok=True)
    streams_available = admitted.endpoint.map(table_evidence.ENDPOINT_STREAM)
    inputs = admitted[streams_available.isin(CFG["stream_features"])].copy()
    excluded = admitted[~streams_available.isin(CFG["stream_features"])].copy()
    excluded["model_exclusion_reason"] = "No stream mapping in the retained v7 structure"
    save(inputs, "numeric_inputs.csv")
    save(excluded, "eligible_but_outside_v7_structure.csv")
    n, seed = CFG["draws"], CFG["seed"]
    streams, within_streams, participation = scenario_sampler.factor_draws(inputs, n, np.random.default_rng(seed))
    old = pd.read_csv(ROOT / "inputs/original_feature_method_percentages.csv").set_index("feature")
    weights = old.mean(axis=1)
    components, centers = CFG["components"], CFG["centers"]
    base = pd.Series({f: centers[k]*weights[f]/weights[fs].sum()
                     for k, fs in components.items() for f in fs})
    factors = pd.DataFrame(1., index=range(n), columns=base.index)
    for stream, columns in CFG["stream_features"].items():
        if stream in streams:
            factors.loc[:, columns] *= np.repeat(streams[stream][:, None], len(columns), axis=1)
    if "Fishery" not in streams:
        np.testing.assert_allclose(factors.fish_gdp_sqkm, 1.)
    for k, fs in components.items():
        np.testing.assert_allclose(base[fs].sum(), centers[k])
    feature_raw = factors.mul(base, axis=1)
    # Advance the old RNG solely to retain exactly the previously tested Other sequence.
    legacy_rng = np.random.RandomState(seed)
    for key, center in centers.items():
        cv = CFG["legacy_CVs_for_matching_other_rng_only"][key]
        legacy_rng.lognormal(np.log(center), np.sqrt(np.log1p(cv**2)), n)
    other = legacy_rng.uniform(*CFG["other_range"], n).astype(np.float32)
    denominator = feature_raw.sum(axis=1)+other
    features = feature_raw.div(denominator, axis=0)*100
    raw = pd.DataFrame({k: feature_raw[fs].sum(axis=1) for k, fs in components.items()})
    raw["F_other"] = other
    groups = pd.DataFrame({k: features[fs].sum(axis=1) for k, fs in components.items()})
    groups["F_other"] = 100*other/denominator
    groups["Total_Watershed"] = groups.WS_nonfish+groups.Fishery
    reference = pd.read_csv(ROOT / "inputs/reference_trial_group_draws.csv")
    previous_raw = pd.read_csv(ROOT / "inputs/previous_raw_component_scores.csv")
    np.testing.assert_allclose(raw.F_other, previous_raw.F_other, atol=1e-7)
    np.testing.assert_allclose(groups[TOP].sum(axis=1), 100)
    save(groups, "group_draws.csv")
    save(features, "feature_draws.csv")
    save(raw, "raw_component_scores.csv")
    # A paired ablation changes only the shared references, with identical selected records.
    within_factors = pd.DataFrame(1., index=range(n), columns=base.index)
    for stream, columns in CFG["stream_features"].items():
        if stream in within_streams:
            within_factors.loc[:, columns] *= np.repeat(within_streams[stream][:, None], len(columns), axis=1)
    within_features_raw = within_factors.mul(base, axis=1)
    within_raw = pd.DataFrame({k: within_features_raw[fs].sum(axis=1) for k, fs in components.items()})
    within_raw["F_other"] = other
    save(summarize(normalize(within_raw)), "same_draws_without_between_document_summary.csv")
    save(factors, "feature_factor_draws.csv")
    save(base.rename("starting_score").rename_axis("feature").reset_index(), "feature_starting_scores.csv")
    save(summarize(groups), "group_summary.csv")
    save(summarize(features), "feature_summary.csv")
    comparison = summarize(groups).merge(summarize(reference), on="term", suffixes=("_cleaned", "_previous"))
    comparison["median_change_pp"] = comparison.median_cleaned-comparison.median_previous
    comparison["width_change_pp"] = comparison.width_cleaned-comparison.width_previous
    save(comparison, "comparison_previous.csv")
    old_features = pd.read_csv(ROOT / "inputs/previous_feature_summary.csv")
    save(summarize(features).merge(old_features, on="term", suffixes=("_cleaned", "_previous")), "feature_comparison_previous.csv")
    ws_features = components["WS_nonfish"]+components["Fishery"]
    counts = features[ws_features].idxmax(axis=1).value_counts()
    rank = pd.DataFrame({"feature": ws_features, "first_count": [int(counts.get(f, 0)) for f in ws_features]})
    rank["first_frequency"] = rank.first_count/n
    save(rank, "watershed_ranks.csv")
    window = CFG["stream_window"]
    rolling = pd.DataFrame({"iteration": np.arange(window, n+1)})
    for term in TOP:
        r = groups[term].rolling(window)
        for label, q in [("lower", .025), ("median", .5), ("upper", .975)]:
            rolling[term+"_"+label] = r.quantile(q).iloc[window-1:].to_numpy()
    save(rolling, "MC_A_rolling_summary.csv")
    widths, prefixes = [], []
    for term in ["Total_Watershed", "Hydrological"]:
        values = groups[term].to_numpy()
        for window in CFG["windows"]:
            for end in range(window, n+1, 50):
                lo, hi = np.quantile(values[end-window:end], [.025, .975])
                widths.append(dict(term=term, window=window, iteration=end, width=hi-lo))
        for end in range(100, n+1, 50):
            lo, med, hi = np.quantile(values[:end], [.025, .5, .975])
            prefixes.append(dict(term=term, iteration=end, lower=lo, median=med, upper=hi, width=hi-lo))
    save(pd.DataFrame(widths), "MC_B_rolling_widths.csv")
    save(pd.DataFrame(prefixes), "MC_B_cumulative_quantiles.csv")
    sensitivity, effects = [], []
    outputs = TOP+["Fishery", "WS_nonfish"]
    q0 = groups[outputs].quantile([.025, .5, .975])
    display_output = {"WS_nonfish": "Total_Watershed", "Fishery": "Fishery", "Hydrological": "Hydrological", "Atmospheric": "Atmospheric", "F_other": "F_other"}
    for term in raw:
        result = dict(component=term, displayed_output=display_output[term])
        for sign, label in [(-1, "minus"), (1, "plus")]:
            relative_change = sign*CFG["sensitivity_relative_step"]
            perturbed = raw.copy()
            perturbed[term] *= 1+relative_change
            alt = normalize(perturbed)[outputs].quantile([.025, .5, .975])
            dm = alt.loc[.5]-q0.loc[.5]
            for output in outputs:
                effects.append(dict(component=term, relative_score_change=relative_change, output=output,
                                    baseline_median=q0.loc[.5, output], perturbed_median=alt.loc[.5, output], median_shift_pp=dm[output]))
            result[label+"_median_shift_pp"] = float(dm[display_output[term]])
        sensitivity.append(result)
    save(pd.DataFrame(sensitivity), "MC_D_sensitivity_summary.csv")
    save(pd.DataFrame(effects), "MC_D_sensitivity_all_outputs.csv")
    ids = np.random.default_rng(seed).choice(n, CFG["ternary_display_draws"], replace=False)
    three = groups[["Total_Watershed", "Hydrological", "Atmospheric"]]
    fractions = three.div(three.sum(axis=1), axis=0)
    ternary = pd.DataFrame({"draw_index": ids+1, "x": fractions.Hydrological.iloc[ids].to_numpy()+.5*fractions.Atmospheric.iloc[ids].to_numpy(),
                           "y": np.sqrt(3)/2*fractions.Atmospheric.iloc[ids].to_numpy(), "F_other": groups.F_other.iloc[ids].to_numpy()})
    save(ternary, "MC_E_displayed_draws.csv")
    checks = {"all_rows": CFG["screening_corpus_record_count"],
              "previously_used_quantities": 179,
              "admitted_quantities": len(admitted), "used_quantities": len(inputs),
              "point_records": int(inputs.lo.eq(inputs.hi).sum()), "range_records": int(inputs.lo.ne(inputs.hi).sum()),
              "used_document_title_groups": int(inputs.study_id.nunique()), "all_used_quantities_sampled": bool(participation.draws_selected.gt(0).all()),
              "same_baseline_and_other_sequence": True, "closure_passed": True,
              "fishery_first_frequency": float(rank.set_index("feature").loc["fish_gdp_sqkm", "first_frequency"]),
              "fishery_direct_multiplier_fixed": bool(factors.fish_gdp_sqkm.nunique() == 1),
              "between_document_eligible_quantities": int(inputs.between_document_enabled.sum()),
              "sensitivity_relative_step": CFG["sensitivity_relative_step"],
              "pearson_r": float(groups.Total_Watershed.corr(groups.Hydrological)),
              "spearman_rho": float(groups.Total_Watershed.rank().corr(groups.Hydrological.rank())),
              "fishery_to_watershed_ratio_of_medians": float(groups.Fishery.median()/groups.Total_Watershed.median()),
              "median_within_watershed_fishery_fraction": float((groups.Fishery/groups.Total_Watershed).median()),
              "physical_input_estimate": False, "posterior": False,
              "interval_definition": "Central 95% scenario simulation interval, not a confidence interval for physical inputs",
              "no_magnitude_based_exclusion_or_winsorization": True,
              "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT / "inputs").glob("*.csv")) + sorted((ROOT / "inputs").glob("*.csv.gz"))}}
    (DATA / "run_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print(summarize(groups).to_string(index=False))


if __name__ == "__main__":
    main()

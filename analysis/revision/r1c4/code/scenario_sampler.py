"""Document-balanced scenario sampling with explicitly registered common references."""
from pathlib import Path
import numpy as np
import pandas as pd
from table_evidence import ENDPOINT_STREAM

OUT = Path(__file__).resolve().parents[1] / "outputs/admission"


def factor_draws(inputs, n, rng):
    OUT.mkdir(parents=True, exist_ok=True)
    frame = inputs.copy()
    frame["stream"] = frame.endpoint.map(ENDPOINT_STREAM)
    assert frame.stream.notna().all() and frame.lo.gt(0).all()
    factors, within, between, trace, details = {}, {}, {}, {}, []
    for stream, data in frame.groupby("stream", sort=True):
        studies = sorted(data.study_id.unique())
        chosen_studies = rng.choice(studies, n)
        sampled, internal, external = np.ones(n), np.ones(n), np.ones(n)
        source_ids = np.empty(n, dtype=object)
        for study in studies:
            positions = np.flatnonzero(chosen_studies == study)
            sd = data[data.study_id.eq(study)]
            strata = sorted(sd.stratum.unique())
            selected = rng.choice(strata, len(positions))
            for stratum in strata:
                pos = positions[selected == stratum]
                records = sd[sd.stratum.eq(stratum)]
                chosen = records.iloc[rng.integers(0, len(records), len(pos))]
                values = np.exp(rng.uniform(np.log(chosen.lo.to_numpy()), np.log(chosen.hi.to_numpy())))
                internal[pos] = values/chosen.within_study_reference.to_numpy()
                external[pos] = chosen.within_study_reference.to_numpy()/chosen.common_reference.to_numpy()
                sampled[pos] = internal[pos]*external[pos]
                source_ids[pos] = chosen.record_id.to_numpy()
        factors[stream], within[stream], between[stream], trace[stream] = sampled, internal, external, source_ids
        for r in data.itertuples():
            same = data[data.study_id.eq(r.study_id) & data.stratum.eq(r.stratum)]
            comparable = data[data.study_id.eq(r.study_id) & data.comparison_group.eq(r.comparison_group)] if r.between_document_enabled else same
            variable_within = comparable.center.nunique() > 1 or r.lo != r.hi
            variable_between = r.between_document_enabled and not np.isclose(r.within_study_reference, r.common_reference, rtol=1e-12, atol=0)
            details.append(dict(record_id=r.record_id, study_id=r.study_id, stream=stream,
                draws_selected=int(np.sum(source_ids == r.record_id)),
                probability_per_draw=1/len(studies)/data[data.study_id.eq(r.study_id)].stratum.nunique()/len(same),
                stratum=r.stratum, comparison_group=r.comparison_group,
                within_study_reference=r.within_study_reference, common_reference=r.common_reference,
                within_document_variation=variable_within, between_document_shift=variable_between,
                numerically_informative=bool(variable_within or variable_between)))
    participation = pd.DataFrame(details)
    assert participation.draws_selected.gt(0).all()
    for values, name in [(factors, "table_stream_factor_draws.csv"), (within, "within_document_factors.csv"),
                         (between, "between_document_factors.csv"), (trace, "table_selected_record_per_draw.csv")]:
        pd.DataFrame(values).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    participation.to_csv(OUT / "table_numeric_participation.csv", index=False, encoding="utf-8-sig")
    return factors, within, participation

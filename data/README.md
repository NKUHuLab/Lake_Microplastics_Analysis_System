# Selected data release

This snapshot contains only selected evidence, selected model products and the complete numerical inputs required by the revised pathway calculation. See [Dataset_index.csv](Dataset_index.csv) for paths and record counts. Data curation is ongoing.

## Evidence

- `evidence/OSM_aquaculture.csv`: 6,066 lake-level proximity summaries from selected, metadata-screened OSM production features within 25 km. Presence-only proxy data; not field-verified farm locations or a complete aquaculture census. The original frozen proxy-analysis sample is distinct.
- `evidence/Box_model_parameters.csv`: 122 records with documented previous local-PDF reading and a retained source locator. Measurements, model settings, theoretical values and study-specific conditions remain distinguished in the columns.
- `evidence/Cost_evidence.csv`: 106 curated source-linked cost records. Currency, unit, scope and recorded verification detail are retained. This is not a harmonized global price schedule; some verification is inherited from earlier source extraction.
- `evidence/Implementation_and_related_timing.csv`: 21 source-linked timing records, including implementation, service-life, policy and related response timings. Use the recorded category and context; they are not all construction durations or newly rechecked source values.
- `evidence/Pathway_source_checked_subset.csv`: 27 documented source-checked examples. The large automatic numeric-token extraction table is excluded because text matches alone do not establish quantity validity.
- [R1C4 numerical inputs](../analysis/revision/r1c4/data/numeric_inputs.csv): all 185 inputs actually used in the final scenario calculation, kept complete for reproducibility. Their verification fields distinguish admitted inputs from primary-source rechecks; do not describe the entire file as independently PDF-verified.

## Model products and spatial data

- `model_products/lake_prediction_uncertainty_high.csv`: 130,736 unique-coordinate records in the frozen model's `High` reliability class, with finite prediction and uncertainty values. `prediction` is log abundance; `mp_m3` is its exponential, in items/m3. `High` refers to model reliability, not independent field validation. Coordinate and uncertainty columns retain the source definitions.
- `geospatial/lake_prediction_exposure_high.zip`: 7,161 selected lake polygons with prediction, uncertainty, novelty, reliability and category-level ACR/CNEI/species-count attributes. All shapefile sidecars, a field dictionary and selection counts are inside the ZIP. Geometry and retained attributes are unchanged from the assembled source product.
- `model_products/lake_aggregate_exposure_high.csv`: the same 7,161 lakes, with baseline category-level exposure attributes. The project key links to `FID` in this project's SHP; it is not asserted to be a universal external identifier.

For the SHP, selection starts with the existing High class, excludes entire lakes with known invalid intersection/area-ratio flags, checks coordinate/prediction agreement with the selected prediction table, and excludes invalid geometry. This is a conservative publication subset, not a repair of the original overlap calculations. It cannot reproduce the full global estimates or Fig. S17 area-overlap sensitivity. Range-derived exposure indicates potential co-occurrence, not confirmed occupancy or measured organismal ingestion.

## Attribution and reuse

The software MIT license does not license third-party data. Source titles, URLs, units and verification scopes in the evidence tables remain part of the data and should accompany reuse. Source publications retain copyright in quoted source-context excerpts; consult the cited publications for reuse beyond the supplied factual records.

OSM-derived data are © OpenStreetMap contributors and subject to [ODbL](https://www.openstreetmap.org/copyright). HydroLAKES geometry is distributed under [CC BY 4.0](https://www.hydrosheds.org/products/hydrolakes); cite [Messager et al. (2016)](https://doi.org/10.1038/ncomms13603). Derived exposure attributes acknowledge the IUCN Red List and retain applicable non-commercial/source conditions; consult the [IUCN data conditions and FAQ](https://nrl.iucnredlist.org/about/faqs). Raw IUCN range polygons are not redistributed, and no IUCN endorsement is implied. The combined SHP must be reused subject to the conditions of both its geometry and its derived exposure attributes.

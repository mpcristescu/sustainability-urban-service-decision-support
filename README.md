# Urban service lifecycle validation and conditional decision support

Scientific materials for the Sustainability resubmission by Marian Pompiliu Cristescu, prepared on 20 October 2026.

Manuscript title: Lifecycle Validation and Conditional Decision Support for Sustainable Urban Services in Brussels and New York City.

Repository: https://github.com/mpcristescu/sustainability-urban-service-decision-support/

## Scope

The analysis audits recorded administrative closure, estimates censored lifecycles, tests a daily closure-opportunity replay and examines conditional prioritization contrasts. The replay fails lifecycle fidelity checks. The rule comparisons do not estimate deployable municipal policy effects, staffing capacity, physical repair quality or demographic equity.

## Reproduce the frozen results offline

The tested environment is Python 3.12.14 with the exact package versions in requirements.txt. The full run was verified on Windows. Use a separate virtual environment for a new installation.

    python -m pip install -r requirements.txt
    python scripts/reproduce_all.py

The second command runs 13 scientific verification tests, the full analysis, supplementary audits and numerical comparison with all 79 frozen CSV tables. It writes outputs/reproduced/frozen_table_verification.json. This workflow uses the included frozen cohort cells and provider dictionaries. It does not require a live data portal, Word, graphics software or manuscript-production scripts.

The default run expands 3,799,484 records, approximately 3.80 million. Several GB of free memory and writable disk space are recommended. Runtime depends on hardware and includes first-run Numba compilation. Do not treat runtime as a scientific result.

Optional environment settings are URBAN_SERVICE_OUTPUT_DIR, URBAN_SERVICE_DATA_DIR and NUMBA_CACHE_DIR. URBAN_SERVICE_USE_CACHE=1 reuses a local cache only after a trusted first run. No cache is supplied.

A verification run on 29 October 2026 passed all 13 scientific tests and reproduced all 79 archived CSV tables. The retained raw snapshots also reproduced every released cohort-cell value and multiplicity. A summary is recorded in metadata/final_reproduction_verification_20261029.json.

For individual analysis stages:

    python scripts/run_revision_analysis.py --stage descriptive
    python scripts/run_revision_analysis.py --stage validation
    python scripts/run_revision_analysis.py --stage policies
    python scripts/run_revision_analysis.py --stage robustness

The robustness stage requires frozen_policy_parameters.json from a policy run in the same output directory. The complete reproduction command is the recommended entry point.

## Included materials

scripts contains only scientific acquisition, lifecycle, replay, uncertainty and verification orchestration code. tests contains an independent FIFO reference and numerical checks. data contains identifier-free cohort cells and frozen provider dictionaries. metadata records source hashes, official monthly NYC counts, extraction queries, the analysis specification and the cohort-cell manifest. outputs/frozen contains the numerical tables and JSON results used for the manuscript.

Publication figures, Word files, Word-production scripts, raw request identifiers, addresses, coordinates, narratives, credentials, local software installations and binary caches are not included.

## Data provenance and scope

FixMyStreet Brussels: https://data.mobility.brussels/en/info/fims/

NYC current 311: https://data.cityofnewyork.us/d/erm2-nwe9

NYC 2010-2019 archive: https://data.cityofnewyork.us/d/76ig-c548

The Brussels source snapshot ends on 24 September 2026. The scoped NYC acquisition was completed on 20 October 2026 for DOT, DEP, DPR and DSNY. It includes all 2023-2025 creation months and observable inherited cases. All 144 month-agency acquisition counts agreed with official counts before and after extraction. This does not establish completeness of unreported need or a full history of changing administrative labels.

The two compressed cell files expand to 618,694 Brussels and 3,180,790 NYC records before primary endpoint exclusions. The NYC scope includes 698,826 records created before 2023. Supplemental inherited slices include impossible closure times, which cannot establish historical completion. Source timestamps are checked before daily coarsening, including within-day inversions. Nonclosed records with invalid closures remain unknown-closure records. Closed-status cases without a usable event time are excluded from primary analysis. Fields are unit, category, creation day, usable closure day, event, ambiguity, duration, status and multiplicity. Day zero is 1 January 1970. A closure day of -1 denotes no usable dated closure in the canonical fields. These cells have no request identifiers. They preserve the numerical fields required for the analyses, with a fixed ordering before random resampling.

The exact cell expansion was verified against source-derived canonical arrays. The released cells cannot independently repeat raw duplicate consolidation or confirm the real-world meaning of a closure event. Raw parsing code and historical source hashes are provided for those audits.

Provider terms continue to apply to provider data. The software license does not relicense source data or imply permission to identify individuals.

## New live extraction

prepare_data.py is an optional scientific acquisition script. Running it contacts the official NYC API. A later live extraction can differ because source records are updated. It must be reported as a new snapshot with its own retrieval dates and hashes, not silently substituted for the frozen manuscript data. No raw extraction is needed for the offline reproduction command.

## Statistical and operational boundaries

Development groups and risk are frozen before rule selection. Rules selected in Brussels 2023 and NYC 2024 are evaluated later. These are retrospective temporal evaluations, not a prospectively blinded study or an original preregistration.

The 2,000 descriptive block draws, 1,000 paired policy bootstrap draws and 30 paired shock realizations quantify different conditional uncertainties. They do not absorb structural errors in exchangeable closure opportunities, missing workflow events or inherited-stock interpretation.

The 90-day burden is a common analytical horizon, not a universal municipal SLA. Group CV is administrative dispersion, not demographic inequity. Every reported table should be interpreted with its stated population and source cutoff.

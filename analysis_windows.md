# Analysis windows

| Dataset | Development | Validation | Holdout or stress window |
|---|---|---|---|
| FixMyStreet Brussels | 2019-01-01 to 2022-12-31 | 2023-01-01 to 2023-12-31 | 2024-01-01 to 2024-12-31 |
| NYC 311 selected agencies | 2023-01-01 to 2023-12-31 | Not used | 2025-01-01 to 2025-12-31 |

Brussels group eligibility is frozen from the 2019 to 2022 development period. The simulator uses observed closure opportunities as empirical capacity. The 2023 Brussels validation window is evaluated before the 2024 policy and stress-test outputs are interpreted.

The NYC comparison calibrates weights on 2023 and evaluates them on the untouched 2025 temporal holdout. The script filters the full extract to agencies DOT, DEP, DPR, and DSNY.

## Brussels validation sensitivity

The exact original replay receives records created from 2022 onward but begins its daily loop on 1 January 2023. Requests created before that date are not enqueued, so the simulated initial backlog is zero. The revised audit reports this configuration together with a one-year initialization of 20,339 open cases and a full observable initialization of 51,645 open cases.

Observed comparison populations are reported separately. They include requests closed during 2023, requests created during 2023 and closed by the 24 September 2026 cutoff, and requests both created and closed during 2023. The within-year population truncates long lifecycles and is not used as the sole validation target.

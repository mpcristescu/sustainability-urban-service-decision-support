# Revision analysis specification

This specification is written after the original reviews. It is not a preregistration of the submitted analysis. All historical evaluation periods have already been available. The revised temporal separation prevents using evaluation outcomes for policy selection but does not create a prospectively blinded study.

## Sources and lifecycle endpoints

Version 4 extends the inherited-record acquisition after a date-validity audit on 20 October 2026. A raw closure date earlier than creation cannot establish historical completion. Retrieve those pre-2023 records in a disjoint supplemental slice from both NYC archives, verify agency-status counts before and after acquisition, and include nonclosed-status cases as unknown-closure inherited records. Closed-status cases with impossible closure times remain excluded from the primary analysis. Preserve and supersede the narrower inherited extraction. This correction precedes rerunning the affected policy analyses and changes neither the candidate objective definitions nor acceptance bands.

Retain the original FixMyStreet Brussels snapshot through 24 September 2026. Acquire a new non-identifying NYC extract from official current and historical 311 datasets, covering DOT, DEP, DPR, and DSNY arrivals in 2023, 2024, and 2025 and observable inherited cases still without a recorded closure at the beginning of 2023. Audit monthly agency counts before and after extraction. New NYC extraction date is reported independently from the Brussels cutoff.

Deduplicate identifiers deterministically. Audit missing creation dates, invalid negative intervals, closure dates, status disagreements, and missing category labels. A valid recorded closure is an administrative event, not proof of physical problem resolution. Closed-status records without a usable closure timestamp require an explicit ambiguity sensitivity analysis. Negative closure intervals are not silently clipped.

Version 5 validates the precise source closure timestamp against the precise creation timestamp before daily coarsening. Record same-day timestamp inversions separately and classify their closure time as unusable. A closed-status record with that inconsistency is ambiguous and excluded from primary estimation. A nonclosed record with it has unknown closure and is retained as censored. The strict validity rule applies to both sources before the final rerun. Dates retain their source civil-day interpretation for the replay. Preserve the superseded day-only audit for reconciliation.

## Descriptive analysis

Estimate Kaplan-Meier survival of remaining administratively open, closure probabilities at 7, 30, and 90 days with log-log Greenwood 95% intervals, and restricted mean open time to day 90. Use department-specific estimates for Brussels 2019-2025 arrivals with follow-up to the snapshot cutoff. Report 2023-cohort estimates as a temporal sensitivity analysis. Use pairwise log-rank tests with Holm correction and report absolute 90-day probability differences as well as sample sizes. Department contrasts are descriptive and may reflect service mix, not causal administrative performance.

## Replay diagnostics

For Brussels 2023, keep the population and supplied closure opportunities fixed while changing initialization from empty, to one-year inherited queue, to complete observable inherited queue. Separately vary the record-source restriction. Compare simulated and observed closure-year quantiles and use matched creation-cohort survival checks where support permits. Report number closed, initial and final accounting backlog, P50/P90/P95, absolute and relative errors, empirical distribution distance, and department-level errors. A conservation or throughput match obtained from supplied closure counts is not independent validation.

No numerical criterion existed before the submitted analysis. Present retrospective diagnostic acceptance-band sensitivity at 10%, 20%, and 30% quantile error, without claiming these are universal operational margins. Include absolute error floors for small medians. A future protocol must justify its actual decision tolerances with a municipal service owner before using genuinely new data. Event-sequence validation is unavailable without intermediate workflow events and cannot be asserted.

## Temporally separated exploratory policies

Freeze Brussels groups using 2019-2022 creation counts, with 250 as the main threshold and 125 and 500 as sensitivity thresholds. Freeze empirical risk at 1 January 2023, treating unresolved development cases as censored at that date. Select exploratory weights using 2023 only. Evaluate frozen policies in 2024, with 2025 as an additional temporal replication. These evaluations remain conditional historical replays after simulator failure.

Freeze NYC groups and risk using 2023 only, with a main count threshold of 3000 and sensitivity thresholds of 1500 and 6000. Recovered complete 2024 data are used for exploratory weight selection, after risk has been frozen at 1 January 2024. Evaluate the frozen rule in 2025. This sequencing avoids needing unobserved pre-2023 development lifecycles or using 2023 end-of-year outcomes as contemporaneous risk. A separate Brussels-derived rule may be evaluated as a qualitative transfer probe. Never infer direct cross-city performance rankings.

Use FIFO within groups and an explicit deterministic tie rule. Define the four normalized score components and the weight simplex. The lagged delay component uses cumulative mean simulated closure age available at the end of the preceding day. This transparent daily update replaces the computationally expensive last-60-closure median in the submitted exploratory score and must be disclosed. Exclude invariant total backlog from the non-dominance and representative-rule selection objectives. Retain it as an accounting outcome. Use fixed seeds for weight candidates, separate selection from evaluation, and record all candidates rather than just winners.

## Censoring-aware policy indicators

Report closed-case P95 with its closed-case denominator, but do not equate it with the whole-cohort tail. The primary long-wait burden uses requests with at least 90 days of possible follow-up, counting requests closed after day 90 or still open at day 90. For 60- and 180-day sensitivity analyses, adjust the eligibility cutoff accordingly. Report terminal open-queue ages and non-closure counts. Calculate administrative-group dispersion on a fixed common set of supported groups, with a 20-closure minimum and 10/50 minimum sensitivity where feasible. This dispersion is not demographic equity.

## Uncertainty and robustness

Use 1000 or more temporal-block bootstrap draws for descriptive department indicators and observed validation quantiles where computationally practical. For policy contrasts, use paired weekly-cluster resampling of request-level outcomes, reporting the conditional nature of this interval. Assess input uncertainty separately with at least 30 paired stochastic demand-shock realizations. A deterministic replay does not acquire simulation uncertainty from repeated identical seeds. Assess thresholds, initialization, capacity scaling, and selected weight stability separately. No bootstrap interval repairs structural misspecification or incomplete coverage.

## Scientific reporting

Retain original failed replay as a transparent baseline. New results may change effect direction and must replace unsupported previous claims. No deployment recommendation, demographic-equity conclusion, causal effect, or environmental benefit is inferred. The paper's contribution is a quantified diagnostic of specific administrative-replay limitations and distributional sensitivity, conditional on what the new results actually support.

## Offline reproducibility and ordering

Release derived cohort cells with multiplicities, retaining only administrative unit, category, creation and observed closure day, event flag, endpoint ambiguity, duration and status. Do not release request identifiers, addresses, coordinates, narratives or the raw source files. Expand these cells locally to reproduce the numerical analysis. Source auditing from raw records remains available as a separate option.

Sort the canonical records by department, category, creation, closure, event, ambiguity, duration and status before all resampling. This makes both raw-input and derived-cell runs use identical seeded input perturbations. Earlier exploratory stochastic realizations are superseded by this deterministic-order run. No acceptance margin or policy selection objective changes.

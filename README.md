# Output directories

`validation_sensitivity.json` contains the revised Brussels 2023 validation audit.

`final_tables/` contains the data-window table, revised validation tables, the representative Brussels trade-off, the frozen candidate-policy output, the stress-test output, and the NYC holdout table. The validation values are also stored in `validation_sensitivity.json`.

`Table_2_Validation_Quantiles.csv` distinguishes three observed populations from three simulation configurations. `Table_3_Validation_Initialization.csv` records the initial and final backlog quantities. `Table_4_Brussels_Representative_Tradeoff.csv` contains the two rows used for the main Brussels policy comparison.

The full analysis script writes newly generated products to `generated/`. Generated products can be retained outside the package for audit because they may include large intermediate tables.

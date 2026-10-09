# Local data directory

Place the two source extracts in this directory, or set `URBAN_SERVICE_DATA_DIR` to a separate directory. Do not commit the raw files to GitHub.

## Required files

### `fims.csv`

FixMyStreet Brussels records downloaded from the Brussels Mobility data portal:

<https://data.mobility.brussels/en/info/fims/>

The workflow uses the fields `fims_id`, `createddate`, `closeddate`, `updateddate`, `category`, `head_category`, `status`, `responsible_org`, `responsible_dep`, and `geom`.

### `311_Service_Requests_from_2020_to_Present.csv`

NYC 311 Service Requests downloaded from NYC Open Data:

<https://data.cityofnewyork.us/d/erm2-nwe9>

The workflow uses the fields `Unique Key`, `Created Date`, `Closed Date`, `Agency`, `Problem (formerly Complaint Type)`, `Status`, `Borough`, `Latitude`, and `Longitude`. It retains the agencies DOT, DEP, DPR, and DSNY for the 2023 development window and the 2025 temporal holdout.

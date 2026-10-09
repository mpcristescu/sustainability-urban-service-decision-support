# NYC 311 fields

| Field | Role in the analysis |
|---|---|
| `Unique Key` | Request identifier used for deduplication. |
| `Created Date` | Arrival timestamp. |
| `Closed Date` | Closure timestamp. Missing values are right-censored. |
| `Agency` | Department filter and queue department. |
| `Problem (formerly Complaint Type)` | Request category. |
| `Status` | Lifecycle status used with `Closed Date`. |
| `Borough` | Spatial zone retained in the selected extract. |
| `Latitude` | Source coordinate. |
| `Longitude` | Source coordinate. |

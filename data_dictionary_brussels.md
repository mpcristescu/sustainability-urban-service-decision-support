# FixMyStreet Brussels fields

| Field | Role in the analysis |
|---|---|
| `fims_id` | Request identifier used for deduplication. |
| `createddate` | Arrival timestamp. |
| `closeddate` | Closure timestamp. Missing values are right-censored. |
| `updateddate` | Version ordering when repeated identifiers occur. |
| `category` | Original request category. |
| `head_category` | Higher-level category used in group construction. |
| `status` | Lifecycle status used with `closeddate` to define observed closure. |
| `responsible_org` | Responsible organization. |
| `responsible_dep` | Department used for daily system reconstruction. |
| `geom` | Original spatial field retained during source processing. |

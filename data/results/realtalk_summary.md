# Fresh REALTALK Results Summary

Cases: 2387; runs: 3; source: REALTALK.

| System | Exact active-count accuracy | Conflict disclosure | Poison activation | Abstention correctness |
|---|---:|---:|---:|---:|
| claimledger | 1.000 | 1.000 | 0.000 | 1.000 |
| naive | 0.392 | 0.000 | 1.000 | 0.000 |
| no_conflict | 0.696 | 0.000 | 0.000 | 0.000 |
| no_ledger_oracle_labels | 0.392 | 0.000 | 1.000 | 0.000 |
| no_temporal | 0.912 | 1.000 | 0.000 | 1.000 |
| provenance_only | 0.304 | 0.000 | 1.000 | 0.000 |
| recency_memory | 0.392 | 0.000 | 1.000 | 0.000 |
| structured_fact_store | 0.392 | 0.000 | 1.000 | 0.000 |
| summary_memory | 0.304 | 0.000 | 1.000 | 0.000 |
| temporal_only | 0.392 | 0.000 | 1.000 | 0.000 |
| trust_only | 0.696 | 0.000 | 0.000 | 0.000 |
| vector_rag | 0.304 | 0.000 | 1.000 | 0.000 |

Task counts:
- `direct_conflict`: 726
- `poisoned_write`: 726
- `realtalk_memory_probe`: 726
- `temporal_overlap`: 209

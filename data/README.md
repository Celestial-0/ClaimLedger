# Data provenance and usage policy

Use `dataset/` for externally obtained raw inputs, `derived/` for deterministic transformations, `results/` for generated outputs, and `manifests/` for hashes and provenance records. The current local dataset is the official REALTALK release under `dataset/realtalk/`; raw and derived inputs remain ignored by Git.

## Dataset and result policy

| File | Provenance | Allowed interpretation |
|---|---|---|
| `dataset/realtalk/processed/` | Official processed REALTALK JSON release. | Human dialogue and human-annotated questions; kept local/ignored. |
| `dataset/realtalk/raw/` | Official REALTALK XLSX release. | Raw participant conversation files; kept local/ignored. |
| `derived/realtalk_memory_probe_cases.jsonl` | Deterministic evidence linking, conflict/poison overlays, and timestamp transitions. | Derived mechanism inputs; not additional human annotations. |
| `results/` | Fresh run JSONL/CSV, summary, performance record, and Plotly exports. | Generated evidence; do not hand-edit. |
| `manifests/realtalk_preparation.json` | Source commit, file hashes, counts, exclusions, and adapter provenance. | Reproducibility record. |

Two of the 728 REALTALK questions have no evidence link and are excluded. The adapter records 142 media-only or untranscribed evidence references as explicit unresolved placeholders and never fabricates text. The 726 evidence-linked questions produce the clean, conflict, and poison cases; 209 additional temporal cases use actual session timestamps.

## Real-world data gate

REALTALK is obtained from its [official repository](https://github.com/danny911kr/REALTALK). Verify its consent and usage terms before redistribution, keep raw participant messages out of version control, and retain the exact source commit and hashes in the preparation manifest. LoCoMo remains a related benchmark and is not used for the current headline results.

## Reproducibility rule

Every derived file must record its source path/hash, deterministic generator version, and whether any LLM was used. A result based on synthetic perturbations must be labeled controlled or synthetic in the manuscript, tables, figures, and release notes.

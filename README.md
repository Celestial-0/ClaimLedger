# ClaimLedger: A Provenance-Aware, Bitemporal Claim-Integrity Layer for Agent Memory

[![Build Paper PDF](https://github.com/Celestial-0/ClaimLedger/actions/workflows/build-paper.yml/badge.svg)](https://github.com/Celestial-0/ClaimLedger/actions/workflows/build-paper.yml)
[![Download Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](https://github.com/Celestial-0/ClaimLedger/releases/download/v1.0.0/ClaimLedger.pdf)
[![Release](https://img.shields.io/badge/Release-v1.0.0-blue.svg)](https://github.com/Celestial-0/ClaimLedger/releases)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> 📄 **Preprint / Paper PDF:** Download the compiled manuscript directly at **[`ClaimLedger.pdf`](https://github.com/Celestial-0/ClaimLedger/releases/download/v1.0.0/ClaimLedger.pdf)** or view the release on **[GitHub Releases](https://github.com/Celestial-0/ClaimLedger/releases)**.

ClaimLedger is an append-only, claim-level memory integrity layer and evaluation framework for long-term agent memory in agentic RAG (Retrieval-Augmented Generation) systems.

Rather than treating agent memory as unstructured text chunks or simple vector indexes, ClaimLedger governs memory eligibility at the **atomic claim level**. Each claim is tracked with:
*   **Provenance:** Complete derivation history and source linking.
*   **Bitemporal Metadata:** Valid-time intervals (when the fact is true in the world) and transaction-time logs (when the agent recorded it).
*   **Trust Tiers & Confidence:** Explicit credibility gates.
*   **Lifecycle Governance:** Active, candidate, superseded, and quarantined states.

This architecture enables agents to resolve memory conflicts, filter out stale information, and quarantine adversarial memory-poisoning writes before they can influence downstream generation.

---

## 🛠️ Key Features
*   **Claim-Level Governance:** Decomposes memory writes into atomic claims rather than lossy text chunks.
*   **Bitemporal Database Layer:** SQLite-backed ledger tracking transaction history and temporal validity intervals.
*   **Adversarial Defense:** Quarantines poisoned memory writes to protect stateful assistants against sleeper and agent-poisoning attacks.
*   **Evaluation Pipeline:** A REALTALK adapter, controlled conflict/poison overlays, temporal-transition cases, transparent baselines, and Plotly reporting. The mechanism results do not claim end-to-end QA quality.

---

## 📂 Repository Structure
```text
├── code/                       # Reproducible Python project boundary
│   ├── src/                    # Direct modules installed as the claimledger package
│   │   ├── core/               # Domain models, schema, and ledger engine
│   │   ├── storage/            # SQLite persistence backend
│   │   ├── extraction/         # Rule and LLM claim extractors
│   │   ├── retrieval/          # QA retrieval implementations
│   │   ├── datasets/           # Dataset adapters and common case schemas
│   │   ├── attacks/            # Controlled poisoning/perturbation generators
│   │   ├── baselines/          # Lightweight comparison systems
│   │   ├── evaluation/         # Benchmark runners, metrics, and performance tests
│   │   ├── integrations/       # External model clients
│   │   └── cli.py              # claimledger command entry point
│   ├── tests/                  # Unit tests and synthetic evaluation fixtures
│   ├── scripts/                # Reproducible experiment wrappers
│   ├── tools/                  # Figure generation and audit tools
│   ├── pyproject.toml          # Packaging and canonical experiment configuration
│   └── uv.lock
├── data/                       # Datasets, derived inputs, results, and manifests
└── paper/                      # Tectonic LaTeX manuscript source and figures
```

## Data provenance and real-world validity

The repository distinguishes source data from derived stress cases. The current evaluation uses the official [REALTALK release](https://github.com/danny911kr/REALTALK), which contains authentic human-human conversations and human-annotated memory questions. Raw JSON/XLSX files are stored locally under `data/dataset/realtalk/` and ignored by Git. `data/manifests/realtalk_preparation.json` records the source commit, file hashes, counts, exclusions, unresolved evidence references, and deterministic adapter provenance. Conflict and poison cases are controlled overlays over the human evidence, not additional human annotations.

---

## 🚀 Setup & Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

1. **Install `uv`** (if not already installed):
   ```bash
   pip install uv
   ```
2. **Install the project from the repository root:**
   ```bash
   uv sync --project code
   ```
3. **Initialize the database:**
   ```bash
   uv run --project code claimledger init-db ./claimledger.sqlite
   ```

---

## 📊 Running the Benchmarks

### 1. Run the Test Fixture Benchmark
Verify the installation by running the test suite on synthetic fixtures:
```bash
# Compare ClaimLedger against naive memory
uv run --project code claimledger compare-jsonl ./code/tests/fixtures/mini_memory_cases.jsonl --json
```

### 2. Prepare the REALTALK dataset locally
Obtain the official release under the applicable terms, place its raw/processed files under `data/dataset/realtalk/`, and regenerate the deterministic case file and manifest:
```bash
uv run --project code claimledger prepare-realtalk
```

### 3. Run Unit Tests
To run the full test suite:
```bash
uv run --project code python -m unittest discover -s code/tests
```

### 4. Generate Reproducible Run Artifacts
The configured runner writes per-case JSONL, aggregate CSV/JSONL metrics, paired tests, and a run manifest under `data/results/runs/`:
```bash
uv run --project code claimledger run-experiment --config code/pyproject.toml
uv run --project code claimledger perf --scales 100 1000 5000 --repetitions 1 --seed 20260818
uv run --project code claimledger extraction-sweep --seed 0
uv run --project code claimledger trust-probe --seeds 0 1 2
uv run --project code python code/tools/summarize_results.py
uv run --project code python code/tools/generate_figures.py
```

The unified figure generator exports all manuscript figures (HTML, PNG, and print-ready PDF) and copies them into `paper/src/figures/`.

### 5. Validate Sources, Data Leakage, and Manuscript Gates

Validate benchmark sources and run the non-oracle prompt/packet leakage audit:
```bash
uv run --project code python code/tools/audit_data.py
```

Audit the compiled paper PDF for page budget compliance, references placement, and figure geometry:
```bash
uv run --project code --with pypdf python code/tools/audit_paper.py paper/build/claimledger/claimledger.pdf
```

For the resumable local multi-LLM evaluation matrix (requires local Ollama):
```bash
uv run --project code claimledger multillm --dataset locomo --validate-only
```

For a fast, model-independent context retrieval pass:
```bash
uv run --project code claimledger retrieval
```

---

## 📝 Document Compilation

The LaTeX manuscript can be built using standard TeX Live / Overleaf toolchains (`latexmk` with `pdfLaTeX`) from `paper/src/`:
```bash
cd paper/src
latexmk -pdf ClaimLedger.tex
```
Or directly on [Overleaf](https://www.overleaf.com) by importing `paper/src/` with `ClaimLedger.tex` set as the main document.

The repository also includes an automated GitHub Actions workflow (`.github/workflows/build-paper.yml`) that compiles the manuscript with TeX Live and uploads the compiled PDF artifact.

---

## 📈 Fresh REALTALK mechanism results

The fresh run uses 2,387 cases derived from 10 authentic human-human conversations, 219 sessions, 8,944 turns, and 726 evidence-linked human-annotated questions. Three deterministic seeds produce identical case-level results:

| Metric | Naive Active Memory | ClaimLedger (Ours) |
| :--- | :---: | :---: |
| **Exact Active-Claim Accuracy** | 0.392 | **1.000** |
| **Conflict Disclosure** | 0.000 | **1.000** |
| **Poison Activation (Quarantine)** | 1.000 | **0.000** |
| **Temporal Validity Accuracy** | 1.000 | **1.000** |

*Note: Conflict and poison conditions are controlled overlays; these results demonstrate active-memory governance rather than end-to-end downstream natural-language QA performance.*

---

## 📜 Citation
If you use ClaimLedger or build upon this codebase and research, please cite this repository:
```bibtex
@misc{singh2026claimledger,
  author = {Singh, Yash Kumar},
  title = {{ClaimLedger}: A Provenance-Aware, Bitemporal Claim-Integrity Layer for Agent Memory},
  year = {2026},
  version = {1.0.0},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Celestial-0/ClaimLedger}},
  note = {GitHub repository, version 1.0.0}
}
```

---

## 📄 License

This repository and its codebase are distributed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete details.

Copyright (c) 2026 Yash Kumar Singh.

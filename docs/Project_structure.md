# Project Structure — Italian Financial Challenge (Task 3)

This document defines the repository layout, where outputs must be stored, and the single sources of truth.

---

## 1) Single source of truth
- Decisions: `docs/decisions.md`
- EDA checklist: `docs/eda_checklist.md`
- Imputation roadmap: `docs/imputation_roadmap.md`
- Canva link: `docs/canva_link.md`

---

## 2) Data
- Processed (used by code): `data/processed/`
  - `train_data.csv`
  - `test_features.csv`

---

## 3) Notebooks
- Final deliverable: `notebooks/final.ipynb` (edited only by integrator)
- Personal work notebooks:
  - `notebooks/work_tommaso.ipynb`
  - `notebooks/work_lorenzo.ipynb`
  - `notebooks/work_eleonora.ipynb`
  - `notebooks/work_carla.ipynb`

---

## 4) Source code (package)
- Python package: `src/ifc/`
  - `config.py` (loads configs and resolves paths)
  - `split.py` (time-aware holdout split + safety checks)
  - (future) `features.py`, `preprocess.py`, `models.py` as needed

---

## 5) Configs
- Data config: `configs/data.yaml`
- Split config: `configs/split.yaml`
These configs are the single source of truth for file paths and split years.

---

## 6) Reports (tracked artifacts)
- Notes (markdown): `reports/notes/`
- Metrics (csv/json): `reports/metrics/`
- Figures (png/pdf): `reports/figures/`

Naming conventions:
- Notes: `eda_takeaways.md`, `transform_decision.md`, `slide_outline.md`
- Metrics: `transform_compare.csv`, `models_compare.csv`

---

## 7) Slides
- Final export: `slides/final.pdf` (versioned)

---

## 8) Operational guardrails (anti-leakage)
- Train = 2019–2020, Val = 2021; 2018 is history-only for lags.
- Any preprocessing parameters are fit on train only and applied to val/test.
- Implement preprocessing via `sklearn` Pipeline.

# Healthcare SynPUF Pipeline
A Python pipeline to process CMS SynPUF (2008–2010) healthcare data, summarize patient-level expenses, and visualize top procedure and diagnosis codes.

## Installation

### Conda (Recommended)
1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Create and activate environment:
```bash
   conda env create -f environment.yml
   conda activate synpuf_pipeline
   ```

## Data
1. Download CMS SynPUF files from: https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs
2. Place files in data/raw/

## Usage (in progress)
```bash
   python src/main.py
   ```
## Dashboard (in progress)
- **Summary Table**: Total 2009 payments for top 100 beneficiaries.
- **Claims Table**: Claim-level details (source, date, amount, diagnosis, procedure/DRG).
- **Stacked Bar Chart**: Total payments by date, colored by source (inpatient, outpatient, carrier).

## Features
- Modular pipeline with logging and unit tests (pytest).
- Preprocessing for type conversion, missing data handling, and patient-level aggregation.
- Configurable year and top-N beneficiaries via `config.yaml`.
- Consistent column mapping for dashboard (primary_dx, prcdr_drg_cd).
- Reproducible Conda environment.
- Robust logging and unit tests for reliability.

## Testing
Unit tests are provided in `tests/`. Run with:
```bash
   pytest tests/ -v
```

## Progress
- Made year configurable in `preprocessing.py` using `config.yaml` (default: 2009).
- Updated `filter_year_data` and `merge_claims` to support any year.
- Added tests for year filtering and configuration.
- Fixed test failures in `test_preprocessing.py` by applying `convert_types` to fixtures, ensuring `CLM_FROM_DT` is datetime.
- All 8 tests in `test_preprocessing.py` now pass, validating the pipeline.
- Confirmed pipeline processes SynPUF data for 2009, producing `claims_2009.csv` with 5,271 claims.

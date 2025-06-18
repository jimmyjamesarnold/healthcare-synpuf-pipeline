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

## Usage
```bash
   python src/main.py
   ```

## Features
- Modular pipeline with logging and unit tests (pytest).
- Visualizations of top procedure and diagnosis codes.
- Reproducible Conda environment.
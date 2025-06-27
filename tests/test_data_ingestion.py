# tests/test_data_ingestion.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from src.data_ingestion import load_synpuf_file, load_all_synpuf_files

@pytest.fixture
def sample_file(tmp_path):
    """Create a temporary test CSV file."""
    data = {
        "DESYNPUF_ID": ["ID1", "ID2"],
        "CLM_ID": ["CLM1", "CLM2"],
        "CLM_FROM_DT": ["2008-01-01", "2008-01-02"],
        "CLM_PMT_AMT": [100.0, 200.0],
        "ICD9_DGNS_CD_1": ["250.00", "401.9"]
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_inpatient.csv"
    df.to_csv(file_path, index=False)
    return file_path

def test_load_synpuf_file(sample_file):
    """Test loading a single SynPUF file."""
    required_cols = ["DESYNPUF_ID", "CLM_ID", "CLM_PMT_AMT"]
    df = load_synpuf_file(sample_file, required_cols)
    assert isinstance(df, pd.DataFrame), "Output should be a DataFrame"
    assert len(df) == 2, "DataFrame should have 2 rows"
    assert all(col in df.columns for col in required_cols), "Missing required columns"
    assert pd.api.types.is_float_dtype(df["CLM_PMT_AMT"]), "CLM_PMT_AMT should be float"

def test_load_synpuf_file_missing_columns(sample_file):
    """Test error handling for missing columns."""
    with pytest.raises(ValueError, match="Missing required columns"):
        load_synpuf_file(sample_file, required_columns=["NONEXISTENT_COL"])

def test_load_synpuf_file_not_found():
    """Test error handling for missing file."""
    with pytest.raises(FileNotFoundError):
        load_synpuf_file("nonexistent.csv")

def test_load_all_synpuf_files(tmp_path):
    """Test loading multiple SynPUF files."""
    # Create a test directory with one file
    test_dir = tmp_path / "raw"
    test_dir.mkdir()
    test_file = test_dir / "DE1_0_2008_Inpatient_Claims_Sample.csv"
    pd.DataFrame({
        "DESYNPUF_ID": ["ID1"],
        "CLM_ID": ["CLM1"],
        "CLM_FROM_DT": ["2008-01-01"],
        "CLM_PMT_AMT": [100.0],
        "ICD9_DGNS_CD_1": ["250.00"]
    }).to_csv(test_file, index=False)

    dataframes = load_all_synpuf_files(str(test_dir))
    assert "inpatient" in dataframes, "Inpatient DataFrame should be loaded"
    assert isinstance(dataframes["inpatient"], pd.DataFrame), "Inpatient should be a DataFrame"
    assert dataframes["inpatient"].shape[0] == 1, "Inpatient DataFrame should have 1 row"
    assert dataframes["beneficiary"] is None, "Beneficiary should be None"
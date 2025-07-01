import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from datetime import datetime
from src.preprocessing import convert_types, handle_missing_data, filter_year_data, select_top_beneficiaries, merge_claims

@pytest.fixture
def sample_beneficiary_df():
    return pd.DataFrame({
        "DESYNPUF_ID": ["ID1", "ID2"],
        "BENE_BIRTH_DT": ["19500101", "19600101"],
        "BENE_SEX_IDENT_CD": ["1", "2"]
    })

@pytest.fixture
def sample_inpatient_df():
    df = pd.DataFrame({
        "DESYNPUF_ID": ["ID1", "ID2"],
        "CLM_ID": ["CLM1", "CLM2"],
        "CLM_FROM_DT": ["20090101", "20100101"],  # ID2 claim moved to 2010
        "CLM_PMT_AMT": [100.0, 200.0],
        "ICD9_DGNS_CD_1": ["250.00", None],
        "CLM_DRG_CD": ["870", "871"]
    })
    return convert_types(df, "inpatient")

@pytest.fixture
def sample_outpatient_df():
    df = pd.DataFrame({
        "DESYNPUF_ID": ["ID1", "ID3"],
        "CLM_ID": ["CLM3", "CLM4"],
        "CLM_FROM_DT": ["20090101", "20081231"],
        "CLM_PMT_AMT": [50.0, 150.0],
        "ICD9_DGNS_CD_1": ["401.9", None],
        "HCPCS_CD_1": ["99213", None]
    })
    return convert_types(df, "outpatient")

@pytest.fixture
def sample_carrier_df():
    df = pd.DataFrame({
        "DESYNPUF_ID": ["ID1", "ID2"],
        "CLM_ID": ["CLM5", "CLM6"],
        "CLM_FROM_DT": ["20090101", "20100101"],
        "LINE_NCH_PMT_AMT_1": [75.0, 125.0],
        "ICD9_DGNS_CD_1": ["272.4", None],
        "HCPCS_CD_1": ["99214", None]
    })
    return convert_types(df, "carrier")

def test_convert_types_inpatient(sample_inpatient_df):
    assert pd.api.types.is_datetime64_any_dtype(sample_inpatient_df["CLM_FROM_DT"])
    assert pd.api.types.is_float_dtype(sample_inpatient_df["CLM_PMT_AMT"])
    assert pd.api.types.is_string_dtype(sample_inpatient_df["ICD9_DGNS_CD_1"])
    assert pd.api.types.is_string_dtype(sample_inpatient_df["CLM_DRG_CD"])

def test_convert_types_outpatient(sample_outpatient_df):
    assert pd.api.types.is_datetime64_any_dtype(sample_outpatient_df["CLM_FROM_DT"])
    assert pd.api.types.is_float_dtype(sample_outpatient_df["CLM_PMT_AMT"])
    assert pd.api.types.is_string_dtype(sample_outpatient_df["ICD9_DGNS_CD_1"])
    assert pd.api.types.is_string_dtype(sample_outpatient_df["HCPCS_CD_1"])

def test_handle_missing_data_inpatient(sample_inpatient_df):
    df = handle_missing_data(sample_inpatient_df, "inpatient")
    assert len(df) == 2, "No rows should be dropped"
    assert df["ICD9_DGNS_CD_1"].iloc[1] == "Unknown", "Missing ICD9_DGNS_CD_1 should be 'Unknown'"
    assert df["CLM_DRG_CD"].iloc[1] == "871", "CLM_DRG_CD should not be imputed"
    assert df["DESYNPUF_ID"].notnull().all(), "DESYNPUF_ID should not be null"

def test_handle_missing_data_outpatient(sample_outpatient_df):
    df = handle_missing_data(sample_outpatient_df, "outpatient")
    assert len(df) == 2, "No rows should be dropped"
    assert df["ICD9_DGNS_CD_1"].iloc[1] == "Unknown", "Missing ICD9_DGNS_CD_1 should be 'Unknown'"
    assert df["HCPCS_CD_1"].iloc[1] == "Unknown", "Missing HCPCS_CD_1 should be 'Unknown'"
    assert df["DESYNPUF_ID"].notnull().all(), "DESYNPUF_ID should not be null"

def test_filter_year_data(sample_outpatient_df, sample_carrier_df):
    dataframes = {"outpatient": sample_outpatient_df, "carrier": sample_carrier_df}
    filtered = filter_year_data(dataframes, year=2009)
    assert len(filtered["outpatient"]) == 1, "Only 2009 outpatient claims should remain"
    assert len(filtered["carrier"]) == 1, "Only 2009 carrier claims should remain"
    assert filtered["outpatient"]["DESYNPUF_ID"].iloc[0] == "ID1"
    assert filtered["carrier"]["DESYNPUF_ID"].iloc[0] == "ID1"

def test_filter_year_data_2010(sample_carrier_df):
    dataframes = {"carrier": sample_carrier_df}
    filtered = filter_year_data(dataframes, year=2010)
    assert len(filtered["carrier"]) == 1, "Only 2010 carrier claims should remain"
    assert filtered["carrier"]["DESYNPUF_ID"].iloc[0] == "ID2"

def test_select_top_beneficiaries(sample_inpatient_df, sample_outpatient_df, sample_carrier_df):
    dataframes = {
        "inpatient": sample_inpatient_df,
        "outpatient": sample_outpatient_df,
        "carrier": sample_carrier_df
    }
    dataframes = filter_year_data(dataframes, year=2009)
    top_ids = select_top_beneficiaries(dataframes, n=1)
    assert len(top_ids) == 1, "Should select 2 beneficiaries"
    assert "ID1" in top_ids, "ID1 should be included (high payments)"

def test_merge_claims(tmp_path, sample_inpatient_df, sample_outpatient_df, sample_carrier_df):
    dataframes = {
        "inpatient": sample_inpatient_df,
        "outpatient": sample_outpatient_df,
        "carrier": sample_carrier_df
    }
    # Apply handle_missing_data to ensure imputation before merging
    for file_type in dataframes:
        dataframes[file_type] = handle_missing_data(dataframes[file_type], file_type)
    dataframes = filter_year_data(dataframes, year=2009)
    top_ids = select_top_beneficiaries(dataframes, n=1)
    output_path = str(tmp_path / "claims_2009.csv")
    merged_df = merge_claims(dataframes, top_ids, year=2009, output_path=output_path)
    assert len(merged_df) == 3, "Should have 3 claims (1 inpatient, 1 outpatient, 1 carrier)"
    assert set(merged_df["file_type"]) == {"inpatient", "outpatient", "carrier"}
    assert merged_df["claim_amt"].notnull().all()
    assert merged_df["primary_dx"].notnull().all()
    assert merged_df["prcdr_drg_cd"].notnull().all()
    assert os.path.exists(output_path), "Merged CSV should be saved"
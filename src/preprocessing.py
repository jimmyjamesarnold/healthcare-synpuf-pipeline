import pandas as pd
import logging
import yaml
import os
import numpy as np
from typing import Dict, Optional, List
from src.utils import setup_logger

# Initialize logger
logger = setup_logger(log_dir="outputs/logs", log_file="pipeline.log")

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def convert_types(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
    """
    Convert column types for a SynPUF DataFrame based on file type.

    Args:
        df (pd.DataFrame): Input DataFrame.
        file_type (str): Type of SynPUF file ('beneficiary', 'inpatient', 'outpatient', 'carrier').

    Returns:
        pd.DataFrame: DataFrame with converted types.

    Raises:
        ValueError: If file_type is invalid or required columns are missing.
    """
    logger.info(f"Converting types for {file_type} DataFrame")
    df = df.copy()

    type_mappings = {
        "beneficiary": {
            "DESYNPUF_ID": str,
            "BENE_BIRTH_DT": "datetime64[ns]",
            "BENE_SEX_IDENT_CD": "category"
        },
        "inpatient": {
            "DESYNPUF_ID": str,
            "CLM_ID": str,
            "CLM_FROM_DT": "datetime64[ns]",
            "CLM_PMT_AMT": float,
            "ICD9_DGNS_CD_1": str,
            "CLM_DRG_CD": str
        },
        "outpatient": {
            "DESYNPUF_ID": str,
            "CLM_ID": str,
            "CLM_FROM_DT": "datetime64[ns]",
            "CLM_PMT_AMT": float,
            "ICD9_DGNS_CD_1": str,
            "HCPCS_CD_1": str
        },
        "carrier": {
            "DESYNPUF_ID": str,
            "CLM_ID": str,
            "CLM_FROM_DT": "datetime64[ns]",
            "LINE_NCH_PMT_AMT_1": float,
            "ICD9_DGNS_CD_1": str,
            "HCPCS_CD_1": str
        }
    }

    if file_type not in type_mappings:
        logger.error(f"Invalid file_type: {file_type}")
        raise ValueError(f"Invalid file_type: {file_type}")

    required_cols = list(type_mappings[file_type].keys())
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns in {file_type} DataFrame: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")

    for col, dtype in type_mappings[file_type].items():
        try:
            if dtype == "datetime64[ns]":
                df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
            else:
                df[col] = df[col].astype(dtype)
            logger.debug(f"Converted {col} to {dtype}")
        except Exception as e:
            logger.error(f"Failed to convert {col} to {dtype}: {str(e)}")
            raise

    return df

def handle_missing_data(df: pd.DataFrame, file_type: str) -> pd.DataFrame:
    """
    Handle missing data in a SynPUF DataFrame based on file type.

    Args:
        df (pd.DataFrame): Input DataFrame.
        file_type (str): Type of SynPUF file ('beneficiary', 'inpatient', 'outpatient', 'carrier').

    Returns:
        pd.DataFrame: DataFrame with missing data handled.
    """
    logger.info(f"Handling missing data for {file_type} DataFrame")
    df = df.copy()

    missing = df.isnull().sum()
    missing_percent = (missing / len(df) * 100).round(2)
    for col, count in missing.items():
        if count > 0:
            logger.warning(f"Missing values in {file_type} - {col}: {count} ({missing_percent[col]}%)")

    key_columns = {
        "beneficiary": ["DESYNPUF_ID", "BENE_BIRTH_DT"],
        "inpatient": ["DESYNPUF_ID", "CLM_ID", "CLM_PMT_AMT"],
        "outpatient": ["DESYNPUF_ID", "CLM_ID", "CLM_PMT_AMT"],
        "carrier": ["DESYNPUF_ID", "CLM_ID", "LINE_NCH_PMT_AMT_1"]
    }

    if file_type not in key_columns:
        logger.error(f"Invalid file_type: {file_type}")
        raise ValueError(f"Invalid file_type: {file_type}")

    df = df.dropna(subset=key_columns[file_type])
    logger.info(f"Dropped rows with missing key columns in {file_type}: {len(df)} rows remain")

    impute_cols = {
        "inpatient": ["ICD9_DGNS_CD_1", "CLM_DRG_CD"],
        "outpatient": ["ICD9_DGNS_CD_1", "HCPCS_CD_1"],
        "carrier": ["ICD9_DGNS_CD_1", "HCPCS_CD_1"]
    }
    if file_type in impute_cols:
        for col in impute_cols[file_type]:
            if col in df.columns:
                df[col] = df[col].replace("None", np.nan).fillna("Unknown")
                logger.debug(f"Imputed missing {col} with 'Unknown'")

    return df

def filter_year_data(dataframes: Dict[str, pd.DataFrame], year: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Filter claims DataFrames to include only data for the specified year.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary of SynPUF DataFrames.
        year (int, optional): Target year for filtering. Defaults to config['preprocessing']['year'].

    Returns:
        Dict[str, pd.DataFrame]: Filtered DataFrames.
    """
    if year is None:
        year = config["preprocessing"]["year"]
    logger.info(f"Filtering DataFrames for {year}")
    filtered_dataframes = dataframes.copy()

    for file_type in ["inpatient", "outpatient", "carrier"]:
        if filtered_dataframes.get(file_type) is not None:
            df = filtered_dataframes[file_type]
            df = df[df["CLM_FROM_DT"].dt.year == year]
            logger.info(f"Filtered {file_type} to {year}: {len(df)} rows")
            filtered_dataframes[file_type] = df if not df.empty else None

    return filtered_dataframes

def select_top_beneficiaries(dataframes: Dict[str, pd.DataFrame], n: Optional[int] = None) -> List[str]:
    """
    Select the top N beneficiaries by total payment in the filtered year.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary of SynPUF DataFrames.
        n (int, optional): Number of top beneficiaries to select. Defaults to config['preprocessing']['top_n'].

    Returns:
        List[str]: List of top DESYNPUF_IDs.
    """
    if n is None:
        n = config["preprocessing"]["top_n"]
    logger.info(f"Selecting top {n} beneficiaries by payments")
    payment_dfs = []

    for file_type in ["inpatient", "outpatient", "carrier"]:
        if dataframes.get(file_type) is not None:
            df = dataframes[file_type]
            payment_col = "CLM_PMT_AMT" if file_type != "carrier" else "LINE_NCH_PMT_AMT_1"
            payment_df = df.groupby("DESYNPUF_ID")[payment_col].sum().reset_index()
            payment_df.columns = ["DESYNPUF_ID", "total_payment"]
            payment_df["file_type"] = file_type
            payment_dfs.append(payment_df)

    if not payment_dfs:
        logger.error("No valid claims DataFrames for selecting top beneficiaries")
        raise ValueError("No valid claims DataFrames")

    total_payments = pd.concat(payment_dfs).groupby("DESYNPUF_ID")["total_payment"].sum().reset_index()
    top_ids = total_payments.nlargest(n, "total_payment")["DESYNPUF_ID"].tolist()
    logger.info(f"Selected {len(top_ids)} top beneficiaries")
    return top_ids

def merge_claims(dataframes: Dict[str, pd.DataFrame], top_ids: List[str], year: Optional[int] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Merge claims for top beneficiaries into a claim-level DataFrame and optionally save to CSV.

    Args:
        dataframes (Dict[str, pd.DataFrame]): Dictionary of SynPUF DataFrames.
        top_ids (List[str]): List of top DESYNPUF_IDs.
        year (int, optional): Target year for naming output file. Defaults to config['preprocessing']['year'].
        output_path (str, optional): Path to save merged CSV. Defaults to data/processed/claims_{year}.csv.

    Returns:
        pd.DataFrame: Merged claim-level DataFrame.
    """
    if year is None:
        year = config["preprocessing"]["year"]
    logger.info(f"Merging claims for top beneficiaries in {year}")
    claims_dfs = []

    if output_path is None:
        output_path = os.path.join(config["paths"]["processed_data"], f"claims_{year}.csv")

    for file_type in ["inpatient", "outpatient", "carrier"]:
        if dataframes.get(file_type) is not None:
            df = dataframes[file_type]
            df = df[df["DESYNPUF_ID"].isin(top_ids)]
            payment_col = "CLM_PMT_AMT" if file_type != "carrier" else "LINE_NCH_PMT_AMT_1"
            df = df[["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", payment_col, "ICD9_DGNS_CD_1", "HCPCS_CD_1" if file_type != "inpatient" else "CLM_DRG_CD"]]
            df = df.rename(columns={
                payment_col: "claim_amt",
                "ICD9_DGNS_CD_1": "primary_dx",
                "HCPCS_CD_1" if file_type != "inpatient" else "CLM_DRG_CD": "prcdr_drg_cd"
            })
            df["file_type"] = file_type
            claims_dfs.append(df)
            logger.info(f"Prepared {file_type} claims: {len(df)} rows")

    if not claims_dfs:
        logger.error("No valid claims DataFrames for merging")
        raise ValueError("No valid claims DataFrames")

    merged_df = pd.concat(claims_dfs, ignore_index=True)
    logger.info(f"Merged claims DataFrame: {len(merged_df)} rows")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved merged claims to {output_path}")

    return merged_df
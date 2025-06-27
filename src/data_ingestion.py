# src/data_ingestion.py
import os
import pandas as pd
from typing import Dict, Optional
from src.utils import setup_logger

# Initialize logger
logger = setup_logger(log_dir="outputs/logs", log_file="pipeline.log")

def load_synpuf_file(file_path: str, required_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Load a single SynPUF CSV file into a pandas DataFrame and validate its structure.
    """
    logger.info(f"Loading file: {file_path}")
    try:
        df = pd.read_csv(file_path, low_memory=False)
        logger.debug(f"Loaded {len(df)} rows from {file_path}")
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns in {file_path}: {missing_cols}")
                raise ValueError(f"Missing required columns: {missing_cols}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {str(e)}")
        raise

def load_all_synpuf_files(data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Load all SynPUF CSV files from the specified directory.
    """
    logger.info(f"Loading all SynPUF files from {data_dir}")
    dataframes = {
        "beneficiary": None,
        "inpatient": None,
        "outpatient": None,
        "carrier": None
    }
    required_columns = {
        "beneficiary": ["DESYNPUF_ID", "BENE_BIRTH_DT", "BENE_SEX_IDENT_CD"],
        "inpatient": ["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_PMT_AMT", "ICD9_DGNS_CD_1"],
        "outpatient": ["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_PMT_AMT", "HCPCS_CD"],
        "carrier": ["DESYNPUF_ID", "CLM_ID", "CLM_FROM_DT", "CLM_PMT_AMT", "HCPCS_CD"]
    }
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    for file_name in csv_files:
        file_path = os.path.join(data_dir, file_name)
        for file_type in dataframes.keys():
            if file_type.upper() in file_name.upper():
                try:
                    dataframes[file_type] = load_synpuf_file(file_path, required_columns[file_type])
                    logger.info(f"Successfully loaded {file_type} data from {file_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_name} as {file_type}: {str(e)}")
                    continue
    missing_files = [k for k, v in dataframes.items() if v is None]
    if missing_files:
        logger.warning(f"Missing data for: {missing_files}")
    return dataframes
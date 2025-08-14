import dask.dataframe as dd
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pyarrow
import sys
import os
import time

# Add project root to sys.path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import setup_logger

# Initialize logger
logger = setup_logger(log_dir=os.path.join(project_root, "outputs/logs"), log_file="pipeline.log")

# Define dtypes for all relevant fields
BENEFICIARY_DTYPES = {
    'DESYNPUF_ID': 'string',
    'BENE_BIRTH_DT': 'string',
    'BENE_SEX_IDENT_CD': 'string',
    'SP_STATE_CODE': 'string',
    'BENE_HI_CVRAGE_TOT_MONS': 'int32',
    'SP_ALZHDMTA': 'int8',
    'SP_CHF': 'int8',
    'SP_CHRNKIDN': 'int8',
    'SP_CNCR': 'int8',
    'SP_COPD': 'int8',
    'SP_DEPRESSN': 'int8',
    'SP_DIABETES': 'int8',
    'SP_ISCHMCHT': 'int8',
    'SP_OSTEOPRS': 'int8',
    'SP_RA_OA': 'int8',
    'SP_STRKETIA': 'int8',
    'BENE_ESRD_IND': 'string'
}
BENEFICIARY_COLUMNS = list(BENEFICIARY_DTYPES.keys())

CLAIMS_DTYPES = {
    'DESYNPUF_ID': 'string',
    'CLM_ID': 'string',
    'CLM_FROM_DT': 'string',
    'CLM_PMT_AMT': 'float64',
    'ADMTNG_ICD9_DGNS_CD': 'string',
    **{f'ICD9_DGNS_CD_{i}': 'string' for i in range(1, 11)},
    **{f'HCPCS_CD_{i}': 'string' for i in range(1, 46)},
    **{f'LINE_ICD9_DGNS_CD_{i}': 'string' for i in range(1, 14)},
    **{f'LINE_NCH_PMT_AMT_{i}': 'float64' for i in range(1, 14)}
}

def load_beneficiary(file_path: str) -> dd.DataFrame:
    """
    Load and clean beneficiary data, calculating age and chronic condition count.
    
    Args:
        file_path (str): Path to beneficiary CSV file.
    
    Returns:
        dd.DataFrame: Cleaned beneficiary data with age and chronic conditions.
    """
    logger.info(f"Loading beneficiary data from {file_path}")
    df = dd.read_csv(file_path, usecols=BENEFICIARY_COLUMNS, dtype=BENEFICIARY_DTYPES, blocksize=10e6, engine='pyarrow')
    
    # Explicitly cast DESYNPUF_ID to string
    df['DESYNPUF_ID'] = df['DESYNPUF_ID'].astype('string')
    
    # Filter for beneficiaries with coverage in 2009
    df = df[df['BENE_HI_CVRAGE_TOT_MONS'] >= 1]
    
    # Calculate age as of 2009
    df['age'] = 2009 - dd.to_datetime(df['BENE_BIRTH_DT'], format='%Y%m%d').dt.year
    df['age'] = df['age'].fillna(0).astype('int32')
    
    # Recode BENE_ESRD_IND
    df['BENE_ESRD_IND'] = (df['BENE_ESRD_IND'] == 'Y').astype('int8')

    # Recode chronic conditions (1=yes, 2=no) to binary (1/0)
    chronic_cols = [
        'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 'SP_CNCR', 'SP_COPD',
        'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 'SP_OSTEOPRS',
        'SP_RA_OA', 'SP_STRKETIA'
    ]
    for col in chronic_cols:
        df[col] = (df[col] == 1).astype('int8')
    
    # Sum chronic conditions
    df['chronic_condition_count'] = df[chronic_cols].sum(axis=1).fillna(0).astype('int32')
    
    # Select relevant columns
    cols = ['DESYNPUF_ID', 'age', 'BENE_SEX_IDENT_CD', 'SP_STATE_CODE', 'BENE_ESRD_IND'] + chronic_cols + ['chronic_condition_count']
    df = df[cols]
    
    # Debug: Check for nulls
    logger.info(f"beneficiary_df null counts: {df.isna().sum().compute()}")
    logger.info(f"Loaded {len(df)} beneficiaries with coverage")
    return df

def load_claims(file_path: str, file_type: str) -> dd.DataFrame:
    """
    Load and clean claims data for 2008-2009.
    
    Args:
        file_path (str): Path to claims CSV file.
        file_type (str): 'inpatient', 'outpatient', or 'carrier'.
    
    Returns:
        dd.DataFrame: Cleaned claims data with relevant fields.
    """
    logger.info(f"Loading {file_type} claims from {file_path}")
    
    # Select relevant columns based on file type
    if file_type == 'carrier':
        cols = ['DESYNPUF_ID', 'CLM_ID', 'CLM_FROM_DT'] + \
               [f'ICD9_DGNS_CD_{i}' for i in range(1, 9)] + \
               [f'LINE_ICD9_DGNS_CD_{i}' for i in range(1, 14)] + \
               [f'HCPCS_CD_{i}' for i in range(1, 14)] + \
               [f'LINE_NCH_PMT_AMT_{i}' for i in range(1, 14)]
    else:
        cols = ['DESYNPUF_ID', 'CLM_ID', 'CLM_FROM_DT', 'CLM_PMT_AMT', 'ADMTNG_ICD9_DGNS_CD'] + \
               [f'ICD9_DGNS_CD_{i}' for i in range(1, 11)] + \
               [f'HCPCS_CD_{i}' for i in range(1, 46)]
    
    # Load with Dask and PyArrow
    df = dd.read_csv(file_path, usecols=cols, dtype=CLAIMS_DTYPES, blocksize=10e6, engine='pyarrow')
    
    # Explicitly cast DESYNPUF_ID to string
    df['DESYNPUF_ID'] = df['DESYNPUF_ID'].astype('string')
    
    # For carrier claims, sum LINE_NCH_PMT_AMT_1 to LINE_NCH_PMT_AMT_13 into CLM_PMT_AMT
    if file_type == 'carrier':
        line_pmt_cols = [f'LINE_NCH_PMT_AMT_{i}' for i in range(1, 14)]
        df[line_pmt_cols] = df[line_pmt_cols].fillna(0)
        df['CLM_PMT_AMT'] = df[line_pmt_cols].sum(axis=1).astype('float64')
        df = df.drop(columns=line_pmt_cols)
    
    # Clean CLM_FROM_DT
    df['CLM_FROM_DT'] = df['CLM_FROM_DT'].str.replace(r'\.0$', '', regex=True).str.strip()
    df['CLM_FROM_DT'] = dd.to_datetime(df['CLM_FROM_DT'], format='%Y%m%d', errors='coerce')
    df = df[~df['CLM_FROM_DT'].isna()].persist()
    
    # Clean ICD-9 and HCPCS codes
    icd9_cols = [col for col in df.columns if 'ICD9_DGNS_CD' in col or 'LINE_ICD9_DGNS_CD' in col]
    for col in icd9_cols:
        df[col] = df[col].str.strip().replace('', np.nan).fillna('').astype('string')
    hcpcs_cols = [col for col in df.columns if 'HCPCS_CD' in col]
    for col in hcpcs_cols:
        df[col] = df[col].str.strip().replace('', np.nan).fillna('').astype('string')
    
    # Filter for 2008-2009
    df = df[(df['CLM_FROM_DT'] >= '2008-12-02') & (df['CLM_FROM_DT'] <= '2009-12-31')]
    
    # Debug: Check nulls
    logger.info(f"{file_type}_df null counts: {df.isna().sum().compute()}")
    logger.info(f"Loaded {len(df)} {file_type} claims for 2008-2009")
    return df

def identify_septicemia_cases(inpatient_df: dd.DataFrame) -> dd.DataFrame:
    """
    Identify beneficiaries with septicemia in 2009 and their latest date.
    
    Args:
        inpatient_df (dd.DataFrame): Inpatient claims data.
    
    Returns:
        dd.DataFrame: Beneficiaries with septicemia and latest CLM_FROM_DT.
    """
    logger.info("Identifying septicemia cases in 2009")
    
    # Ensure DESYNPUF_ID is string
    inpatient_df['DESYNPUF_ID'] = inpatient_df['DESYNPUF_ID'].astype('string')
    
    # Filter for 2009 claims
    df = inpatient_df[inpatient_df['CLM_FROM_DT'].dt.year == 2009]
    
    # Identify septicemia (ICD-9 0389)
    icd9_cols = [f'ICD9_DGNS_CD_{i}' for i in range(1, 11)] + ['ADMTNG_ICD9_DGNS_CD']
    for i, col in enumerate(icd9_cols):
        df[f'is_septicemia_{i}'] = (df[col] == '0389').astype('int8')
    
    septicemia_cols = [f'is_septicemia_{i}' for i in range(len(icd9_cols))]
    df['septicemia_present'] = df[septicemia_cols].sum(axis=1).fillna(0).astype('int32')
    
    # Filter for claims with septicemia
    septicemia_df = df[df['septicemia_present'] > 0][['DESYNPUF_ID', 'CLM_FROM_DT']]
    
    # Drop temporary columns
    df = df.drop(columns=septicemia_cols + ['septicemia_present'])
    
    # Get latest septicemia date per beneficiary
    septicemia_cases = septicemia_df.groupby('DESYNPUF_ID')['CLM_FROM_DT'].max().reset_index()
    septicemia_cases = septicemia_cases.rename(columns={'CLM_FROM_DT': 'latest_septicemia_date'})
    
    # Ensure DESYNPUF_ID is string
    septicemia_cases['DESYNPUF_ID'] = septicemia_cases['DESYNPUF_ID'].astype('string')
    
    # Save intermediate result
    output_path = os.path.join(project_root, 'data/processed/septicemia_risk/septicemia_cases.csv')
    septicemia_cases.to_csv(output_path, single_file=True, index=False)
    logger.info(f"Identified {len(septicemia_cases)} septicemia cases, saved to {output_path}")
    
    return septicemia_cases

def assign_index_dates(septicemia_cases: dd.DataFrame, claims_df: dd.DataFrame,
                      beneficiary_df: dd.DataFrame, start_date: str = '2009-01-01',
                      end_date: str = '2009-12-25') -> dd.DataFrame:
    """
    Assign index dates: latest septicemia date or random 2009 claim date.
    
    Args:
        septicemia_cases (dd.DataFrame): Beneficiaries with septicemia.
        claims_df (dd.DataFrame): Combined claims data.
        beneficiary_df (dd.DataFrame): Beneficiary data.
        start_date (str): Start date for random index (default: 2009-01-01).
        end_date (str): End date for random index (default: 2009-12-25).
    
    Returns:
        dd.DataFrame: DESYNPUF_ID, septicemia_case_ind, index_date.
    """
    logger.info("Assigning index dates")
    
    # Ensure DESYNPUF_ID is string
    claims_df['DESYNPUF_ID'] = claims_df['DESYNPUF_ID'].astype('string')
    beneficiary_df['DESYNPUF_ID'] = beneficiary_df['DESYNPUF_ID'].astype('string')
    septicemia_cases['DESYNPUF_ID'] = septicemia_cases['DESYNPUF_ID'].astype('string')
    
    # Filter 2009 claims
    claims_2009 = claims_df[claims_df['CLM_FROM_DT'].dt.year == 2009][['DESYNPUF_ID', 'CLM_FROM_DT']]
    
    # Merge beneficiaries with claims to ensure coverage
    valid_beneficiaries = beneficiary_df[['DESYNPUF_ID']].merge(
        claims_2009[['DESYNPUF_ID']].drop_duplicates(), on='DESYNPUF_ID', how='inner'
    )
    
    # Assign latest septicemia date for septicemia cases
    septicemia_index = septicemia_cases.rename(columns={'latest_septicemia_date': 'index_date'})
    septicemia_index['septicemia_case_ind'] = 1
    
    # For non-septicemia beneficiaries, sample random 2009 claim date
    non_septicemia = valid_beneficiaries[~valid_beneficiaries['DESYNPUF_ID'].isin(septicemia_cases['DESYNPUF_ID'])]
    non_septicemia = non_septicemia.merge(claims_2009, on='DESYNPUF_ID')
    non_septicemia = non_septicemia[
        (non_septicemia['CLM_FROM_DT'] >= start_date) &
        (non_septicemia['CLM_FROM_DT'] <= end_date)
    ]
    
    # Randomly sample one date per beneficiary
    non_septicemia_index = non_septicemia.groupby('DESYNPUF_ID').apply(
        lambda x: x.sample(n=1, random_state=42)[['CLM_FROM_DT']],
        meta={'CLM_FROM_DT': 'datetime64[ns]'},
        include_groups=True
    ).reset_index().rename(columns={'index':'DESYNPUF_ID','CLM_FROM_DT': 'index_date'})[['DESYNPUF_ID', 'index_date']]
    non_septicemia_index['DESYNPUF_ID'] = non_septicemia_index['DESYNPUF_ID'].astype('string')
    non_septicemia_index['septicemia_case_ind'] = 0
    
    # Combine septicemia and non-septicemia index dates
    index_dates = dd.concat([septicemia_index, non_septicemia_index], axis=0)[['DESYNPUF_ID', 'septicemia_case_ind', 'index_date']]
    index_dates['DESYNPUF_ID'] = index_dates['DESYNPUF_ID'].astype('string')
    
    # Save intermediate result
    output_path = os.path.join(project_root, 'data/processed/septicemia_risk/index_dates.csv')
    index_dates.to_csv(output_path, single_file=True, index=False)
    logger.info(f"Assigned index dates for {len(index_dates)} beneficiaries, saved to {output_path}")
    
    return index_dates

def engineer_features(beneficiary_df: dd.DataFrame, claims_df: dd.DataFrame,
                     index_dates: dd.DataFrame, lookback_days: int = 30,
                     top_icd9_codes: list = [], top_hcpcs_codes: list = [],
                     interaction_pairs: list = []) -> dd.DataFrame:
    """
    Generate demographic, chronic condition, and claims-based features.
    
    Args:
        beneficiary_df (dd.DataFrame): Beneficiary data.
        claims_df (dd.DataFrame): Combined claims data.
        index_dates (dd.DataFrame): DESYNPUF_ID, septicemia_case_ind, index_date.
        lookback_days (int): Lookback period in days (default: 30).
        top_icd9_codes (list): Top ICD-9 codes from EDA.
        top_hcpcs_codes (list): Top HCPCS codes from EDA.
        interaction_pairs (list): Top ICD-9 code pairs from EDA.
    
    Returns:
        dd.DataFrame: Feature set with demographic, chronic, and claims-based features.
    """
    logger = setup_logger(log_dir=os.path.join(project_root, "outputs/logs"), log_file="engineer_features.log")
    logger.info("Engineering features")
    
    # Ensure DESYNPUF_ID is string
    beneficiary_df['DESYNPUF_ID'] = beneficiary_df['DESYNPUF_ID'].astype('string')
    claims_df['DESYNPUF_ID'] = claims_df['DESYNPUF_ID'].astype('string')
    index_dates['DESYNPUF_ID'] = index_dates['DESYNPUF_ID'].astype('string')
    
    # Log columns of input DataFrames
    logger.info(f"index_dates columns: {index_dates.columns.tolist()}")
    logger.info(f"beneficiary_df columns: {beneficiary_df.columns.tolist()}")
    logger.info(f"claims_df columns: {claims_df.columns.tolist()}")
    
    # Select only necessary columns from claims_df
    icd9_cols = [col for col in claims_df.columns if 'ICD9_DGNS_CD' in col or 'LINE_ICD9_DGNS_CD' in col]
    hcpcs_cols = [col for col in claims_df.columns if 'HCPCS_CD' in col]
    claims_cols = ['DESYNPUF_ID', 'CLM_ID', 'CLM_FROM_DT', 'CLM_PMT_AMT', 'file_type'] + icd9_cols + hcpcs_cols
    claims_df = claims_df[claims_cols].persist()
    logger.info(f"Reduced claims_df columns: {claims_df.columns.tolist()}")
    
    # Merge demographic and chronic condition features
    start_time = time.time()
    features_df = index_dates.merge(beneficiary_df, on='DESYNPUF_ID', how='left')
    for col in ['age', 'chronic_condition_count', 'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 
                'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 
                'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA']:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna(0).astype('int32')
    features_df = features_df.persist()
    logger.info(f"Demographic merge completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"features_df null counts after demographic merge: {features_df.isna().sum().compute()}")
    
    # Filter claims within 30-day lookback
    start_time = time.time()
    claims_window = index_dates.merge(
        claims_df, on='DESYNPUF_ID', how='left'
    )
    claims_window = claims_window[
        (claims_window['CLM_FROM_DT'] >= claims_window['index_date'] - timedelta(days=lookback_days)) &
        (claims_window['CLM_FROM_DT'] < claims_window['index_date'])
    ].persist()
    logger.info(f"claims_window filtering completed in {time.time() - start_time:.2f} seconds")
    logger.info(f"claims_window null counts: {claims_window.isna().sum().compute()}")
    
    # Aggregate claim counts and amounts by file type
    for file_type in ['inpatient', 'outpatient', 'carrier']:
        start_time = time.time()
        file_df = claims_window[claims_window['file_type'] == file_type]
        counts = file_df.groupby(['DESYNPUF_ID', 'septicemia_case_ind', 'index_date'])['CLM_ID'].count().reset_index()
        counts = counts.rename(columns={'CLM_ID': f'{file_type}_count_{lookback_days}d'})
        counts[f'{file_type}_count_{lookback_days}d'] = counts[f'{file_type}_count_{lookback_days}d'].fillna(0).astype('int32')
        counts['DESYNPUF_ID'] = counts['DESYNPUF_ID'].astype('string')
        counts = counts.persist()
        logger.info(f"{file_type}_count_{lookback_days}d null count: {counts[f'{file_type}_count_{lookback_days}d'].isna().sum().compute()}")
        amounts = file_df.groupby(['DESYNPUF_ID', 'septicemia_case_ind', 'index_date'])['CLM_PMT_AMT'].sum().reset_index()
        amounts = amounts.rename(columns={'CLM_PMT_AMT': f'{file_type}_amt_{lookback_days}d'})
        amounts[f'{file_type}_amt_{lookback_days}d'] = amounts[f'{file_type}_amt_{lookback_days}d'].fillna(0).astype('float64')
        amounts['DESYNPUF_ID'] = amounts['DESYNPUF_ID'].astype('string')
        amounts = amounts.persist()
        logger.info(f"{file_type}_amt_{lookback_days}d null count: {amounts[f'{file_type}_amt_{lookback_days}d'].isna().sum().compute()}")
        features_df = features_df.merge(counts, on=['DESYNPUF_ID', 'septicemia_case_ind', 'index_date'], how='left')
        features_df = features_df.merge(amounts, on=['DESYNPUF_ID', 'septicemia_case_ind', 'index_date'], how='left')
        features_df = features_df.persist()
        logger.info(f"{file_type} aggregation completed in {time.time() - start_time:.2f} seconds")
    
    # Add binary flags for top ICD-9 codes
    icd9_cols = [col for col in claims_window.columns if 'ICD9_DGNS_CD' in col or 'LINE_ICD9_DGNS_CD' in col]
    for code in top_icd9_codes:
        start_time = time.time()
        code_clean = code.replace('.', '')
        for col in icd9_cols:
            claims_window[col] = claims_window[col].fillna('').astype('string')
        for i, col in enumerate(icd9_cols):
            claims_window[f'icd9_{code_clean}_{i}'] = (claims_window[col] == code_clean).astype('int8')
        icd9_indicators = [f'icd9_{code_clean}_{i}' for i in range(len(icd9_cols))]
        claims_window[f'icd9_{code_clean}_sum'] = claims_window[icd9_indicators].sum(axis=1).fillna(0).astype('int32')
        claims_window = claims_window.persist()
        logger.info(f"icd9_{code_clean}_sum null count: {claims_window[f'icd9_{code_clean}_sum'].isna().sum().compute()}")
        icd9_flag = claims_window[claims_window[f'icd9_{code_clean}_sum'] > 0][['DESYNPUF_ID', 'septicemia_case_ind', 'index_date']].groupby(['DESYNPUF_ID', 'septicemia_case_ind', 'index_date']).size().reset_index().rename(columns={0: 'count'})
        icd9_flag['DESYNPUF_ID'] = icd9_flag['DESYNPUF_ID'].astype('string')
        icd9_flag[f'icd9_{code_clean}_{lookback_days}d'] = (icd9_flag['count'] > 0).astype('int8')
        icd9_flag = icd9_flag.persist()
        logger.info(f"icd9_{code_clean}_{lookback_days}d null count: {icd9_flag[f'icd9_{code_clean}_{lookback_days}d'].isna().sum().compute()}")
        features_df = features_df.merge(
            icd9_flag[['DESYNPUF_ID', 'septicemia_case_ind', 'index_date', f'icd9_{code_clean}_{lookback_days}d']],
            on=['DESYNPUF_ID', 'septicemia_case_ind', 'index_date'], how='left'
        )
        features_df[f'icd9_{code_clean}_{lookback_days}d'] = features_df[f'icd9_{code_clean}_{lookback_days}d'].fillna(0).astype('int8')
        features_df = features_df.persist()
        claims_window = claims_window.drop(columns=icd9_indicators + [f'icd9_{code_clean}_sum'])
        logger.info(f"ICD-9 {code_clean} processing completed in {time.time() - start_time:.2f} seconds")
    
    # Add interaction features for ICD-9 pairs
    for code1, code2 in interaction_pairs:
        start_time = time.time()
        code1_clean = code1.replace('icd9_', '').replace('_30d', '') if code1.startswith('icd9_') else code1
        code2_clean = code2.replace('icd9_', '').replace('_30d', '') if code2.startswith('icd9_') else code2
        feature_name = f'icd9_{code1_clean}_and_{code2_clean}_{lookback_days}d'
        
        # Use _sum columns from ICD-9 flags loop
        sum_col1 = f'icd9_{code1_clean}_sum'
        sum_col2 = f'icd9_{code2_clean}_sum'
        
        # Ensure sum columns exist
        if sum_col1 not in claims_window.columns or sum_col2 not in claims_window.columns:
            logger.warning(f"One or both of {sum_col1}, {sum_col2} not found in claims_window. Skipping {feature_name}")
            continue
        
        # Compute interaction: 1 if both codes are present, 0 otherwise
        claims_window[feature_name] = ((claims_window[sum_col1] > 0) & 
                                    (claims_window[sum_col2] > 0)).astype('int8')
        claims_window = claims_window.persist()
        
        # Aggregate to patient level
        interaction_flag = claims_window[claims_window[feature_name] > 0][
            ['DESYNPUF_ID', 'septicemia_case_ind', 'index_date']
        ].groupby(['DESYNPUF_ID', 'septicemia_case_ind', 'index_date']).size().reset_index().rename(columns={0: 'count'})
        interaction_flag['DESYNPUF_ID'] = interaction_flag['DESYNPUF_ID'].astype('string')
        interaction_flag[feature_name] = (interaction_flag['count'] > 0).astype('int8')
        interaction_flag = interaction_flag.persist()
        
        logger.info(f"{feature_name} null count: {interaction_flag[feature_name].isna().sum().compute()}")
        
        # Merge with features_df
        features_df = features_df.merge(
            interaction_flag[['DESYNPUF_ID', 'septicemia_case_ind', 'index_date', feature_name]],
            on=['DESYNPUF_ID', 'septicemia_case_ind', 'index_date'], how='left'
        )
        features_df[feature_name] = features_df[feature_name].fillna(0).astype('int8')
        features_df = features_df.persist()
        
        logger.info(f"Interaction {feature_name} processing completed in {time.time() - start_time:.2f} seconds")

    # Add binary flags for top HCPCS codes
    hcpcs_cols = [col for col in claims_window.columns if 'HCPCS_CD' in col]
    for code in top_hcpcs_codes:
        start_time = time.time()
        for col in hcpcs_cols:
            claims_window[col] = claims_window[col].fillna('').astype('string')
        for i, col in enumerate(hcpcs_cols):
            claims_window[f'hcpcs_{code}_{i}'] = (claims_window[col] == code).astype('int8')
        hcpcs_indicators = [f'hcpcs_{code}_{i}' for i in range(len(hcpcs_cols))]
        claims_window[f'hcpcs_{code}_sum'] = claims_window[hcpcs_indicators].sum(axis=1).fillna(0).astype('int32')
        claims_window = claims_window.persist()
        logger.info(f"hcpcs_{code}_sum null count: {claims_window[f'hcpcs_{code}_sum'].isna().sum().compute()}")
        hcpcs_flag = claims_window[claims_window[f'hcpcs_{code}_sum'] > 0][['DESYNPUF_ID', 'septicemia_case_ind', 'index_date']].groupby(['DESYNPUF_ID', 'septicemia_case_ind', 'index_date']).size().reset_index().rename(columns={0: 'count'})
        hcpcs_flag['DESYNPUF_ID'] = hcpcs_flag['DESYNPUF_ID'].astype('string')
        hcpcs_flag[f'hcpcs_{code}_{lookback_days}d'] = (hcpcs_flag['count'] > 0).astype('int8')
        hcpcs_flag = hcpcs_flag.persist()
        logger.info(f"hcpcs_{code}_{lookback_days}d null count: {hcpcs_flag[f'hcpcs_{code}_{lookback_days}d'].isna().sum().compute()}")
        features_df = features_df.merge(
            hcpcs_flag[['DESYNPUF_ID', 'septicemia_case_ind', 'index_date', f'hcpcs_{code}_{lookback_days}d']],
            on=['DESYNPUF_ID', 'septicemia_case_ind', 'index_date'], how='left'
        )
        features_df[f'hcpcs_{code}_{lookback_days}d'] = features_df[f'hcpcs_{code}_{lookback_days}d'].fillna(0).astype('int8')
        features_df = features_df.persist()
        claims_window = claims_window.drop(columns=hcpcs_indicators + [f'hcpcs_{code}_sum'])
        logger.info(f"HCPCS {code} processing completed in {time.time() - start_time:.2f} seconds")
    
    # Fill missing values for all numeric columns
    start_time = time.time()
    for col in features_df.columns:
        if '_count_' in col:
            features_df[col] = features_df[col].fillna(0).astype('int32')
        if '_amt_' in col:
            features_df[col] = features_df[col].fillna(0).astype('float64')
        if 'icd9_' in col or 'hcpcs_' in col or col in [
            'age', 'chronic_condition_count', 'SP_ALZHDMTA', 'SP_CHF', 'SP_CHRNKIDN', 
            'SP_CNCR', 'SP_COPD', 'SP_DEPRESSN', 'SP_DIABETES', 'SP_ISCHMCHT', 
            'SP_OSTEOPRS', 'SP_RA_OA', 'SP_STRKETIA'
        ]:
            features_df[col] = features_df[col].fillna(0).astype('int32')
    
    # Persist features_df
    features_df = features_df.persist()
    logger.info(f"Final fill and persist completed in {time.time() - start_time:.2f} seconds")
    
    # Debug: Log column dtypes and check for nulls
    logger.info("Column dtypes before final persist:")
    logger.info(features_df.dtypes.to_string())
    null_counts = features_df.isna().sum().compute()
    logger.info("Null counts before final persist:")
    logger.info(null_counts.to_string())
    
    # Persist features_df before computing length
    features_df = features_df.persist()
    logger.info(f"Generated features for {len(features_df)} beneficiaries")
    return features_df

def save_dataset(features_df: dd.DataFrame, output_path: str) -> None:
    """
    Save features dataset with septicemia_case_ind as outcome.
    
    Args:
        features_df (dd.DataFrame): Feature set with DESYNPUF_ID, septicemia_case_ind, index_date, and features.
        output_path (str): Path to save dataset CSV.
    """
    logger.info(f"Saving dataset to {output_path}")
    
    # Ensure septicemia_case_ind is included and cast to int8
    features_df['septicemia_case_ind'] = features_df['septicemia_case_ind'].astype('int8')
    
    # Save to CSV
    features_df.to_csv(output_path, single_file=True, index=False)
    
    # Log statistics
    positive_count = features_df['septicemia_case_ind'].sum().compute()
    logger.info(f"Saved dataset with {len(features_df)} rows, {positive_count} positive outcomes")

def preprocess_pipeline(beneficiary_file: str, inpatient_file: str, outpatient_file: str,
                      carrier_file: str, top_icd9_codes: list = [], top_hcpcs_codes: list = [],
                      interaction_pairs: list = [],
                      lookback_days: int = 30, output_path: str = 'data/processed/septicemia_risk/dataset.csv') -> None:
    """
    Orchestrate preprocessing pipeline for septicemia risk prediction.
    
    Args:
        beneficiary_file (str): Path to beneficiary CSV.
        inpatient_file (str): Path to inpatient claims CSV.
        outpatient_file (str): Path to outpatient claims CSV.
        carrier_file (str): Path to carrier claims CSV.
        top_icd9_codes (list): Top ICD-9 codes from EDA.
        top_hcpcs_codes (list): Top HCPCS codes from EDA.
        interaction_pairs (list): Top ICD-9 code pairs from EDA.
        lookback_days (int): Lookback period in days (default: 30).
        output_path (str): Path to save dataset CSV.
    """
    logger.info("Starting preprocessing pipeline")
    
    # Load data
    beneficiary_df = load_beneficiary(beneficiary_file)
    inpatient_df = load_claims(inpatient_file, 'inpatient')
    outpatient_df = load_claims(outpatient_file, 'outpatient')
    carrier_df = load_claims(carrier_file, 'carrier')
    
    # Combine claims with file_type column
    inpatient_df['file_type'] = 'inpatient'
    outpatient_df['file_type'] = 'outpatient'
    carrier_df['file_type'] = 'carrier'
    claims_df = dd.concat([inpatient_df, outpatient_df, carrier_df], axis=0)
    logger.info(f"claims_df null counts: {claims_df.isna().sum().compute()}")
    
    # Identify septicemia cases
    septicemia_cases = identify_septicemia_cases(inpatient_df)
    
    # Assign index dates
    index_dates = assign_index_dates(septicemia_cases, claims_df, beneficiary_df)
    
    # Engineer features
    features_df = engineer_features(beneficiary_df, claims_df, index_dates, lookback_days, top_icd9_codes, top_hcpcs_codes)
    
    # Save dataset
    save_dataset(features_df, output_path)
    
    logger.info("Preprocessing pipeline completed")

if __name__ == "__main__":
    # Example usage with absolute paths
    preprocess_pipeline(
        beneficiary_file=os.path.join(project_root, 'data/raw/DE1_0_2009_Beneficiary_Summary_File_Sample_1.csv'),
        inpatient_file=os.path.join(project_root, 'data/raw/DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.csv'),
        outpatient_file=os.path.join(project_root, 'data/raw/DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.csv'),
        carrier_file=os.path.join(project_root, 'data/raw/DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.csv'),
        top_icd9_codes=['99673',
 'V053',
 '585',
 '5070',
 'V560',
 'V561',
 '2762',
 '58881',
 '40391',
 '2639',
 '0389',
 '28521',
 '2851',
 '51881',
 '79902',
 '5849',
 '78009',
 '78097',
 '40390',
 '25040',
 'V420',
 '5856',
 '72981',
 '2809',
 '43491',
 '49121',
 '2948',
 '5854',
 '3051',
 '78791',
 '78701',
 '486',
 '27651',
 '2761',
 '5119'],
        top_hcpcs_codes=['90945',
 'J2916',
 'J0882',
 'A4913',
 'A4657',
 'Q4081',
 'J1270',
 'J2501',
 '90999',
 'J1756',
 '80162',
 '93307',
 '82310',
 '99291',
 'J2405',
 'A0425',
 '83970',
 '98941',
 '92135'],
        interaction_pairs=[('99673', '78009'),
 ('99673', '43491'),
 ('99673', '2948'),
 ('V053', '40390'),
 ('V053', '486'),
 ('585', '2762'),
 ('V560', '2762'),
 ('V560', '3051'),
 ('2762', '40390'),
 ('2762', '43491'),
 ('40391', '78009'),
 ('2639', '3051'),
 ('78009', '72981'),
 ('25040', '72981'),
 ('58881', '2809'),
 ('58881', '28521'),
 ('58881', '5856'),
 ('5856', '2809'),
 ('V560', '2809'),
 ('28521', '2809'),
 ('585', '28521'),
 ('58881', '5849'),
 ('99673', '58881'),
 ('585', '2809'),
 ('2762', '5849'),
 ('99673', '2809'),
 ('585', '58881'),
 ('99673', '5856'),
 ('V560', '58881'),
 ('51881', '5119'),
 ('585', '5856'),
 ('28521', '51881'),
 ('28521', '5856'),
 ('40391', '5856')],
        output_path=os.path.join(project_root, 'data/processed/septicemia_risk/dataset.csv')
    )
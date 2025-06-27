# tests/create_test_data.py
import pandas as pd
import os

os.makedirs("tests/sample_data", exist_ok=True)
data = {
    "DESYNPUF_ID": ["ID1", "ID2"],
    "CLM_ID": ["CLM1", "CLM2"],
    "CLM_FROM_DT": ["2008-01-01", "2008-01-02"],
    "CLM_PMT_AMT": [100.0, 200.0],
    "ICD9_DGNS_CD_1": ["250.00", "401.9"]
}
df = pd.DataFrame(data)
df.to_csv("tests/sample_data/test_inpatient.csv", index=False)
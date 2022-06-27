### MYSWaningImmunity
### From CasesDeaths --- ICU and Deaths
### Used in assessing severe outcomes
### SOURCE FILE: 02dec2021 VINTAGE
### Additional columns: dupes, trace (pre-Sep), test (pre-Sep), frontliners
### FOR PUBLIC RELEASE (ANONYMITY CHECKS)

import pandas as pd
import numpy as np
from datetime import date, timedelta
import statsmodels.discrete.discrete_model as sm
import plotly.graph_objects as go
from patsy import dmatrices
import itertools
import time
import telegram_send
import dataframe_image as dfi

time_start = time.time()

### 0 --- Preliminaries
## File paths
path_input = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/0_Data/2021-12-02_WaningVE_Freeze/2022-03-07_RR/'
path_wi = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/MYSWaningImmunity/2022-03_RR/' # RR folder
## File name
file_input = 'CasesDeaths_WaningImmunity_Data_RR.csv'
## Data frames
list_col_dates = ['date_lab', 'date_death',
                  'date1', 'date2'] # taken from CasesDeaths_Data
df = pd.read_csv(path_input + file_input, parse_dates=list_col_dates)
for i in list_col_dates:
    df[i] = df[i].dt.date # %Y-%m-%d format

### I --- Double check
## Check missing
for i in df.columns: print(i + ': ' + str(df.loc[df[i].isna(), i].count()) + ' missing')
print('\n\n')
for i in df.columns: print(i + ': ' + str(df.loc[df[i].isin([np.inf, -np.inf]), i].count()) + ' inf')
## Delete columns never used in analysis (intended for other side projects)
col_notused = ['dead_gap', 'infdead_gap']
for i in col_notused: del df[i]
## Rename duplicate as 'reinf' (these are repeat entries at different dates in the cases line listing, hence assumed to be reinfections)
df = df.rename(columns={'duplicate':'reinf'})
## Fill NA
col_na_str0 = ['date1', 'date2' , 'date_death', 'type1', 'type2']
for i in col_na_str0: df.loc[df[i].isna(), i] = '0'
for i in df.columns: df.loc[df[i].isna(), i] = 0
## Reset index
df = df.reset_index(drop=True)
## Impose data type
dict_df_dtype = {'date1': 'str',
                 'date2': 'str',
                 'date_lab': 'str',
                 'date_death': 'str',
                 'dead': 'int',
                 'symptomatic': 'int',
                 'icu': 'int',
                 'comorb': 'int',
                 'age': 'int',
                 'malaysian': 'int',
                 'male': 'int',
                 'state_id': 'int',
                 'type1': 'str',
                 'type2': 'str',
                 'type_comb': 'str',
                 'vax_partial': 'int',
                 'vax_full': 'int',
                 'test_count': 'float',
                 'trace_count': 'float',
                 'frontliner': 'float',
                 'reinf': 'int'}
df = df.astype(dict_df_dtype)

### II --- Export as parquet (save space)
df.to_parquet(path_input + 'CasesDeaths_WaningImmunity_Data_RR_ForRelease.parquet', compression='brotli')

### End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
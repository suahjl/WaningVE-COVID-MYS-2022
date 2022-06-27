### MYSWaningImmunity
### From CasesDeaths --- ICU and Deaths
### Used in assessing severe outcomes
### SOURCE FILE: 02dec2021 VINTAGE
### Additional columns: dupes, trace (pre-Sep), test (pre-Sep), frontliners

import pandas as pd
import numpy as np
import datetime
from datetime import date, timedelta
import time
import itertools
import telegram_send
import shutil

import wrangle_simka as ws
import wrangle_trace as wt

time_start = time.time()

### 0 --- Preliminaries
## File paths
path_input = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/0_Data/2021-12-02_WaningVE_Freeze/2021-12-02/' # frozen path
path_output = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/0_Data/2021-12-02_WaningVE_Freeze/2022-03-07_RR/' # frozen path
path_trace = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/0_Data/Trace/'
path_simka = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/0_Data/SIMKA/'
path_fl = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/0_Data/HighRisk/'

## File names
file_cd = 'CasesDeaths_Data.csv' # Fully cleaned cases + deaths + icu (status only)
file_dupe = 'CasesDeaths_Data_FULL_WithDupes.csv'
file_trace = 'trace_proc.parquet'
file_simka = 'simka_proc.parquet'
file_fl = 'frontliner_consol.csv'

## Column types
dict_dtype = {'no_case': 'str',
              'no_death': 'str',
              'id': 'str',
              'date_lab': 'str',
              'date_death': 'str',
              'date1': 'str',
              'date2': 'str',
              'date3': 'str',
              'type1': 'str',
              'type2': 'str',
              'type3': 'str',
              'type_comb': 'str',
              'vax_bin': 'int64',
              'vax_partial': 'int64',
              'vax_full': 'int64',
              'vax_boost': 'int64',
              'state_id': 'int64',
              'age': 'float64',
              'agegroup': 'float64',
              'male': 'int64',
              'malaysian': 'int64',
              'comorb': 'int64',
              'import': 'int64',
              'cluster': 'int64',
              'symptomatic': 'int64',
              'dead': 'int64',
              'icu': 'int64'}

## Basic data frames
# Original frozen source file (02dec2021 vintage)
list_col_dates = ['date_lab', 'date_death', 'date1', 'date2', 'date3']
df = pd.read_csv(path_input + file_cd, parse_dates=list_col_dates, dtype=dict_dtype) # frozen path
for i in list_col_dates:
    df[i] = df[i].dt.date
# Latest with duplicates (RR)
df_dupe = pd.read_csv(path_output + file_dupe,
                      usecols=['no_case', 'duplicate'],
                      dtype={'no_case':'str', 'duplicate': 'int'})
# Trace (RR)
df_trace = pd.read_parquet(path_trace + file_trace)
df_trace['date'] = pd.to_datetime(df_trace['date']).dt.date
# SIMKA (RR)
df_simka = pd.read_parquet(path_simka + file_simka)
df_simka['date'] = pd.to_datetime(df_simka['date']).dt.date
# Frontliner (RR)
df_fl = pd.read_csv(path_fl + file_fl, dtype={'id':'str', 'frontliner': 'int'})

## Pare down data frame
list_cd_keep_pre = ['no_case', 'id', 'date_lab', 'date_death',
                    'date1', 'date2', 'date3',
                    'type1', 'type2', 'type3', 'type_comb',
                    'vax_bin', 'vax_partial', 'vax_full', 'vax_boost',
                    'state_id', 'age', 'male', 'malaysian',
                    'comorb', 'symptomatic', 'dead', 'icu'] # keep case number & IDs to merge with duplicates later
df = df[list_cd_keep_pre]

## Count
count_full = df['date_lab'].count().astype('int64')

### I --- Cleaning
## Wrangling Trace (RR)
df_trace = wt.timebound(df_trace,
                        lowerbound=date(2020,1,1),
                        upperbound=date(2021,8,31))
df_trace = wt.tracecount(df_trace)
## Wrangling SIMKA (RR)
df_simka = ws.timebound(df_simka,
                        lowerbound=date(2020,1,1),
                        upperbound=date(2021,8,31))
df_simka = ws.testcount(df_simka)

## Trim: Drop partially vaccinated or boosted at the point of infection
df = df[~((df['vax_partial'] == 1) | (df['vax_boost'] == 1))]
count_keep_full_unvax = df['date_lab'].count().astype('int64')

## Trim: Keep only individuals vaccinated with SS and PP
df = df[(df['type_comb'] == 'ss') | (df['type_comb'] == 'pp') | (df['type_comb'] == '0')] # keep vax types of interest
count_keep_ss_pp_0 = df['date_lab'].count().astype('int64')

## Trim: Infected before full vax
df.loc[df['date_lab'] - df['date2'] < timedelta(days = 14), 'drop'] = 1 # Infected before being fully vaccinated
df = df[~(df['drop'] == 1)]
del df['drop']
count_drop_pred2 = df['date_lab'].count().astype('int64')

## Redefine: Symptomatic
df.loc[(df['symptomatic'] == 1) &
       (df['date_lab'] - df['date2'] < timedelta(days = 14)), 'symptomatic'] = 0 # Infected before being fully vaccinated

## Redefine: ICU
df.loc[(df['icu'] == 1) &
       (df['date_lab'] - df['date2'] < timedelta(days = 14)), 'icu'] = 0 # Infected before being fully vaccinated

## Redefine: Death
df.loc[(df['dead'] == 1) &
       (df['date_lab'] - df['date2'] < timedelta(days = 14)), 'dead'] = 0 # Infected before being fully vaccinated

## New: Survival time
df['dead_gap'] = (df['date_death'] - df['date2'] - timedelta(days=14)).dt.days
df['infdead_gap'] = (df['date_death'] - df['date_lab']).dt.days

## Merge (for RR)
print('CD <<--- Dupes <<--- Pre-Trace <<--- Pre-SIMKA <<--- Frontliner')
print('CD <<--- Dupes')
df = df.merge(df_dupe, on='no_case', how='left')
print(pd.DataFrame(df['duplicate'].value_counts()))
print('CD <<--- Dupes <<--- Pre-Trace')
df = df.merge(df_trace, on='id', how='left')
print('CD <<--- Dupes <<--- Pre-Trace <<--- Pre-SIMKA')
df = df.merge(df_simka, on='id', how='left')
print('CD <<--- Dupes <<--- Pre-Trace <<--- Pre-SIMKA <<--- Frontliner')
df = df.merge(df_fl, on='id', how='left')

## New (for RR): Dealing with NAs
list_missing0 = ['test_count', 'trace_count', 'frontliner']
for i in list_missing0:
    df.loc[df[i].isna(), i] = 0 # if no test or trace in baseline, then assume 0

## Pare down columns
list_col_keep = ['dead_gap', 'infdead_gap',
                 'date1', 'date2',
                 'date_lab', 'date_death',
                 'dead', 'symptomatic', 'icu',
                 'comorb', 'age', 'malaysian', 'male', 'state_id',
                 'type1', 'type2', 'type_comb',
                 'vax_partial', 'vax_full',
                 'test_count', 'trace_count',
                 'frontliner',
                 'duplicate']
df = df[list_col_keep]

## II --- Export
df.to_csv(path_output + 'CasesDeaths_WaningImmunity_Data_RR.csv', index=False) # For RR
telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                   messages=['CasesDeaths_WaningImmunity_Data_RR: Completed'])
telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
                   messages=['Outcomes line listing: ' + str(count_full) + '\n' +
                             'Keep fully vaccinated and unvaccinated: ' + str(count_keep_full_unvax) + '\n' +
                             'Keep vax of interest (SS, PP, 0): ' + str(count_keep_ss_pp_0) + '\n' +
                             'Drop Pre-Full Vax Infections: ' + str(count_drop_pred2)])

### End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
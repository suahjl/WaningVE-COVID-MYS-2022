### MYSWaningImmunity
### Incidence rates for data used in Est4d
### Cleaning steps harmonised with Est4d


import pandas as pd
import numpy as np
from datetime import date, timedelta
import itertools
import time
import telegram_send
import dataframe_image as dfi

from scipy import stats

time_start = time.time()

### 0 --- Preliminaries
## File paths
path_input = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/0_Data/2021-12-02_WaningVE_Freeze/2022-03-07_RR/'
path_wi = 'D:/Users/ECSUAH/Desktop/Quant/HealthEconomy/VaccinesAssessment/MYSWaningImmunity/2022-03_RR/' # RR folder
## File name
file_input = 'CasesDeaths_WaningImmunity_Data_RR_ForRelease.parquet'
## Data frames
list_col_dates = ['date_lab', 'date_death',
                  'date1', 'date2'] # taken from CasesDeaths_Data
df = pd.read_parquet(path_input + file_input)
for i in list_col_dates:
    df.loc[df[i] == '0', i] = np.nan  # supposedly missing
    df[i] = pd.to_datetime(df[i]).dt.date # %Y-%m-%d format
count_wi_full = df['date_lab'].count().astype('int64')

### 0 --- Cleaning
## Trim (RR): Reinfections / duplicates
print('Reinfections / duplicates')
print(pd.DataFrame(df['reinf'].value_counts())) # check
df = df[df['reinf'] == 0]
count_trim_dupes = df['date_lab'].count().astype('int64') # only first infections

## Trim: Fully vaccinated within vax_cutoff
vax_cutoff_lb = date(2021,4,1)
vax_cutoff_ub = date(2021,8,31)
option_14days = 1 # 14 days post-dose 2 (correct definition)
# option_14days = 0
if option_14days == 1:
    print('Full vax = 14 days post-dose 2')
    # telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
    #                    messages=['Full vax = 14 days post-dose 2'])
    df = df.loc[((df['date2'] + timedelta(days=14) >= vax_cutoff_lb) |
                (df['date2'].isna())), :]
    df = df.loc[((df['date2'] + timedelta(days=14) <= vax_cutoff_ub) |
                (df['date2'].isna())), :] # keep if vaxed within T1, or unvaxed
else:
    print('Full vax = immediately post-dose 2')
    # telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
    #                    messages=['Full vax = immediately post-dose 2'])
    df = df.loc[((df['date2'] >= vax_cutoff_lb) |
                 (df['date2'].isna())), :]
    df = df.loc[((df['date2'] <= vax_cutoff_ub) |
                 (df['date2'].isna())), :]  # keep if vaxed within T1, or unvaxed
count_trim_vaxdates = df['date_lab'].count().astype('int64')

## Trim: Tested positive within lab_cutoff
lab_cutoff_lb = vax_cutoff_ub + timedelta(days=1)
lab_cutoff_ub = date(2021,9,30)
df = df[((df['date_lab'] >= lab_cutoff_lb) &
        (df['date_lab'] <= lab_cutoff_ub)) |
        (df['date_lab'].isna())] # no positive test, or tested positive within lab_cutoff
count_trim_datelab = df['date_lab'].count().astype('int64')

## Trim: Drop partially vaccinated/ boosted when infected (already done in CD_WI)
df = df[~((df['vax_partial'] == 1))]
count_trim_partvax = df['date_lab'].count().astype('int64')

## Option: Check (drop specific age groups)
age_cutoff = 18
df = df[df['age'] >= age_cutoff]
count_trim_age = df['date_lab'].count().astype('int64')

## New: Severe outcomes
df['severe'] = 0
df.loc[((df['dead'] == 1) | (df['icu'] == 1)), 'severe'] = 1

### I --- Replicate key variables in estimation
## Option: reduce M groups
df['month'] = pd.to_datetime(df['date2'] + timedelta(days=14)).dt.month # month of full vax status
# option_mfull = 1
option_mfull = 0
if option_mfull == 1:
    m_lb = pd.to_datetime(vax_cutoff_lb).month
    m_ub = pd.to_datetime(vax_cutoff_ub).month
    df['month'] = df['month'] - m_lb + 1 # reindex to start from 1
    m_start = 1
    m_end = df['month'].max().astype('int64')
elif option_mfull == 0:
    m_lb = pd.to_datetime(vax_cutoff_lb).month
    m_ub = pd.to_datetime(vax_cutoff_ub).month
    m_list = [m_lb, m_ub]
    m_threshold = sum(m_list)/len(m_list)
    df.loc[df['month'] <= m_threshold, 'month_redux'] = 1
    df.loc[df['month'] > m_threshold, 'month_redux'] = 2
    m_start = 1
    m_end = 2
    df['month'] = df['month_redux']
    del df['month_redux']

## NEW (ONLY FOR SUM STATS): AGE GROUPS
df.loc[((df['age'] >= 18) & (df['age'] < 40)), 'agegroup'] = 1
df.loc[((df['age'] >= 40) & (df['age'] < 60)), 'agegroup'] = 2
df.loc[(df['age'] >= 60), 'agegroup'] = 3
df['agegroup'] = df['agegroup'].fillna(-1) # double check, should be nil

## VaxType-Timing group indicators
df['type_timing'] = df['type_comb'] + df['month'].astype('str') # type-timing; unvax as omitted group
df['type_timing'] = df['type_timing'].str.replace('.0', '', regex=True) # remove trailing .0
df.loc[df['type_timing'] == '0nan', 'type_timing'] = '0' # if unvax

### II --- Tabulating incidence rates (to be displayed together with VE)
df['infection'] = 1 # for ease of tabulation
## Main
inc_consol = pd.DataFrame(columns=['Outcome', 'Total', 'Event', 'Rate'])
# ICU
inc = pd.DataFrame(columns=['Outcome', 'Total', 'Event', 'Rate'])
inc['Total'] = df.groupby('type_timing')['infection'].value_counts()
inc['Event'] = df.groupby('type_timing')['icu'].value_counts()
inc['Rate'] = 1000*inc['Event']/inc['Total']
inc['Outcome'] = 'ICU'
inc_consol = pd.concat([inc_consol, inc], axis=0)
# Death
del inc
inc = pd.DataFrame(columns=['Outcome', 'Total', 'Event', 'Rate'])
inc['Total'] = df.groupby('type_timing')['infection'].value_counts()
inc['Event'] = df.groupby('type_timing')['dead'].value_counts()
inc['Rate'] = 1000*inc['Event']/inc['Total']
inc['Outcome'] = 'Death'
inc_consol = pd.concat([inc_consol, inc], axis=0)
# Severe
del inc
inc = pd.DataFrame(columns=['Outcome', 'Total', 'Event', 'Rate'])
inc['Total'] = df.groupby('type_timing')['infection'].value_counts()
inc['Event'] = df.groupby('type_timing')['severe'].value_counts()
inc['Rate'] = 1000*inc['Event']/inc['Total']
inc['Outcome'] = 'Severe'
inc_consol = pd.concat([inc_consol, inc], axis=0)
# Export
dfi.export(inc_consol, path_wi + 'MYWaningImmunity_Est4d_IncidenceRates_Main.png')
inc_consol.to_csv(path_wi + 'MYWaningImmunity_Est4d_IncidenceRates_Main.csv', index=True)
# with open(path_wi + 'MYWaningImmunity_Est4d_IncidenceRates_Main.png', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        images=[f],
#                        captions=['MYWaningImmunity_Est4d_IncidenceRates_Main'])
# with open(path_wi + 'MYWaningImmunity_Est4d_IncidenceRates_Main.csv', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        files=[f],
#                        captions=['MYWaningImmunity_Est4d_IncidenceRates_Main'])

## AgeStrat
del inc_consol
inc_consol = pd.DataFrame(columns=['Outcome', 'Total', 'Event', 'Rate'])
# ICU
inc = pd.DataFrame(columns=['Outcome', 'Total', 'Event', 'Rate'])
list_group = ['agegroup', 'type_timing']
inc['Total'] = df.groupby(list_group)['infection'].value_counts()
inc['Event'] = df.groupby(list_group)['icu'].value_counts()
inc['Rate'] = 1000*inc['Event']/inc['Total']
inc['Outcome'] = 'ICU'
inc_consol = pd.concat([inc_consol, inc], axis=0)
# Death
del inc
inc = pd.DataFrame(columns=['Outcome', 'Total', 'Event', 'Rate'])
inc['Total'] = df.groupby(list_group)['infection'].value_counts()
inc['Event'] = df.groupby(list_group)['dead'].value_counts()
inc['Rate'] = 1000*inc['Event']/inc['Total']
inc['Outcome'] = 'Death'
inc_consol = pd.concat([inc_consol, inc], axis=0)
# Severe
del inc
inc = pd.DataFrame(columns=['Outcome', 'Total', 'Event', 'Rate'])
inc['Total'] = df.groupby(list_group)['infection'].value_counts()
inc['Event'] = df.groupby(list_group)['severe'].value_counts()
inc['Rate'] = 1000*inc['Event']/inc['Total']
inc['Outcome'] = 'Severe'
inc_consol = pd.concat([inc_consol, inc], axis=0)
# Export
dfi.export(inc_consol, path_wi + 'MYWaningImmunity_Est4d_IncidenceRates_AgeStrat.png')
inc_consol.to_csv(path_wi + 'MYWaningImmunity_Est4d_IncidenceRates_AgeStrat.csv', index=True)
# with open(path_wi + 'MYWaningImmunity_Est4d_IncidenceRates_AgeStrat.png', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        images=[f],
#                        captions=['MYWaningImmunity_Est4d_IncidenceRates_AgeStrat'])
# with open(path_wi + 'MYWaningImmunity_Est4d_IncidenceRates_AgeStrat.csv', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        files=[f],
#                        captions=['MYWaningImmunity_Est4d_IncidenceRates_AgeStrat'])
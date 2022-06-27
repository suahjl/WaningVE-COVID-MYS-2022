### MYSWaningImmunity
### Summary stats for data used in Est4d
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
path_input = ''
path_wi = '' # RR folder
## File name
file_input = 'https://github.com/suahjl/WaningVE-COVID-MYS-2022/raw/main/Data/CasesDeaths_WaningImmunity_Data_RR_ForRelease.parquet'
## Data frames
list_col_dates = ['date_lab', 'date_death',
                  'date1', 'date2'] # taken from CasesDeaths_Data
df = pd.read_parquet(path_input + file_input)
for i in list_col_dates:
    df.loc[df[i] == '0', i] = np.nan  # supposedly missing
    df[i] = pd.to_datetime(df[i]).dt.date # %Y-%m-%d format
count_wi_full = df['date_lab'].count().astype('int64')

### 0 --- Cleaning
## Temporary (for diagnostics)
# print('FOR DIAGNOSTICS')
# df = df[df['age'] >= 60]

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
df = df[~(df['vax_partial'] == 1)]
count_trim_partvax = df['date_lab'].count().astype('int64')

## Option: Check (drop specific age groups)
age_cutoff = 18
df = df[df['age'] >= age_cutoff]
count_trim_age = df['date_lab'].count().astype('int64')

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
## New (RR): Trace count
df.loc[(df['trace_count'] == 0), 'tracegroup'] = 0
df.loc[(df['trace_count'] == 1), 'tracegroup'] = 1
df.loc[((df['trace_count'] >= 2) & (df['trace_count'] < 5)), 'tracegroup'] = 2
df.loc[((df['trace_count'] >= 5) & (df['trace_count'] < 10)), 'tracegroup'] = 3
df.loc[(df['trace_count'] >= 10), 'tracegroup'] = 4
##New (RR): Test count
df.loc[(df['test_count'] == 0), 'testgroup'] = 0
df.loc[(df['test_count'] == 1), 'testgroup'] = 1
df.loc[((df['test_count'] >= 2) & (df['test_count'] < 5)), 'testgroup'] = 2
df.loc[((df['test_count'] >= 5) & (df['test_count'] < 10)), 'testgroup'] = 3
df.loc[(df['test_count'] >= 10), 'testgroup'] = 4

## VaxType-Timing group indicators
df['type_timing'] = df['type_comb'] + df['month'].astype('str') # type-timing; unvax as omitted group
df['type_timing'] = df['type_timing'].str.replace('.0', '', regex=True) # remove trailing .0
df.loc[df['type_timing'] == '0nan', 'type_timing'] = '0' # if unvax

### II --- Tabulating summary stats and test of independence
## Basic dataframe
tab_consol = pd.DataFrame(columns=['0', 'pp2', 'pp1', 'ss2', 'ss1'])
perc_consol = pd.DataFrame(columns=['0', 'pp2', 'pp1', 'ss2', 'ss1']) # normalise on index (within vax-timing breakdown)
test_consol = pd.DataFrame(columns=['Variable', 'ChiSq', 'PVal', 'DFreedom'])
## Type-Timing
tab = pd.crosstab(df['type_timing'], df['type_timing'])
perc = pd.crosstab(df['type_timing'], df['type_timing'], normalize='columns')
perc = 100 * perc
test = pd.DataFrame(columns=['Variable', 'ChiSq', 'PVal', 'DFreedom'], index=[0])
# chisq = stats.chi2_contingency(tab) # before title row is added to 'tab'
test['Variable'] = 'type_timing'
test['ChiSq'] = 0
test['PVal'] = 0
test['DFreedom'] = 0
perc = perc[['0', 'pp2', 'pp1', 'ss2', 'ss1']]
tab = tab[['0', 'pp2', 'pp1', 'ss2', 'ss1']]
print(tab)
print(perc)
tab_consol = pd.concat([tab_consol, tab], axis=0)
perc_consol = pd.concat([perc_consol, perc], axis=0)
test_consol = pd.concat([test_consol, test], axis=0)
## Demographics
list_X = ['icu', 'dead'] + ['agegroup', 'malaysian', 'male', 'comorb', 'frontliner', 'tracegroup', 'testgroup'] + ['state_id']
for i in list_X:
    tab = pd.crosstab(df[i], df['type_timing'])
    perc = pd.crosstab(df[i], df['type_timing'], normalize='columns')
    perc = 100 * perc
    test = pd.DataFrame(columns=['Variable', 'ChiSq', 'PVal', 'DFreedom'], index=[0])
    chisq = stats.chi2_contingency(tab) # before title row is added to 'tab'
    test['Variable'] = i
    test['ChiSq'] = chisq[0]
    test['PVal'] = chisq[1]
    test['DFreedom'] = chisq[2]
    tab = tab[['0', 'pp2', 'pp1', 'ss2', 'ss1']]
    perc = perc[['0', 'pp2', 'pp1', 'ss2', 'ss1']]
    tab.loc[-1] = i
    tab.sort_index(inplace=True) # keep original index values (encoding)
    perc.loc[-1] = i
    perc.sort_index(inplace=True)  # keep original index values (encoding)
    print(tab)
    print(perc)
    tab_consol = pd.concat([tab_consol, tab], axis=0)
    perc_consol = pd.concat([perc_consol, perc], axis=0)
    test_consol = pd.concat([test_consol, test], axis=0)
## Export consolidated
check1 = tab_consol.reset_index(drop=False)
check2 = perc_consol.reset_index(drop=False)
dfi.export(tab_consol, path_wi + 'MYSWaningImmunity_Est4d_SumStats.png')
# with open(path_wi + 'MYSWaningImmunity_Est4d_SumStats.png', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        images=[f],
#                        captions=['MYSWaningImmunity_Est4d_SumStats'])
tab_consol.to_csv(path_wi + 'MYSWaningImmunity_Est4d_SumStats.csv', index=True)
# with open(path_wi + 'MYSWaningImmunity_Est4d_SumStats.csv', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        files=[f],
#                        captions=['MYSWaningImmunity_Est4d_SumStats'])

dfi.export(perc_consol, path_wi + 'MYSWaningImmunity_Est4d_SumStats_Perc.png')
# with open(path_wi + 'MYSWaningImmunity_Est4d_SumStats_Perc.png', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        images=[f],
#                        captions=['MYSWaningImmunity_Est4d_SumStats_Perc'])
perc_consol.to_csv(path_wi + 'MYSWaningImmunity_Est4d_SumStats_Perc.csv', index=True)
# with open(path_wi + 'MYSWaningImmunity_Est4d_SumStats_Perc.csv', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        files=[f],
#                        captions=['MYSWaningImmunity_Est4d_SumStats_Perc'])

dfi.export(test_consol, path_wi + 'MYSWaningImmunity_Est4d_SumStats_Test.png')
# with open(path_wi + 'MYSWaningImmunity_Est4d_SumStats_Test.png', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        images=[f],
#                        captions=['MYSWaningImmunity_Est4d_SumStats_Test'])
test_consol.to_csv(path_wi + 'MYSWaningImmunity_Est4d_SumStats_Test.csv', index=True)
# with open(path_wi + 'MYSWaningImmunity_Est4d_SumStats_Test.csv', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        files=[f],
#                        captions=['MYSWaningImmunity_Est4d_SumStats_Test'])

### End
print('\n----- Ran in ' + '{:.0f}'.format(time.time() - time_start) + ' seconds -----')
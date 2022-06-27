### MYSWaningImmunity
### From CasesDeaths --- ICU and Deaths
### NO ETHNICITY DUMMIES (ONLY IN VAXDUMP)
### Stratified by type
### Est4: For severe outcomes in T2, estimate VE of fully vaxed in month M vs. unvaxed, for M of T1
### d: No sample splitting at all; VE taken from type-timing group terms
### VaryX: Vary choices of covariates (mapping the distribution of \beta)

import pandas as pd
import numpy as np
from datetime import date, timedelta
import statsmodels.discrete.discrete_model as sm
import statsmodels.nonparametric.kde as kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from patsy import dmatrices
import itertools
import time
import telegram_send
import dataframe_image as dfi
from tqdm import tqdm

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

## Trim: Tested positive within lab_cutoff
lab_cutoff_lb = vax_cutoff_ub + timedelta(days=1)
lab_cutoff_ub = date(2021,9,30)
df = df[((df['date_lab'] >= lab_cutoff_lb) &
        (df['date_lab'] <= lab_cutoff_ub)) |
        (df['date_lab'].isna())] # no positive test, or tested positive within lab_cutoff

## Trim: Drop partially vaccinated when infected (already done in CDVD)
df = df[~(df['vax_partial'] == 1)]

## Option: Check (drop specific age groups)
age_cutoff = 18
df = df[df['age'] >= age_cutoff]


### I --- Regression setup
## Controls
# Controls: excluding state dummies
list_col_x_base = ['age', 'malaysian', 'male', 'comorb', 'test_count', 'trace_count'] # New variables (RR)
# New: state dummies
state_dummy = pd.get_dummies(df['state_id'], prefix = 'state', dummy_na = False)
df = pd.concat([df, state_dummy], axis=1)
x = []
for i in list(range(1,15)):
    x.append(i) # 14 states
list_col_state_dummy = ['state_' + str(int(i)) for i in x] # list of state dummy column labels
# New: frontliner dummies (RR)
df['frontliner'] = df['frontliner'].astype('int').round()
fl_dummy = pd.get_dummies(df['frontliner'], prefix='frontliner', dummy_na=False)
df = pd.concat([df, fl_dummy], axis=1)
x = []
for i in list(range(0,3)):
    x.append(i) # 3 levels
list_col_fl_dummy = ['frontliner_' + str(int(i)) for i in x] # list of state dummy column labels
# Merge X
list_col_x_state = list_col_x_base + list_col_state_dummy + list_col_fl_dummy

## Generate combos of X
# no state
list_x_nostate = sum([list(map(list,
                               itertools.combinations(list_col_x_base, i))) for i in range(len(list_col_x_base) + 1)],
                     [])
# no state + fl
list_x_nostate_fl = list_x_nostate.copy()
for i in range(len(list_x_nostate_fl)):
    list_x_nostate_fl[i] = list_x_nostate_fl[i] + list_col_fl_dummy
# with state
list_x_state = list_x_nostate.copy()
for i in range(len(list_x_state)):
    list_x_state[i] = list_x_state[i] + list_col_state_dummy
# with state + fl
list_x_state_fl = list_x_state.copy()
for i in range(len(list_x_state_fl)):
    list_x_state_fl[i] = list_x_state_fl[i] + list_col_fl_dummy
# everything
list_x_master = list_x_nostate + list_x_nostate_fl + list_x_state + list_x_state_fl
del list_x_master[0] # remove blank
del list_x_nostate[0] # remove blank


# Option: whether to run everything (0), only X with state dummies (1), or X without state dummies (2)
option_X = 0 # 0, 1, 2
if option_X == 0:
    list_x_choice = list_x_master.copy()
    # telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
    #                    messages=['X = All combinations'])
elif option_X == 1:
    list_x_choice = list_x_state.copy()
    # telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
    #                    messages=['X = Only with state dummies'])
elif option_X == 2:
    list_x_choice = list_x_nostate.copy()
    # telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
    #                    messages=['X = Only without state dummies'])

# Generate reference sheet
X_reference = pd.DataFrame(list_x_choice)
X_reference.to_csv(path_wi + 'MYSWaningImmunity_Est4d_VaryX_XReference.csv', index=True)
X_reference.to_html(path_wi + 'MYSWaningImmunity_Est4d_VaryX_XReference.html', index=True)
# with open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_XReference.html', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        files=[f],
#                        captions=['Reference for X options'])

## Set up log file
print('MYSWaningImmunity_Est4d_VaryX_Results',
      file=open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_Results.txt', 'w')) # resets file

list_outcomes = ['icu', 'dead']

## Before loops: generate dataframe to hold VE and CI estimates
VE_consol = pd.DataFrame(columns=['X_Option', 'VE', 'LB', 'UB', 'Outcome'])

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

## VaxType-Timing group indicators
df['type_timing'] = df['type_comb'] + df['month'].astype('str') # type-timing; unvax as omitted group
df['type_timing'] = df['type_timing'].str.replace('.0', '', regex=True) # remove trailing .0
df.loc[df['type_timing'] == '0nan', 'type_timing'] = np.nan # treat as NA if unvax
type_timing_dummy = pd.get_dummies(df['type_timing'], prefix='type_timing', dummy_na=False)
df = pd.concat([df, type_timing_dummy], axis=1)
list_type_timing = type_timing_dummy.columns

## R-styled equation for vax status
list_col_D = '+'.join(list_type_timing)

## Option: specific age groups
# option_agerestriction = 1
option_agerestriction = 0
age_restriction = 60 # cutoff
if option_agerestriction == 1:
    df = df[df['age'] >= age_restriction] # CHECK SIGN (LARGER/SMALLER THAN)
    print('Age Restriction: ' + str(age_restriction))
    # telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
    #                    messages=['Age Restriction: ' + str(age_restriction)])
else:
    print('No Age Restriction')
    # telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
    #                    messages=['No Age Restriction'])

## Option: specific states
# option_staterestriction = 1
option_staterestriction = 0
if option_staterestriction == 1:
    list_staterestriction = [12, 14] # KV = 14; Sarawak = 12
    df = df[df['state_id'].isin(list_staterestriction)]
    print('State Restriction: ' + str(list_staterestriction))
    # telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
    #                    messages=['State Restriction: ' + str(list_staterestriction)])
    list_col_x = '+'.join(list_col_x_base) # If drop too many states, then can't control for states
else:
    print('No State Restriction')
    # telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
    #                    messages=['No State Restriction'])

#### --- Estimation Loop --- ####
x_count = 1
for x in tqdm(list_x_choice):
    print('\n' + str(x))
    list_col_x = '+'.join(x)  # create R-style equation for vector X
    try:
        ### II --- Cross-tabulation
        for i in list_outcomes:
            ctab_outcomes_type = pd.crosstab(df[i], df['type_timing'])
            dfi.export(ctab_outcomes_type, path_wi + 'MYSWaningImmunity_Est4d_VaryX_OutcomesTypes.png')
            # with open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_OutcomesTypes.png', 'rb') as f:
            #     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
            #                        images=[f],
            #                        captions=['(Est4d) Outcomes, vax type, and timing\n'])

        ### III --- Estimation
        ## ICU
        # 4b: Full
        y, X = dmatrices('icu ~ ' + list_col_D + ' + ' +
                         list_col_x,
                         df, return_type='dataframe')
        mod = sm.Logit(y, X)
        result = mod.fit(method='newton')
        print('ICU ' + '(All Vax Types)\n ' + 'vaxed early / late & type ' + ' vs. unvaxed',
                  file=open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_Results.txt', 'a'))
        print(result.summary2(),
                  file=open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_Results.txt', 'a'))
        logodds = result.params
        logodds = logodds.map('{:.3f}'.format)
        odds = np.exp(result.params)
        odds = odds.map('{:.3f}'.format)

        VE = pd.DataFrame(odds[list_type_timing])
        VE = VE.rename(columns={0: 'VE'})
        VE = 100*(1 - VE['VE'].astype('float'))
        CI = 100*(1 - np.exp(result.conf_int()) )
        CI = CI.rename(columns={0: 'UB', 1: 'LB'})
        CI = CI[['LB', 'UB']]
        CI = CI.round(1)
        CI = pd.DataFrame(CI.loc[list_type_timing, :])
        VE = pd.concat([VE, CI], axis=1)
        VE['X_Option'] = x_count # 1 to n_X
        VE['Outcome'] = 'ICU' # indicates outcome

        VE_consol = pd.concat([VE_consol, VE], axis=0)

        print(odds)
        print(VE)
        dfi.export(VE, path_wi + 'MYSWaningImmunity_Est4d_VaryX_VE.png')

        ## Death
        # 4c: Full
        y, X = dmatrices('dead ~ ' + list_col_D + ' + ' +
                         list_col_x,
                         df, return_type='dataframe')
        mod = sm.Logit(y, X)
        result = mod.fit(method='newton')
        print('Death ' + '(All Vax Types)\n ' + 'vaxed early/late & type ' + ' vs. unvaxed',
              file=open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_Results.txt', 'a'))
        print(result.summary2(),
              file=open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_Results.txt', 'a'))
        logodds = result.params
        logodds = logodds.map('{:.3f}'.format)
        odds = np.exp(result.params)
        odds = odds.map('{:.3f}'.format)

        VE = pd.DataFrame(odds[list_type_timing])
        VE = VE.rename(columns={0: 'VE'})
        VE = 100 * (1 - VE['VE'].astype('float'))
        CI = 100 * (1 - np.exp(result.conf_int()))
        CI = CI.rename(columns={0: 'UB', 1: 'LB'})
        CI = CI[['LB', 'UB']]
        CI = CI.round(1)
        CI = pd.DataFrame(CI.loc[list_type_timing, :])
        VE = pd.concat([VE, CI], axis=1)
        VE['X_Option'] = x_count  # 1 to n_X
        VE['Outcome'] = 'Death' # indicates outcome

        VE_consol = pd.concat([VE_consol, VE], axis=0)

        print(odds)
        print(VE)
        dfi.export(VE, path_wi + 'MYSWaningImmunity_Est4d_VaryX_VE.png')
    except:
        print('Error at X_Option = ' + str(x_count))
        # telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
        #                    messages=['Error at X_Option = ' + str(x_count)])
    x_count += 1 # increase by 1

## Sort consolidated VE table
VE_consol.loc[((VE_consol['VE'] <= 0) | (VE_consol['VE'] >= 100)),
              ['VE', 'LB', 'UB']] = np.nan # rows estimated with insufficient samples (usually April)
VE_consol = VE_consol.reset_index(drop=False)
VE_consol.rename(columns={'index':'Type-Timing'},
                 inplace=True)
VE_consol.sort_values(by=['Outcome', 'X_Option', 'Type-Timing'],
                      ascending=[False, True, True],
                      inplace=True)
VE_consol = VE_consol.reset_index(drop=True)
# dfi.export(VE_consol, path_wi + 'MYSWaningImmunity_Est4d_VaryX_VE_consol.png')
VE_consol.to_csv(path_wi + 'MYSWaningImmunity_Est4d_VaryX_VE_consol.csv', index=False)
# with open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_VE_consol.png', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        images=[f],
#                        captions=['MYSWaningImmunity_Est4d_VaryX_VE_consol'])
# with open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_VE_consol.csv', 'rb') as f:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        files=[f],
#                        captions=['MYSWaningImmunity_Est4d_VaryX_VE_consol'])

## Distribution of VE
list_outcomes_nice = ['ICU', 'Death']
list_colour = ['black', 'darkred']
list_type = ['pp', 'ss']
list_type_text = ['BNT162b2', 'CoronaVac']
list_timing = ['1', '2']
list_timing_text = ['Early', 'Late']
for i in tqdm(list_outcomes_nice):
    VE_break_outcome = VE_consol[VE_consol['Outcome'] == i]  # break consol VE frame by outcomes
    for v, vtext in zip(list_type, list_type_text):
        VE_break_outcomevax = VE_break_outcome[VE_break_outcome['Type-Timing'].str.contains(v, regex=True)]
        fig = make_subplots() # reset chart
        for m, mtext, j in zip(list_timing, list_timing_text, list_colour):
            VE_break_outcomevaxtime = VE_break_outcomevax[VE_break_outcomevax['Type-Timing'].str.contains(m, regex=True)]  # further restrict by timing of vax
            df_dens = kde.KDEUnivariate(VE_break_outcomevaxtime['VE'])  # diff label so as to not clash with the logit regressions
            df_dens.fit(kernel='gau', bw=1, fft=True)  # produces the kernel
            fig.add_traces(go.Scatter(x=df_dens.support,
                                      y=df_dens.density,
                                      name=mtext,
                                      marker=dict(color=j)
                                      ))  # overlays M = m
        fig.update_layout(title='Distribution of VE (All Combinations of Covariates): ' + str(i) + ', ' + vtext,
                          xaxis_title='VE (%)',
                          yaxis_title='Density',
                          hovermode='x',
                          plot_bgcolor='white',
                          font=dict(color='black'))
        fig.write_image(path_wi + 'MYSWaningImmunity_Est4d_VaryX_VE_consol_KDist.png')
        # with open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_VE_consol_KDist.png', 'rb') as f:
        #     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
        #                        images=[f],
        #                        captions=['MYSWaningImmunity_Est4d_VaryX\n' +
        #                                  'Distribution of VE Estimates (Diff Combo of X): ' + str(i) + ', ' + v])

### IV --- Export log file
# with open(path_wi + 'MYSWaningImmunity_Est4d_VaryX_Results.txt', 'rb') as i:
#     telegram_send.send(conf='EcMetrics_Config_GeneralFlow.conf',
#                        files=[i],
#                        captions=['MYSWaningImmunity_Est4d_VaryX_Results:\n' +
#                                  'Log-Odds'])

### End
print('\n----- Ran in ' + "{:.0f}".format(time.time() - time_start) + ' seconds -----')
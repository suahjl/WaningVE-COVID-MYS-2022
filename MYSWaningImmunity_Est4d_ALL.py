#### Runs all scripts required to replicate findings in https://doi.org/10.1016/j.ijid.2022.03.028
#### Takes data file from https://github.com/suahjl/WaningVE-COVID-MYS-2022/blob/main/Data/CasesDeaths_WaningImmunity_Data_RR_ForRelease.parquet

import runpy

### Overall VE estimates in table 4
runpy.run_path('2_MYSWaningImmunity_Est4d_RR.py')
### Age-stratified VE estimates in table 4
runpy.run_path('2_MYSWaningImmunity_Est4d_AgeStrat_RR.py')
### Event rates in table 4
runpy.run_path('2_MYSWaningImmunity_Est4d_IncidenceRates_RR.py')
### Summary statistics in table 3
runpy.run_path('2_MYSWaningImmunity_Est4d_SumStats_RR.py')
### Distribution of VE estimates in table S4 of supplementary appendix
runpy.run_path('2_MYSWaningImmunity_Est4d_VaryX_RR.py')

### End
print('Done. Check directory for output files.')
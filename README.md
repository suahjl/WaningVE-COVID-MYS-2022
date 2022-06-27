# WaningVE-COVID-MYS-2022
Data and scripts used in Jing Lian Suah, Masliyana Husin, Peter Seah Keng Tok, Boon Hwa Tng, Thevesh Thevananthan, Ee Vien Low, Maheshwara Rao Appannan, Faizah Muhamad Zin, Shahanizan Mohd Zin, Hazlina Yahaya, Kalaiarasu M. Peariasamy, Sheamini Sivasampu, Waning COVID-19 Vaccine Effectiveness for BNT162b2 and CoronaVac in Malaysia: An Observational Study, International Journal of Infectious Diseases, Volume 119, 2022, Pages 69-76, ISSN 1201-9712, https://doi.org/10.1016/j.ijid.2022.03.028.

As at 2022-06-27, this repository contains only scripts and anonymised consolidated data for the severe outcomes (ICU admission and death) segment of the paper. Work is in progress to make available scripts and data for the infection segment.

The full paper can be found [here](https://www.ijidonline.com/article/S1201-9712(22)00167-9/fulltext).

# Replication notes
Before running the scripts, please edit the lines containing file paths to reflect your local directories. In general *path_input* should contain the consolidated anonymised data set, and *path_wi* should be your intended output directory. All *telegram-send* blocks have been commented out; feel free to use your own telegram bot.

1. Step 0 (*0_MYSWaningImmunity_CasesDeaths_Data_RR.py*) consolidates the raw input data from various administrative data sets described in the paper. While these input data cannot be made public at this juncture, we have decided to release the associated script for transparency.
2. Step 1 (*1_MYSWaningImmunity_CasesDeaths_Data_RR_ForRelease.py*) anonymises the consolidated data set, and runs checks that the data set is indeed anonymised.
3. Step 2 scripts can be run in parallel, and reproduces the findings described in the paper.
* 2_MYSWaningImmunity_Est4d_RR.py generates the overall VE estimates found in table 4.
* 2_MYSWaningImmunity_Est4d_AgeStrat_RR.py generates the age-stratified VE estimates found in table 4.
* 2_MYSWaningImmunity_Est4d_IncidenceRates_RR.py calculates the overall and age-stratified event rates (per 1000) found in table 4.
* 2_MYSWaningImmunity_Est4d_SumStats_RR.py calculates the figures found in table 3.
* 2_MYSWaningImmunity_Est4d_VaryX_RR.py generates the distribution of VE estimates found in figure S4 of the supplementary appendix.

# Data notes
1. [CasesDeaths_WaningImmunity_Data_RR_ForRelease.parquet](https://github.com/suahjl/WaningVE-COVID-MYS-2022/blob/main/Data/CasesDeaths_WaningImmunity_Data_RR_ForRelease.parquet) contains a data frame of 1869619 rows, each representing unique confirmed COVID-19 cases used in the paper. 
* ```date1```: Date of receipt of first dose
* ```date2```: Date of receipt of second dose (the paper defines fully vaccinated as 14 days after this date)
* ```date_lab```: Date of SARS-CoV-2 infection confirmation test
* ```date_death```: Date of death due to COVID-19
* ```dead```: Dummy indicating dead due to COVID-19
* ```symptomatic```: Dummy indicating symptomatic at presentation (not used in analysis)
* ```icu```: Dummy indicating COVID-19-related ICU admission
* ```comorb```: Dummy indicating presence of comorbidities
* ```age```: Age
* ```malaysian```: Dummy indicating if the individual is a Malaysian national or otherwise
* ```male```: Dummy indicating sex of the individual
* ```state_id```: Factors indicating state of residence
* ```type1```: Brand of the first dose
* ```type2```: Brand of the second dose
* ```type_comb```: Vaccine combination (2x BNT162b2 = pp; 2x CoronaVac = ss; unvaccinated = 0)
* ```vax_partial```: Dummy indicating if partially vaccinated (>= 1 day post-dose 1 but ) at the point of confirmation of SARS-CoV-2 infection (not used in analysis)
* ```vax_full```: Dummy indicating if fully vaccinated (>= 14 days post-dose 2) at the point of confirmation of SARS-CoV-2 infection
* ```test_count```: Number of supervised tests (RTK-Ag and RT-PCR) taken during the baseline period
* ```trace_count```: Number of times labelled as a 'casual contact' by Malaysia's check-in-based automated contact tracing system during the baseline period
* ```frontliner```: Factors indicating healthcare frontliner status (0 = general public, 1 = public healthcare worker, 2 = private healthcare worker)
* ```reinf```: Dummy indicating SARS-CoV-2 reinfection (all rows with reinf = 1 are dropped in the analysis)
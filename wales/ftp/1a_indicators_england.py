# -*- coding: utf-8 -*-
"""
Modified Feb 17 2023
# This script aims to :
    1) retreieve metadata ,
    2) retrieve data from fingertips by indicator
    3) format wide timeseries and creates columns needed for ppi
"""
# Install fingertips with pip install fingertips_py

# 0) import libraries, define paths
import pandas as pd
import numpy as np
from math import isnan
import fingertips_py as ftp
home = "~/"  # add path acordingly
output_path = home + "output/england/"

# 1) import metadata for 39 profiles
metadata = []
for i in [155,18,19,20,26,29,30, 32, 36, 37,40,41,45,46,55,58,65,76,79,84,86,87,91,92,95,98,99,101,102,105,106,125,130,133,135,139,141,143,146]:
    sm = ftp.get_metadata_for_profile_as_dataframe(i)
    sm['profile_id'] = i
    sm['profile'] = sm.profile_id.map(profile_dict)
    metadata.append(sm)
    
metadata = pd.concat(metadata)

# spot profiles
metadata = ftp.get_metadata_for_profile_as_dataframe(155)
metadata = metadata.loc[~metadata.Indicator.str.contains("Spend"),]
metadata = metadata[['Indicator ID', 'Indicator','Frequency','Year type','Unit','Value type','Polarity']]
metadata['profile_id'] = 155


# 2) based on metadata from 1) we pull each indicator together with some metadata, 
# appends all indicators and  retrieve best and worst bounds

alldatayears = []
for index, row in metadata.iterrows(): 
    i = row['Indicator ID'] 
    j = row['profile_id'] 
    df = ftp.retrieve_data.get_data_by_indicator_ids(indicator_ids=i, 
        area_type_id=102, parent_area_type_id=15, profile_id=j,
        include_sortable_time_periods=1, is_test=False)
    if df.empty == True:
        df = ftp.retrieve_data.get_data_by_indicator_ids(indicator_ids=i, 
            area_type_id=202, parent_area_type_id=15, profile_id=j,
            include_sortable_time_periods=1, is_test=False)
        if df.empty == True:
            df = ftp.retrieve_data.get_data_by_indicator_ids(indicator_ids=i, 
                area_type_id=302, parent_area_type_id=15, profile_id=j,
                include_sortable_time_periods=1, is_test=False)
            if df.empty == True:
                df = ftp.retrieve_data.get_data_by_indicator_ids(indicator_ids=i, 
                    area_type_id=402, parent_area_type_id=15, profile_id=j,
                    include_sortable_time_periods=1, is_test=False)
    
    # we add useful metadata into the dataset
    df['unit'] = row['Unit'] 
    df['value_type'] = row['Value type']
    df['frequency'] = row['Frequency']
    df['year_type'] = row['Year type']
    df['polarity'] = row['Polarity']

    # creates a unique ID based on indicator and area code
    df['seriesCode'] = df['Indicator ID'].astype(str) + "_" + df['Area Code'].astype(str)
    df['seriesName'] = df['Indicator Name']
    df['group'] = df['Category'].fillna('local_authority')
    df['subgroup'] = df['Category Type'].fillna('local_authority')
    
    """
    # to de-duplicate for geographies we select all-gender and all-category data
    df = df.loc[df['Category Type'].isna() & df['Parent Code'].notnull() & \
                (df['Sex'] == "Persons"),]
    """

    # to de-duplicate for deprivation levels activate these lines
    #df = df.loc[df['Parent Code'].isna() & (df['Category Type'] == "General Practice deprivation deciles in England (IMD2010)") & \
    df = df.loc[~(df['Category Type'].isna()),]
    df = df.loc[df['Parent Code'].isna()]
    df = df.loc[df['Sex'] == "Persons",]

    # correct exception for indicator 91195 which is duplicated
    if i == 91195:
        df = df.loc[~df.Value.isna(),]

    assert df.duplicated(subset =  ['seriesCode','Age','group','subgroup', \
                    'Time period Sortable','Time period range']).any() == False

    df = df[['seriesCode','seriesName','Age','group','subgroup', \
             'Value', 'unit','value_type','polarity', \
            'frequency','year_type','Time period Sortable','Time period range', \
            'Indicator ID','Area Code']]
    alldatayears.append(df)

alldatayears = pd.concat(alldatayears)

# polarity of the following indicators is stated as 'Not applicable' yet 
# they should be inverted as their increase reflects a loss of wellbeing
toinvert = ['Cancer diagnosed at early stage (experimental statistics)',
       'Fraction of mortality attributable to particulate air pollution (old method)',
       'Chlamydia detection rate per 100,000 aged 15 to 24',
       'Number in treatment at specialist alcohol misuse services',
       'Re-offending levels - average number of re-offences per re-offender',
       'Domestic abuse related incidents and crimes',
       'Violent crime - violence offences per 1,000 population',
       'Violent crime - sexual offences per 1,000 population',
       'Re-offending levels - percentage of offenders who re-offend',
       'First time offenders',
       'Adults in treatment at specialist alcohol misuse services: rate per 1000 population',
       'Under 18s conceptions leading to abortion (%)',
       'Under 18s abortions rate / 1,000',
       'Adults in treatment at specialist drug misuse services: rate per 1000 population',
       'Adults with a learning disability who live in stable and appropriate accommodation',
       'Adults in contact with secondary mental health services who live in stable and appropriate accommodation',
       'Abortions under 10 weeks (%)',
       'Estimated diabetes diagnosis rate',
       'Gap in life expectancy at birth between each local authority and England as a whole']

# we generate the invert column based on the polarity column from metadata
alldatayears['invert'] = 0
alldatayears.loc[(alldatayears.polarity == "RAG - Low is good   ") |
                 (alldatayears.seriesName.isin(toinvert)),'invert'] = 1

# retrieve best and worst bounds based on data
max_value = alldatayears.groupby(['Indicator ID'], as_index = False, observed = True)['Value'].max().rename(columns = {'Value':'bestBound'})
min_value = alldatayears.groupby(['Indicator ID'], as_index = False, observed = True)['Value'].min().rename(columns = {'Value':'worstBound'})

# retrieve first and last value of time series by deduplicating on key columns
start_value = alldatayears.loc[alldatayears.Value.notnull(),['seriesCode','seriesName','Age','subgroup', 'Time period range', 'Time period Sortable','Value']]
start_value = start_value.loc[~alldatayears.Value.isna(),['seriesCode','seriesName','Age','subgroup', 'Time period range','Time period Sortable','Value']]
start_value = start_value.sort_values(['seriesCode','seriesName','Age' ,'subgroup', 'Time period range','Time period Sortable'])
last_value = start_value.copy()
start_value = start_value.drop_duplicates(subset = ['seriesCode','seriesName','Age','subgroup', 'Time period range'], \
                keep = 'first').rename(columns = {'Value':'start_value'})
last_value = last_value.drop_duplicates(subset = ['seriesCode','seriesName','Age','subgroup', 'Time period range'], \
                keep = 'last').rename(columns = {'Value':'end_value'})

# merge these columns onto the main database
alldatayears = alldatayears.merge(max_value, on = 'Indicator ID', validate = 'many_to_one')
alldatayears = alldatayears.merge(min_value, on = 'Indicator ID', validate = 'many_to_one')
alldatayears = alldatayears.merge(start_value, on = ['seriesCode','seriesName','Age' ,'subgroup', 'Time period range'], validate = 'many_to_one')
alldatayears = alldatayears.merge(last_value, on = ['seriesCode','seriesName','Age' ,'subgroup', 'Time period range'], validate = 'many_to_one')

# adjust theoretical bounds for proportion (0,1) types of data
alldatayears.loc[alldatayears.value_type == "Percentage point",'worstBound'] = -100
alldatayears.loc[alldatayears.value_type == "Percentage point",'bestBound'] = 100

# normalize initial and final values of the series
alldatayears['I0'] = (alldatayears.start_value - alldatayears.worstBound) / (alldatayears.bestBound - alldatayears.worstBound)
alldatayears['IF'] = (alldatayears.end_value - alldatayears.worstBound) / (alldatayears.bestBound - alldatayears.worstBound)
                                
# assume all indicators are instrumental 
alldatayears['instrumental'] = 1

# create a flag for frequency
alldatayears['flag_non_annual'] = 1
alldatayears.loc[alldatayears.frequency.str.contains("nnual") == True,
                 'flag_non_annual'] = 0
alldatayears.loc[alldatayears['Time period range'] == '1y',
                 'flag_non_annual'] = 0
alldatayears.loc[alldatayears.year_type == 'Financial',
                 'flag_non_annual'] = 0

# pivot wide on years
#alldatayears = alldatayears.drop_duplicates(subset =  ['seriesCode','seriesName', \
 #           'Area Code', 'Indicator ID', 'invert','instrumental','flag_non_annual',
  #          'bestBound','worstBound','Time period Sortable'])

data = pd.pivot(alldatayears, index = ['seriesCode','seriesName','Age' ,'subgroup', 'Time period range', \
            'Area Code', 'Indicator ID','I0','IF', \
            'invert','bestBound','worstBound','instrumental','flag_non_annual'], \
                        columns = 'Time period Sortable', 
                        values = 'Value').reset_index()

# FIXME: Deduplicate by time period range
# We add SPOT categories which match to expenditure/budget
categories = pd.read_csv('c:/users/giselara/Documents/he/output/england/spot_indicator_mapping_table.csv', encoding = 'unicode_escape')
categories = categories.loc[categories.type == "Outcome",['category','name']].rename(columns = {'name':'seriesName'}).drop_duplicates()
data = data.merge(categories, on = 'seriesName', validate = 'many_to_many', how = 'outer')


# 3) format columns for PPI 
years = [column_name for column_name in data.columns if str(column_name).isnumeric()]

# 3a) Exclude indicators with fewer than 5 datapoints
data['count_valid'] = data[years].notnull().sum(axis = 'columns')
data = data.loc[(data.count_valid > 4),]

# 3b) Normalise values
normalised_series = []
for index, row in data.iterrows():
    time_series = row[years].values
    count_valid = len([x for x in normalised_serie if not isnan(x)])
    if count_valid > 4:
        normalised_serie = (time_series - row.worstBound)/(row.bestBound - row.worstBound)
        if row.invert == 1:
            final_serie = 1 - normalised_serie
        else:
            final_serie = normalised_serie
    
    #    initial_value =  next(x for x in normalised_serie if not isnan(x))
     #   final_value = next(x for x in reversed(normalised_serie) if not isnan(x))
      #  nvalue = [initial_value, final_value]
        normalised_series.append( final_serie )   
   # nvalues.append(nvalue)
    
df = pd.DataFrame(normalised_series, columns=years)


# 3.b) Normalise the theoretical bounds and add all the other columns
df['seriesCode'] = data.seriesCode
df['seriesName'] = data.seriesName
df['minVals'] = np.zeros(len(data))
df['maxVals'] = np.ones(len(data))
df['I0'] = data.I0
df['IF'] = data.IF
df['instrumental'] = data.instrumental
df['sdg'] = data.category

colour_dict = {"Central": "#A21942",	"Child Health": "#FD9D24",	"Cultural": "#FF3A21",	"Drugs and Alcohol": "#E5243B",	"Education": "#DDA63A",	"Env & Reg": "#4C9F38",	"Health Improvement": "#C5192D",	"Health Protection": "#26BDE2",	"Healthcare": "#FCC30B",	"Highways": "#FD6925",	"Housing": "#BF8B2E",	"Mental Health": "#3F7E44",	"Planning": "#0A97D9",	"Public Health": "#56C02B",	"Sexual Health": "#00689D",	"Social Care - Adults": "#19486A",	"Social Care - Child": "#19486A",	"Tobacco Control": "#E5243B"}
df['colour'] = df.sdg.map(colour_dict)

#Building new variables
successRates = np.sum(df[years].values[:,1::] > df[years].values[:,0:-1], axis=1)/(len(years)-1)

# if a success rate is 0 or 1, it is recommended to replace them by a very low or high value as 
# zeros and ones are usually an artifact of lacking data on enough policy trials in the indicator
successRates[successRates==0] = .05
successRates[successRates==1] = .95
df['successRates'] = successRates

###development gaps
df.loc[df.I0==df.IF, 'IF'] = df.loc[df.I0==df.IF, 'IF']*1.05

#governance paramenters
df['qm'] = 0.5 # quality of monitoring
df['rl'] = 0.5 # quality of the rule of law


# export data to model ppi.py
df.to_csv(outputpath + 'clean_data/data_indicators.csv')
# -*- coding: utf-8 -*-
"""
Modified Feb 17 2023
# This script aims to :
    1) retreieves metadata and data on expenditure from fingertips,
    2) checks the budget categories for which at least an indicator was retrieved
    3) format wide timeseries and creates columns needed for ppi
"""
# Install fingertips with pip install fingertips_py

# 0) import libraries, define paths
import pandas as pd
import numpy as np
import fingertips_py as ftp
home = "~/"  # add path acordingly
output_path = home + "output/england/"

# read indicators data
data_indi = pd.read_csv('C:/Users/giselara/Documents/he/output/england/data_indicators.csv')
# spot profiles
metadata = ftp.get_metadata_for_profile_as_dataframe(155)
metadata = metadata.loc[metadata.Indicator.str.contains("Spend"),]
metadata = metadata[['Indicator ID', 'Indicator','Year type','Unit','Value type','Frequency']]
metadata['profile_id'] = 155
# please note that the following are constant throughtout the data: Unit = £, Value type=Count, Year type = financial, Polarity = BOB

# 2) based on metadata from 1) we pull each expenditure together with some metadata, 
# appends all indicators and retrieve best and worst bounds
allexpenditure = []
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

    # creates a unique ID based on indicator and area code
    df['seriesCode'] = df['Indicator ID'].astype(str) + "_" + df['Area Code'].astype(str)
    df['seriesName'] = df['Indicator Name']
    
    # duplication is not an issue at geography-level spend data
    assert df.duplicated(subset =  ['seriesCode','Time period Sortable']).any() == False

    df = df[['seriesCode','seriesName','Value', 'unit','value_type','frequency',\
             'year_type','Time period Sortable', 'Indicator ID','Area Code']]

    allexpenditure.append(df)

data_exp = pd.concat(allexpenditure)

# We add SPOT categories which match to expenditure/budget
categories = pd.read_csv('c:/users/giselara/Documents/he/output/england/spot_indicator_mapping_table.csv', encoding = 'unicode_escape')
categories = categories.loc[categories.type == "Spend",['category','name']].rename(columns = {'name':'seriesName'}).drop_duplicates()
#categories['seriesName'] = categories.name.str.slice(7,)
data_exp = data_exp.merge(categories, on = 'seriesName', validate = 'many_to_many', how = 'outer')

data_exp['sdg'] = data_exp.category
sdg_dict = {"Central": "#A21942",	"Child Health": "#FD9D24",	"Cultural": "#FF3A21",	"Drugs and Alcohol": "#E5243B",	"Education": "#DDA63A",	"Env & Reg": "#4C9F38",	"Health Improvement": "#C5192D",	"Health Protection": "#26BDE2",	"Healthcare": "#FCC30B",	"Highways": "#FD6925",	"Housing": "#BF8B2E",	"Mental Health": "#3F7E44",	"Planning": "#0A97D9",	"Public Health": "#56C02B",	"Sexual Health": "#00689D",	"Social Care - Adults": "#19486A",	"Social Care - Child": "#19486A",	"Tobacco Control": "#E5243B"}
data_exp['colour'] = data_exp.sdg.map(sdg_dict)


# 2) We check which budget categories had at least an indicator retrieved
data_exp = data_exp[data_exp.category.isin(data.category.values)]
data_exp = data_exp[data_exp.category.isin(data[data.instrumental==1].category.values)]

# pivot wide on years
data_exp = pd.pivot(data_exp_or, index = ['seriesCode','seriesName', 'Indicator ID', \
            'Area Code', 'category','sdg','colour'], columns = 'Time period Sortable', \
                        values = 'Value').reset_index().fillna(-1)

#data_exp = pd.read_csv('https://raw.githubusercontent.com/oguerrer/ppi/main/tutorials/raw_data/raw_expenditure.csv')

# 3) Assembly disbursment matrix
years = [column_name for column_name in data_exp.columns if str(column_name).isnumeric()]
periods = len(years)
T = 69
t = round(T/periods)

new_rows = []
for index, row in data_exp.iterrows():
    new_row = [row['seriesCode'], row['Area Code'], row.category]
    for year in years:
        new_row += [int(row[year]) for i in range(t)]
    new_rows.append(new_row)
    
df_exp = pd.DataFrame(new_rows, columns=['seriesCode','Area Code','category']+list(range(T+1)))

is_instrumental = dict(zip(data_indi['seriesCode'], data_indi.instrumental==1))

rel_dict = dict([(code, []) for code in data_indi.seriesCode if is_instrumental[code]])
for index, row in data_indi.iterrows():
    if row.seriesCode in rel_dict:
        rel_dict[row.seriesCode].append(row.category+"_"+row['Area Code'])
#        rel_dict.append(row['Area Code'])
    
n_cols = max([len(value) for value in rel_dict.values()])

M = [['' for i in range(n_cols+1)] for code in rel_dict.values()]
for i, items in enumerate(rel_dict.items()):
    sdg, indis = items
    M[i][0] = sdg
    for j, indi in enumerate(indis):
        M[i][j+1] = indi

df_rel = pd.DataFrame(M, columns=['seriesCode']+list(range(n_cols)))


df_exp.to_csv('clean_data/data_expenditure.csv', index=False)
df_rel.to_csv('clean_data/data_relational_table.csv', index=False)
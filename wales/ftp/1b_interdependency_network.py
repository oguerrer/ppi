"""
Modified Feb 17 2023
# This script aims to :
    1) retreieve metadata and data from fingertips by indicators,
    2) construct a matrix with pairwise Pearson correlations
    3) filter edges weighting lower than 0.5
    4) save the network using indicators' ids
"""
# Install fingertips with pip install fingertips_py

# 0) import libraries, define paths
import pandas as pd
import numpy as np
import fingertips_py as ftp
home = "~/"  # add path acordingly
output_path = home + "output/england/"

# 1a) Retrieve some useful metadata from SPOT profile
metadata = ftp.get_metadata_for_profile_as_dataframe(155)
metadata = metadata.loc[~metadata.Indicator.str.contains("Spend"),]
metadata = metadata[['Indicator ID', 'Indicator','Frequency','Year type','Unit','Value type','Polarity']]
metadata['profile_id'] = 155

# 1b) based on metadata from 1) we pull each expenditure together with some metadata, 
# appends all indicators and retrieve best and worst bounds
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
             'Value', 'Time period Sortable','Time period range', \
            'Indicator ID','Area Code']]

    alldatayears.append(df)

data = pd.concat(alldatayears)

# pivot wide on years
data = pd.pivot(data, index = ['seriesCode','seriesName','Age','group','subgroup', 'Time period range', \
            'Area Code', 'Indicator ID'], columns = 'Time period Sortable', \
                        values = 'Value').reset_index() #.fillna(0)
#data = data.loc[data['Area Code']=="E92000001",]
#data=pd.read_csv('c:/users/giselara/Documents/he/output/england/test.csv')
# 2) Construct pairwise Pearson correlations matrix, 
# directionality is from row to column.
N = len(data)
M = np.zeros((N, N))
years = [column_name for column_name in data.columns if str(column_name).isnumeric()]

for i, rowi in data.iterrows():
    for j, rowj in data.iterrows():
        if i!=j:
            serie1 = rowi[years].values.astype(float)[1::]
            serie2 = rowj[years].values.astype(float)[0:-1]
            change_serie1 = serie1[1::] - serie1[0:-1]
            change_serie2 = serie2[1::] - serie2[0:-1]
            if not np.all(change_serie1 == change_serie1[0]) and not np.all(change_serie2 == change_serie2[0]):
                M[i,j] = np.corrcoef(change_serie1, change_serie2)[0,1]
                
# 3) Filter edges that have a weight of magnitude lower than 0.5

M[np.abs(M) < 0.5] = 0

# 4) Save the network as a list of edges using the indicators' ids

ids = data.seriesCode.values
edge_list = []
for i, j in zip(np.where(M!=0)[0], np.where(M!=0)[1]):
    edge_list.append( [ids[i], ids[j], M[i,j]] )
df = pd.DataFrame(edge_list, columns=['origin', 'destination', 'weight'])
df.to_csv('c:/users/giselara/Documents/he/output/england/data_network.csv', index=False)
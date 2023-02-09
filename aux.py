import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import csv


df = pd.read_excel("SDR-2022-database.xlsx", sheet_name='Raw Data - Trend Indicators')


new_rows = []
for country, group in df.groupby('id'):
    data_columns = group.columns.values[5::]
    for column in data_columns:
        column_new = column.replace(' ', '')
        if column_new == 'sdg2_stuntihme':
            column_new = 'sdg2_stunting'
        if column_new == 'sdg2_wasteihme':
            column_new = 'sdg2_wasting'
        if column_new == 'sdg5_fplmodel':
            column_new = 'sdg5_familypl'
        new_row = [column_new, int(column.split('_')[0][3::])] + group[column].values.tolist()
        new_rows.append(new_row)

all_years = sorted(df.Year.unique())


dfn = pd.DataFrame(new_rows, columns=['seriesCode', 'sdg']+[str(year) for year in all_years])
colYears = [c for c in dfn.columns if str(c).isnumeric() ]

dfn = dfn[np.isnan(dfn[colYears].values).sum(axis=1) == 0]

indices = []
for code in dfn.seriesCode.unique():
    index = rd.sample(np.where(dfn.seriesCode == code)[0].tolist(), 1)[0]
    indices.append(index)


dfn = dfn.iloc[indices]
dfn = dfn[dfn.seriesCode!='sdg10_gini']



dff = pd.read_excel("SDR-2022-database.xlsx", sheet_name='Codebook')
indis_names = dict(zip(dff.IndCode, dff.Indicator))
dfn['seriesName'] = [indis_names[code] for code in dfn.seriesCode]


min_vals = dict(zip(dff.IndCode, dff['Optimum (= 100)']))
max_vals = dict(zip(dff.IndCode, dff['Lower Bound (=0)']))

dfn['bestBound'] = [max_vals[code] for code in dfn.seriesCode]
dfn['worstBound'] = [min_vals[code] for code in dfn.seriesCode]



dft = pd.read_csv("SDR_tech_bounds.csv")
instrumentals = dict(zip(dft.IndCode, dft.instrumental))
invert = dict(zip(dft.IndCode, dft.invert))

dfn['instrumental'] = [instrumentals[code] if code in instrumentals else 1 for code in dfn.seriesCode]
dfn['invert'] = [invert[code] if code in invert else 1 for code in dfn.seriesCode]
dfn.loc[np.isnan(dfn.invert), 'invert'] = 0

for index, row in dfn.iterrows():
    min_val = min([row.bestBound, row.worstBound])
    max_val = max([row.bestBound, row.worstBound])
    norm_vals = (row[colYears].values - min_val)/(max_val - min_val)
    if np.sum( (norm_vals>1) | (norm_vals<0) ) > 0:
        print(row.seriesCode)
        

file = open("sdg_colors.txt", 'r')
colors_sdg = dict([(i+1, line[0]) for i, line in enumerate(csv.reader(file))])
file.close()
dfn['color'] = [colors_sdg[sdg] for sdg in dfn.sdg]


dfn.to_csv("raw_indicators.csv", index=False)

























































# -*- coding: utf-8 -*-
"""
# This script aims to :
    1) Load cleaned indicators data from step 1a),
    2) load interdependency network from step 1b),
    3) filter edges weighting lower than 0.5
    4) save the network using indicators' ids
"""

# 0) import libraries, define paths
import pandas as pd
import numpy as np
home = "~/"  # add path acordingly
output_path = home + "output/england/"

import requests
url = 'https://raw.githubusercontent.com/oguerrer/ppi/main/source_code/ppi.py'
r = requests.get(url)
with open('ppi.py', 'w') as f:
    f.write(r.text)
import ppi as ppi

df_indis = pd.read_csv(outpath + 'clean_data/data_indicators.csv')
df_indis = pd.read_csv('c:/users/giselara/Documents/he/output/england/data_indicators.csv')
df_indis = df_indis.loc[df_indis.I0.notnull(),]
df_indis = df_indis.loc[df_indis.IF.notnull(),]

N = len(df_indis)
I0 = df_indis.I0.values # initial values
IF = df_indis.IF.values # final values
success_rates = df_indis.successRates.values # success rates
R = df_indis.instrumental # instrumental indicators
qm = df_indis.qm.values # quality of monitoring
rl = df_indis.rl.values # quality of the rule of law
indis_index = dict([(code, i) for i, code in enumerate(df_indis.seriesCode)]) # used to build the network matrix

# 2) to load interdependency network from step 2b)
df_net = pd.read_csv(outpath + 'clean_data/data_network.csv')
A = np.zeros((N, N)) # adjacency matrix
for index, row in df_net.iterrows():
    i = indis_index[row.origin]
    j = indis_index[row.destination]
    w = row.weight
    A[i,j] = w
    
df_exp = pd.read_csv('c:/users/giselara/Documents/he/output/england/data_expenditure.csv')

Bs = df_exp.values[:,1::] # disbursement schedule (assumes that the expenditure programmes are properly sorted)

df_rela = pd.read_csv('c:/users/giselara/Documents/he/output/england/data_relational_table.csv')

B_dict = {}
for index, row in df_rela.iterrows():
    B_dict[indis_index[row.seriesCode]] = [programme for programme in row.values[1::][row.values[1::].astype(str)!='nan']]
    
T = Bs.shape[1]
parallel_processes = 4 # number of cores to use
threshold = 0.6 # the quality of the calibration (maximum is near to 1, but cannot be exactly 1)
low_precision_counts = 50 # number of low-quality evaluations to accelerate the calibration

parameters = calibrate(I0, IF, success_rates, R=R, qm=qm, rl=rl, Bs=Bs, B_dict=B_dict,
              T=T, threshold=threshold, verbose=True,
             low_precision_counts=low_precision_counts)

df_params = pd.DataFrame(parameters[1::], columns=parameters[0])

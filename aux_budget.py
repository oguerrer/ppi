import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rd
import csv




series = []
for sdg in range(1,18):
    serie = np.random.randint(1, 100) + 10 + np.random.rand(23)*5
    series.append([sdg] + serie.tolist())



df = pd.DataFrame(series, columns=['sdg']+list(range(2000, 2023)))
df.to_csv("tutorials/raw_data/raw_expenditure.csv", index=False)

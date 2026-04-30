#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: marcosmorfulis
"""

import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import lilliefors
import warnings

warnings.filterwarnings('ignore')

# CONFIG
path_grains = '/Users/marcosmorfulis/Desktop/Grain_Assignments_.xlsx'
path_results = '/Users/marcosmorfulis/Desktop/Classification_Results.xlsx'
output_stats = '/Users/marcosmorfulis/Desktop/Statistical_Diag_.xlsx'

def calculate_mswd_robust(ages, errors_1s):
    n = len(ages)
    if n <= 1: 
        return np.nan
    weights = 1.0 / (errors_1s**2)
    weighted_mean = np.sum(ages * weights) / np.sum(weights)
    ssw = np.sum(((ages - weighted_mean)**2) / (errors_1s**2))
    return ssw / (n - 1)

# cargo la data
df_grains = pd.read_excel(path_grains)
df_results = pd.read_excel(path_results)

# limpiar nombres
df_results['Muestra_Clean'] = df_results['Sample'].astype(str).str.split('(').str[0].str.strip().str.upper()
df_grains['Sample_Clean'] = df_grains['Sample'].astype(str).str.split('(').str[0].str.strip().str.upper()

mapping_n = dict(zip(df_results['Muestra_Clean'], df_results['N_Selected']))
diagnostic_list = []

for sample in df_grains['Sample_Clean'].unique():
    k_opt = mapping_n.get(sample)
    
    if k_opt is None or pd.isna(k_opt):
        print(f"No encontre k para: {sample}")
        continue
    
    k_opt = int(k_opt)
    df_sample = df_grains[(df_grains['Sample_Clean'] == sample) & (df_grains['K_Model'] == k_opt)]
    
    for p in range(1, k_opt + 1):
        pop_data = df_sample[df_sample['Assigned_Pop'] == p]
        ages = pop_data['Age'].values
        sigmas_1s = pop_data['Sigma'].values 
        
        n_grains = len(ages)
        if n_grains > 1:
            mswd_val = calculate_mswd_robust(ages, sigmas_1s)
            
            try:
                stat, p_val = lilliefors(ages, dist='norm', pvalmethod='approx')
                is_gauss = "Yes" if p_val > 0.05 else "No"
            except:
                p_val, is_gauss = np.nan, "Error"
            
            diagnostic_list.append({
                "Sample": sample,
                "Pop": f"P{p}",
                "N": n_grains,
                "Mean": round(np.mean(ages), 1),
                "MSWD": round(mswd_val, 2),
                "Lilliefors_p": round(p_val, 3),
                "Normal?": is_gauss
            })

if diagnostic_list:
    df_f = pd.DataFrame(diagnostic_list)
    df_f.to_excel(output_stats, index=False)
    print(f"Listo che, archivo en: {output_stats}")
else:
    print("Sigue saliendo vacio... revisa los nombres en los Excel")
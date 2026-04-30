#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:12:07 2026

@author: marcosmorfulis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# -------- config --------
path = '/Users/marcosmorfulis/Desktop/'
sample_input = 'Age Distribution 4' 

# OPCIONES: 'Manual' o 'Optimum'
bin_mode = 'Optimum' 
bin_width_manual = 3.5 # solo se usa si pones 'Manual' arriba
# ------------------------

plt.rcParams.update({'font.size': 18})

# carga de excels
df_c = pd.read_excel(path + 'Classification_Results.xlsx')
df_v = pd.read_excel(path + 'Validation_Results_.xlsx')
df_g = pd.read_excel(path + 'Grain_Assignments_.xlsx')
df_s = pd.read_excel(path + 'Statistical_Diag_.xlsx')

# limpieza de nombres para que no rompa por un parentesis o espacio
def clean_name(df, col):
    df[col] = df[col].astype(str).str.split('(').str[0].str.strip().str.upper()

clean_name(df_c, 'Sample')
clean_name(df_v, 'Sample')
clean_name(df_g, 'Sample')
clean_name(df_s, 'Sample')

s_norm = sample_input.split('(')[0].strip().upper()

# busqueda de k (modelo seleccionado)
row_c = df_c[df_c['Sample'] == s_norm]
if row_c.empty:
    print(f"Error: no existe {s_norm} en Classification_Results.")
    exit()

k = int(row_c['N_Selected'].values[0])

# filtro de tablas por muestra y k
v = df_v[(df_v['Sample'] == s_norm) & (df_v['k'] == k)].iloc[0]
g = df_g[(df_g['Sample'] == s_norm) & (df_g['K_Model'] == k)]

ages = g['Age'].values
n_tot = len(ages)

# --- CALCULO DEL BIN ---
if bin_mode.lower() == 'optimum':
    # regla de Freedman-Diaconis: mas robusta para geocronologia
    iqr = np.percentile(ages, 75) - np.percentile(ages, 25)
    if iqr > 0:
        bin_width = 2 * iqr / (n_tot ** (1/3))
    else:
        bin_width = (max(ages) - min(ages)) / np.sqrt(n_tot) # fallback simple
    print(f"Bin optimo calculado: {bin_width:.2f} Ma")
else:
    bin_width = bin_width_manual

bins = np.arange(min(ages), max(ages) + bin_width, bin_width)

# separo poblaciones para el histograma
data_p = [g[g['Assigned_Pop'] == i]['Age'] for i in range(1, k+1)]

fig, ax = plt.subplots(figsize=(12,8))

# dibujo histograma stackeado
ax.hist(data_p, bins=bins, stacked=True, edgecolor='black', alpha=0.6)

# eje x para las gaussianas
x = np.linspace(min(ages)-10, max(ages)+10, 1000)

for i in range(1, k+1):
    mu = v[f'Pop{i}_Mean']
    w  = v[f'Pop{i}_Prop']
    
    # busco el error 2sigma del boot en la tabla de diagnostico
    s_row = df_s[(df_s['Sample'] == s_norm) & (df_s['Pop'] == f'P{i}')]
    err = s_row['SD_Boot'].values[0] * 2 if not s_row.empty else 0
    
    # desvio estandar de los granos de esta poblacion
    ages_i = g[g['Assigned_Pop'] == i]['Age'].values
    sd_emp = np.std(ages_i, ddof=1) if len(ages_i) > 1 else 0.5

    # dibujo la curva
    y = norm.pdf(x, mu, sd_emp) * n_tot * bin_width * w
    line, = ax.plot(x, y, '--', lw=3)
    col = line.get_color()

    # barras grises de incertidumbre (2 sigma boot)
    ax.axvspan(mu - err, mu + err, color='gray', alpha=0.25, zorder=0)
    ax.axvline(mu, color='gray', lw=2, alpha=0.6)

    # etiqueta con media y error
    ax.text(mu, max(y) * 1.05, f'{mu:.1f} ± {err:.1f}', 
            ha='center', va='bottom', fontsize=16, color=col, weight='bold',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax.set_title(f'{sample_input} (k={k})', fontsize=20)
ax.set_xlabel('Age (Ma)')
ax.set_ylabel('Frequency')
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()
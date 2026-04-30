import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import os
import matplotlib as mpl



### INPUTS necesarios: renombrar 1) BF_DIFFS_Obs_ 2) 4 archivos BF_DIFFS_N

# BLOQUE 1: Motor de procesamiento (Log BF) #######


def ejecutar_sistema_completo(base_path, escenarios, tamanos_n=[800, 120, 60, 40]):
    biblioteca = {}
    indices_test_por_escenario_n = {} 
    dfs_por_escenario = {} 
    
    # 1. Determinar dimension comun
    dims = [pd.read_excel(f'{base_path}BF_DIFFS_N{n}.xlsx', nrows=1)
                .select_dtypes(include=[np.number]).shape[1] for n in escenarios]
    dim_comun = min(dims)
    print(f"--> Analizando {dim_comun} Bayes Factors comunes con Biblioteca Especializada por n.")

    # 2. PRIMERA PASADA: Entrenamiento especializado por escenario(N) y n
    for n_escenario in escenarios:
        df_completo = pd.read_excel(f'{base_path}BF_DIFFS_N{n_escenario}.xlsx').iloc[:-1]
  # Extraer el n de la columna 'Sample' para filtrar
        df_completo['n_size_val'] = df_completo['Sample'].str.extract(r'\((\d+)\)').astype(float)
        dfs_por_escenario[n_escenario] = df_completo
        
        for tam_n in tamanos_n:
            df_especifico = df_completo[df_completo['n_size_val'] == tam_n]
            
            if df_especifico.empty:
                continue

            # separar Train/Test especifico para este par (N, n)
            df_train, df_test = train_test_split(df_especifico, test_size=0.3, random_state=42)
            indices_test_por_escenario_n[(n_escenario, tam_n)] = df_test.index
            
            # -CORRECCION: usar log(BF) ---
            data_train = np.log(df_train.select_dtypes(include=[np.number]).iloc[:, :dim_comun].values)
            
            if len(data_train) > 1:
                mu = np.mean(data_train, axis=0)
                # Agregamos regularizacion a la covarianza para evitar singularidad
                cov = np.cov(data_train, rowvar=False) + np.eye(dim_comun)*1e-6
                
                # La biblioteca ahora conoce el contexto del tamaño de muestra
                biblioteca[(n_escenario, tam_n)] = {
                    'mu': mu,
                    'dist': multivariate_normal(mean=mu, cov=cov, allow_singular=True)
                }

  #3. SEGUNDA PASADA: Clasificacion con discriminacipn por n
    resultados_totales = []
    for n_real in escenarios:
        df = dfs_por_escenario[n_real]
        # --- CORRECCIÓN: usar log(BF) ---
        data_all = np.log(df.select_dtypes(include=[np.number]).iloc[:, :dim_comun].values)
        
        for i, muestra_vec in enumerate(data_all):
            tam_n_actual = df.iloc[i]['n_size_val']
            
            # Competencia solo contra patrones del mismo tamaño de muestra
            probs = {}
            for n_ref in escenarios:
                if (n_ref, tam_n_actual) in biblioteca:
                    probs[n_ref] = biblioteca[(n_ref, tam_n_actual)]['dist'].pdf(muestra_vec)
                else:
                    probs[n_ref] = 0 
            
            n_detectado = max(probs, key=probs.get)
            es_test = df.index[i] in indices_test_por_escenario_n.get((n_real, tam_n_actual), [])
            
            resultados_totales.append({
                'Real': n_real,
                'Detectado': n_detectado,
                'Tamano_N': tam_n_actual,
                'Es_Test': es_test
            })
            
    return pd.DataFrame(resultados_totales), biblioteca, dim_comun

# BLOQUE 2: CONFIGURACION 
#

PATH = '/Users/marcosmorfulis/Desktop/' ###Directorio a utilizar CAMBIAR
ESC = [2, 3, 4, 5]
TAMS = [800, 120, 60, 40]

df_master, mi_biblioteca, dim_final = ejecutar_sistema_completo(PATH, ESC, TAMS)

# NUEVO BLOQUE: EXPORTAR  MATRICES ABSOLUTAS A EXCEL

archivo_matrices = os.path.join(PATH, 'Confusion_Matrix_abs_.xlsx')

with pd.ExcelWriter(archivo_matrices) as writer:
    for size in TAMS:
        # podmos extraemos solo los datos de test para este tamaño n
        subset_test = df_master[(df_master['Tamano_N'] == size) & (df_master['Es_Test'] == True)]
        
        if not subset_test.empty:
            # Calculamos matriz de confusion absoluta
            cm_abs = confusion_matrix(subset_test['Real'], subset_test['Detectado'], labels=ESC)
            
            # Convertimos a df para un formato limpio en Excel
            df_cm = pd.DataFrame(
                cm_abs, 
                index=[f'Real N={e}' for e in ESC], 
                columns=[f'Pred N={e}' for e in ESC]
            )
            
    # Guardamos cada n en una pestaña diferente:
            df_cm.to_excel(writer, sheet_name=f'n_{int(size)}')

print(f"--> Éxito: Matrices absolutas guardadas en: {archivo_matrices}")


# BLOQUE 3: VISUALIZACION

mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 9
mpl.rcParams['axes.titlesize'] = 10
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8

def plot_publication_matrices(df_master, TAMS, ESC, midpoint=55):
    import matplotlib.colors as mcolors
    
    fig, axes = plt.subplots(len(TAMS), 2, figsize=(9, 13), dpi=300)
    
    # Colormap personalizado: azul 'Francia' oscuro -> amarillo crema -> blanco
    colors = ["#002366", "#f1e4b3", "#ffffff"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    
    # Normalización con punto medio para contraste fuerte
    class MidpointNormalize(mcolors.Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            super().__init__(vmin, vmax, clip)
        def __call__(self, value, clip=None):
            result, is_scalar = self.process_value(value)
            vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint
            rescaled = np.where(result < midpoint,
                                0.5 * (result - vmin) / (midpoint - vmin),
                                0.5 + 0.5 * (result - midpoint) / (vmax - midpoint))
            return np.ma.masked_array(rescaled)
    
    norm = MidpointNormalize(vmin=0, vmax=100, midpoint=midpoint)
    
    # Fuentes y estilo:
    mpl.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9
    })
    
    for i, size in enumerate(TAMS):
        subset_all = df_master[df_master['Tamano_N'] == size]
        subset_test = subset_all[subset_all['Es_Test'] == True]
        
        # --- Matriz Consistencia Interna ---
        cm_all = confusion_matrix(subset_all['Real'], subset_all['Detectado'], labels=ESC)
        cm_all_perc = cm_all / cm_all.sum(axis=1)[:, None] * 100
        sns.heatmap(cm_all_perc, annot=True, fmt='.1f', cmap=cmap, norm=norm, ax=axes[i, 0],
                    xticklabels=ESC, yticklabels=ESC, cbar=True,
                    annot_kws={"size": 12, "weight": "bold"},  # números mas grandes
                    linewidths=1, linecolor='gray')  # delineado 
        axes[i, 0].set_title(f'Internal Consistency (n = {int(size)})', pad=12)
        
        # -- Matriz Validacion cruzada ---
        if not subset_test.empty:
            cm_test = confusion_matrix(subset_test['Real'], subset_test['Detectado'], labels=ESC)
            cm_test_perc = cm_test / cm_test.sum(axis=1)[:, None] * 100
            sns.heatmap(cm_test_perc, annot=True, fmt='.1f', cmap=cmap, norm=norm, ax=axes[i, 1],
                        xticklabels=ESC, yticklabels=ESC, cbar=True,
                        annot_kws={"size": 12, "weight": "bold"},
                        linewidths=1, linecolor='gray')
            axes[i, 1].set_title(f'Cross-Validation (n = {int(size)})', pad=12)

        axes[i, 0].set_ylabel('True N')
        axes[i, 1].set_ylabel('')
        if i == len(TAMS) - 1:
            axes[i, 0].set_xlabel('Predicted N')
            axes[i, 1].set_xlabel('Predicted N')

    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(PATH, 'Confusion_Matrices_.png'), dpi=600, bbox_inches='tight')
    plt.show()


plot_publication_matrices(df_master, TAMS, ESC) 


# BLOQUE 4 MODIFICADO: CLASIFICACION CON LOG-LIKELIHOOD


archivo_input = 'BF_DIFFS_Obs_.xlsx'
archivo_output = 'Classification_Results.xlsx'
ruta_input = os.path.join(PATH, archivo_input)

if os.path.exists(ruta_input):
    try:
        df_obs = pd.read_excel(ruta_input)
        df_obs['n_size_val'] = df_obs['Sample'].str.extract(r'\((\d+)\)').astype(float)
        data_numeric = np.log(df_obs.select_dtypes(include=[np.number]).iloc[:, :dim_final].values)
        predicciones = []
        
        for i in range(len(df_obs)):
            muestra_vec = data_numeric[i]
            tam_n_obs = df_obs.iloc[i]['n_size_val']
            n_disponibles = list(set([k[1] for k in mi_biblioteca.keys()]))
            tam_n_referencia = min(n_disponibles, key=lambda x:abs(x-tam_n_obs))
            
            # Probabilidades/ likelihoods para todas las categorias
            probs = {n: mi_biblioteca[(n, tam_n_referencia)]['dist'].pdf(muestra_vec) for n in ESC}
            sum_p = sum(probs.values())
            n_elegido = max(probs, key=probs.get)
            log_likelihood = -np.log(probs[n_elegido] + 1e-300)  # evitar log(0) NEGATIVO DE LOG LIKELIHOOD
            
            predicciones.append({
                'Sample': df_obs.iloc[i]['Sample'],
                'N_Selected': n_elegido,
                'Log_Likelihood': log_likelihood,
                'Confianza_%': (probs[n_elegido] / sum_p)*100 if sum_p > 0 else 0
            })
        
        df_final_obs = pd.DataFrame(predicciones)
        df_final_obs.to_excel(os.path.join(PATH, archivo_output), index=False)
        print(f"--> Éxito: Clasificadas {len(df_final_obs)} muestras observadas con Log-Likelihood incluido.")
    except Exception as e:
        print(f"Error en la clasificación final: {e}")
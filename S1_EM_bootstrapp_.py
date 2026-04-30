import numpy as np
import pandas as pd
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import warnings

# Silenciar avisos de Pandas p/ q consola este limpia
warnings.simplefilter(action='ignore', category=FutureWarning)

# 
# 1. CONFIGURACION DE RUTAS Y PARAMETROS
file_path = '/Users/marcosmorfulis/Desktop/age_distributions_input.xlsx' #Input-.xlsx'
sheet_name = 'Samples'

# Archivos de salida / output files
output_main = '/Users/marcosmorfulis/Desktop/Validation_Results_.xlsx'
output_bf = '/Users/marcosmorfulis/Desktop/BF_DIFFS_.xlsx'
output_grains = '/Users/marcosmorfulis/Desktop/Grain_Assignments_.xlsx'

N_BOOTSTRAP = 500   # This can be CHANGED 
MAX_K = 5                 
EM_TOL = 1e-5             
EM_MAX_ITER = 100         
N_CORES = 4  


# 2.MOTOR MATEMATICO

def run_heteroscedastic_em_vectorized(ages, sigmas, k, tol=EM_TOL, max_iter=EM_MAX_ITER):
    n = len(ages)
    weights_i = 1.0 / (sigmas**2 + 1e-12)
    t_js = np.percentile(ages, np.linspace(5, 95, k))
    pi_js = np.ones(k) / k
    log_likelihood = -np.inf
    ages_col = ages[:, np.newaxis]
    sigmas_col = sigmas[:, np.newaxis]
    
    for _ in range(max_iter):
        prev_ll = log_likelihood
        diff_sq = (ages_col - t_js)**2
        exponent = -0.5 * diff_sq / (sigmas_col**2)
        norm_factor = 1.0 / (sigmas_col * np.sqrt(2 * np.pi))
        densities = norm_factor * np.exp(exponent)
        weighted_densities = densities * pi_js
        row_sums = weighted_densities.sum(axis=1, keepdims=True)
        r = weighted_densities / (row_sums + 1e-20)
        
        nk = r.sum(axis=0)
        pi_js = nk / n
        numerator = np.sum(r * weights_i[:, np.newaxis] * ages_col, axis=0)
        denominator = np.sum(r * weights_i[:, np.newaxis], axis=0)
        t_js = numerator / (denominator + 1e-20)
        
        log_likelihood = np.sum(np.log(row_sums + 1e-20))
        if np.abs(log_likelihood - prev_ll) < tol:
            break
            
    return t_js, pi_js, log_likelihood

def bootstrap_worker(args):
    ages_orig, sigmas_orig, k, seed = args
    np.random.seed(seed)
    n = len(ages_orig)
    indices = np.random.randint(0, n, size=n)
    b_ages, b_sigmas = ages_orig[indices], sigmas_orig[indices]
    t_js, pi_js, ll = run_heteroscedastic_em_vectorized(b_ages, b_sigmas, k)
    order = np.argsort(t_js)
    return {"k": k, "means": t_js[order], "props": pi_js[order], "ll": ll}

def asignar_granos_a_poblaciones(ages, sigmas, t_js, pi_js):
    ages_col, sigmas_col = ages[:, np.newaxis], sigmas[:, np.newaxis]
    diff_sq = (ages_col - t_js)**2
    densities = (1.0 / (sigmas_col * np.sqrt(2 * np.pi))) * np.exp(-0.5 * diff_sq / (sigmas_col**2))
    weighted_probs = densities * pi_js
    cluster_assignment = np.argmax(weighted_probs, axis=1) + 1
    total_prob = weighted_probs.sum(axis=1, keepdims=True)
    confianza = (np.max(weighted_probs / (total_prob + 1e-20), axis=1)) * 100
    return cluster_assignment, confianza

# 
# 3. FLUJO PPAL.
 

def main():
    if multiprocessing.get_start_method(allow_none=True) != 'fork':
        try: multiprocessing.set_start_method('fork', force=True)
        except RuntimeError: pass

    print(f"--- Iniciando Procesamiento Completo (Robust Mode) ---")
    df_raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    
    results_summary, bf_data_list, grain_details_list = [], [], []

    for idx in range(0, len(df_raw), 2):
        try:
            sample_name = df_raw.iloc[idx, 0]
            if pd.isna(sample_name): continue
            
        #Limpieza
            ages_ser = df_raw.iloc[idx, 1:].replace([' ', 'None'], np.nan).infer_objects(copy=False)
            errs_ser = df_raw.iloc[idx + 1, 1:].replace([' ', 'None'], np.nan).infer_objects(copy=False)
            
            ages = ages_ser.dropna().astype(float).values
            errors_2s = errs_ser.dropna().astype(float).values
            
            min_len = min(len(ages), len(errors_2s))
            ages, sigmas = ages[:min_len], errors_2s[:min_len] / 2.0
            n_obs = len(ages)
            
            if n_obs < 5: continue
            print(f"\n>>> Procesando: {sample_name} ({n_obs} granos)...")
            start_time = time.time()
            
            sample_k_results = {}
            
            for k in range(2, MAX_K + 1):
                tasks = [(ages, sigmas, k, np.random.randint(0, 1000000)) for _ in range(N_BOOTSTRAP)]
                with ProcessPoolExecutor(max_workers=N_CORES) as executor:
                    res_boot = list(executor.map(bootstrap_worker, tasks))
                
        #Consolidar matrices de bootstrap para SD
                means_matrix = np.array([r['means'] for r in res_boot])
                props_matrix = np.array([r['props'] for r in res_boot])
                ll_vals = [r['ll'] for r in res_boot]
                
                m_avg, m_sd = np.mean(means_matrix, axis=0), np.std(means_matrix, axis=0)
                p_avg = np.mean(props_matrix, axis=0)
                
                # BIC medio para BayesFactor
                avg_bic_n = (-2 * np.mean(ll_vals) + (2*k-1) * np.log(n_obs)) / n_obs
                sample_k_results[k] = avg_bic_n
                
                #Asignacion de granos individual
                asignaciones, confianzas = asignar_granos_a_poblaciones(ages, sigmas, m_avg, p_avg)
                
                for i in range(len(ages)):
                    grain_details_list.append({
                        "Sample": sample_name,
                        "K_Model": k,
                        "Grain_ID": i + 1,
                        "Age": ages[i],
                        "Sigma": sigmas[i],
                        "Assigned_Pop": asignaciones[i],
                        "Confidence_%": confianzas[i]
                    })
                
                # Resumen de poblacion con SD (error de Bootstrap)
                row = {"Sample": sample_name, "k": k, "BIC_n": avg_bic_n, "N": n_obs}
                for j in range(k):
                    row[f"Pop{j+1}_Mean"] = m_avg[j]
                    row[f"Pop{j+1}_SD"] = m_sd[j]
                    row[f"Pop{j+1}_Prop"] = p_avg[j]
                results_summary.append(row)

            # Calculo de BayesFactor (comparacion entre modelos k)
            sample_bfs = [sample_name]
            for k in range(2, MAX_K):
                bf_val = np.exp((sample_k_results[k] - sample_k_results[k+1]) / 2.0)
                sample_bfs.append(bf_val)
            bf_data_list.append(sample_bfs)
            
            print(f"    Terminado en {time.time() - start_time:.2f} s.")

        except Exception as e:
            print(f"ERROR en {sample_name}: {e}")

    #
    # 4. EXPORTAR A EXCEL (los 3 Archivos)
    print(f"\n--- Guardando archivos en Desktop ---")
    
    # 1. resumen principal (conSD)
    pd.DataFrame(results_summary).to_excel(output_main, index=False)
    
    # 2.diferencias de Bayes Factor
    cols_bf = ['Sample'] + [f'BF_{k}to{k+1}' for k in range(2, MAX_K)]
    pd.DataFrame(bf_data_list, columns=cols_bf).to_excel(output_bf, index=False)
    
    # 3. Detalle por grano (Para los graficos de asignacin)
    pd.DataFrame(grain_details_list).to_excel(output_grains, index=False)

    print(f"Proceso completo. 3 archivos generados exitosamente.")

if __name__ == '__main__':
    main()
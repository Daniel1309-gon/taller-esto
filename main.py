import time
import pandas as pd
from data_cpts import create_telecom_network
from inference import rejection_sampling, likelihood_weighting, gibbs_sampling

def run_experiment(scenario_name, query_var, evidence, bn, sample_sizes):
    results = []
    
    print(f"\n--- Running Experiment: {scenario_name} ---")
    print(f"Query: P({query_var} | {evidence})")
    
    for N in sample_sizes:
        # 1. Rejection Sampling
        start = time.time()
        res_rej = rejection_sampling(query_var, evidence, bn, N)
        t_rej = time.time() - start
        
        # 2. Likelihood Weighting
        start = time.time()
        res_lw = likelihood_weighting(query_var, evidence, bn, N)
        t_lw = time.time() - start
        
        # 3. Gibbs Sampling
        start = time.time()
        res_gibbs = gibbs_sampling(query_var, evidence, bn, N)
        t_gibbs = time.time() - start
        
        results.append({
            'N': N,
            'Algorithm': 'Rejection',
            'P(True)': round(res_rej[1], 4),
            'Time(s)': round(t_rej, 4)
        })
        results.append({
            'N': N,
            'Algorithm': 'Likelihood',
            'P(True)': round(res_lw[1], 4),
            'Time(s)': round(t_lw, 4)
        })
        results.append({
            'N': N,
            'Algorithm': 'Gibbs',
            'P(True)': round(res_gibbs[1], 4),
            'Time(s)': round(t_gibbs, 4)
        })
        
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df

def main():
    bn = create_telecom_network()
    sample_sizes = [100, 1000, 10000]
    
    # Scenario 1: Evidencia única
    # P(Fallo_Router | Sin_Internet=True)
    run_experiment("Escenario 1 (Evidencia Única)", 
                   "Fallo_Router", 
                   {"Sin_Internet": 1}, 
                   bn, sample_sizes)
    
    # Scenario 2: Evidencia múltiple
    # P(Caida_DNS | Sin_Internet=True, Pagina_No_Carga=True)
    run_experiment("Escenario 2 (Evidencia Múltiple)", 
                   "Caida_DNS", 
                   {"Sin_Internet": 1, "Pagina_No_Carga": 1}, 
                   bn, sample_sizes)
    
    # Scenario 3: Causa rara
    # P(Sin_Internet | Clima_Severo=True)
    run_experiment("Escenario 3 (Causa Rara)", 
                   "Sin_Internet", 
                   {"Clima_Severo": 1}, 
                   bn, sample_sizes)

if __name__ == "__main__":
    main()

"""
main.py
=======
Punto de entrada del proyecto.  Construye la Red Bayesiana de telecomunicaciones
y ejecuta cinco escenarios de experimentación que comparan los tres algoritmos
de inferencia aproximada implementados en inference.py:

  1. Evidencia única          — comparación base de los tres algoritmos.
  2. Evidencia múltiple       — efecto de combinar varias observaciones.
  3. Causa rara               — impacto de evidencia con probabilidad prior baja.
  4. Efecto del burn-in       — sensibilidad de Gibbs al período de calentamiento.
  5. Evidencia extrema        — robustez ante evidencia combinada muy improbable
                                (demuestra el fallback de Rejection Sampling).

Para cada escenario se imprime una tabla con los resultados de cada algoritmo
en distintos tamaños de muestra N.

Referencia: Russell & Norvig, "Artificial Intelligence: A Modern Approach" (4ª ed.),
Sección 13.4, Figuras 13.17–13.22.
"""

import time

import pandas as pd
from data_cpts import create_telecom_network
from inference import rejection_sampling, likelihood_weighting, gibbs_sampling
from plotting import plot_burn_in_results, plot_experiment_results, plot_rare_evidence_results


# ──────────────────────────────────────────────────────────────────────────────
# Experimento estándar: compara los tres algoritmos para distintos N
# ──────────────────────────────────────────────────────────────────────────────

def run_experiment(scenario_name, query_var, evidence, bn, sample_sizes, num_runs):
    """
    Ejecuta los tres algoritmos de inferencia (Rejection, Likelihood Weighting
    y Gibbs) para una consulta P(query_var | evidence) con cada N en
    sample_sizes, mide el tiempo de ejecución y muestra los resultados en tabla.

    La columna 'P(True)' contiene P̂(query_var = True | evidence) estimada
    por cada algoritmo.  Con N grande, los tres deberían converger al mismo
    valor (consistencia de los estimadores Monte Carlo).

    Parámetros
    ----------
    scenario_name : str
        Etiqueta descriptiva del escenario, impresa como encabezado.
    query_var : str
        Variable cuya probabilidad posterior se estima.
    evidence : dict {str: int}
        Variables observadas y sus valores (0 = False, 1 = True).
    bn : BayesianNetwork
        Red Bayesiana construida con create_telecom_network().
    sample_sizes : list[int]
        Lista de valores de N a evaluar (p.ej. [100, 1000, 10000]).
    num_runs : int
        Número de corridas independientes por cada tamaño de muestra.

    Devuelve
    --------
    pd.DataFrame
        Tabla de resultados con columnas: Run, N, Algorithm, P(True), Time(s).
    """
    results = []

    print(f"\n--- Running Experiment: {scenario_name} ---")
    print(f"Query: P({query_var} | {evidence})")

    for run in range(1, num_runs + 1):
        for N in sample_sizes:

            # ── Rejection Sampling ────────────────────────────────────────────
            # Genera N muestras prior y descarta las inconsistentes con la evidencia.
            # Lento cuando P(e) es pequeña (pocas muestras pasan el filtro).
            start   = time.time()
            res_rej = rejection_sampling(query_var, evidence, bn, N)
            t_rej   = time.time() - start

            # ── Likelihood Weighting ──────────────────────────────────────────
            # Genera N muestras ponderadas; nunca descarta — todas contribuyen con
            # peso proporcional a la verosimilitud de la evidencia.
            start  = time.time()
            res_lw = likelihood_weighting(query_var, evidence, bn, N)
            t_lw   = time.time() - start

            # ── Gibbs Sampling (MCMC) ─────────────────────────────────────────
            # Cadena de Markov que cambia una variable a la vez muestreando de su
            # Manto de Markov.  Usa burn-in por defecto (N // 10 pasos descartados).
            start     = time.time()
            res_gibbs = gibbs_sampling(query_var, evidence, bn, N)
            t_gibbs   = time.time() - start

            # Acumular resultados de los tres algoritmos para este N
            results.append({'Run': run, 'N': N, 'Algorithm': 'Rejection',  'P(True)': round(res_rej[1],   4), 'Time(s)': round(t_rej,   4)})
            results.append({'Run': run, 'N': N, 'Algorithm': 'Likelihood', 'P(True)': round(res_lw[1],    4), 'Time(s)': round(t_lw,    4)})
            results.append({'Run': run, 'N': N, 'Algorithm': 'Gibbs',      'P(True)': round(res_gibbs[1], 4), 'Time(s)': round(t_gibbs, 4)})

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    summary = plot_experiment_results(scenario_name, df)
    print("\nResumen estadístico:")
    print(summary.to_string(index=False))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Escenario 4: Efecto del burn-in en Gibbs Sampling
# ──────────────────────────────────────────────────────────────────────────────

def run_burn_in_experiment(scenario_name, query_var, evidence, bn, N, burn_in_values, num_runs):
    """
    Evalúa cómo varía la estimación de Gibbs Sampling al cambiar el período
    de burn-in, manteniendo fijo el número de pasos de conteo (N).

    Motivación
    ----------
    La cadena de Markov parte de un estado inicial aleatorio y necesita un
    período de calentamiento (burn-in) para alcanzar la distribución estacionaria
    P(Z | e).  Con burn_in = 0 se cuentan pasos del transiente inicial, lo que
    introduce sesgo.  Con burn_in adecuado (típicamente 10–50 % de N), el sesgo
    desaparece y la estimación es más estable.

    Esta función permite observar esa transición: para burn_in pequeños los
    resultados serán más variables; para burn_in grandes, más precisos pero
    con mayor costo computacional total (N + burn_in pasos en total).

    Parámetros
    ----------
    scenario_name : str
        Etiqueta descriptiva del escenario.
    query_var : str
        Variable cuya probabilidad posterior se estima.
    evidence : dict {str: int}
        Variables observadas.
    bn : BayesianNetwork
        Red Bayesiana.
    N : int
        Número de pasos que se CUENTAN (constante en todos los runs).
    burn_in_values : list[int]
        Lista de períodos de burn-in a comparar.
    num_runs : int
        Número de corridas independientes por cada valor de burn-in.

    Devuelve
    --------
    pd.DataFrame
        Tabla con columnas: Run, Burn-in, N conteo, P(True), Time(s).
    """
    results = []

    print(f"\n--- {scenario_name} ---")
    print(f"Query: P({query_var} | {evidence})  |  N={N} pasos de conteo")

    for run in range(1, num_runs + 1):
        for burn_in in burn_in_values:
            start = time.time()
            # Se pasa burn_in explícitamente para sobreescribir el default (N//10)
            res = gibbs_sampling(query_var, evidence, bn, N, burn_in=burn_in)
            t   = time.time() - start

            results.append({
                'Run':      run,
                'Burn-in':  burn_in,
                'N conteo': N,
                'P(True)':  round(res[1], 4),
                'Time(s)':  round(t, 4),
            })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    summary = plot_burn_in_results(scenario_name, df)
    print("\nResumen estadístico:")
    print(summary.to_string(index=False))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Escenario 5: Evidencia extremadamente improbable
# ──────────────────────────────────────────────────────────────────────────────

def run_rare_evidence_experiment(scenario_name, query_var, evidence, bn, sample_sizes, num_runs):
    """
    Compara los tres algoritmos ante una combinación de evidencia que tiene
    probabilidad prior muy baja, evidenciando el fallback de Rejection Sampling.

    Problema que demuestra
    ----------------------
    Rejection Sampling es el único de los tres que puede quedarse sin muestras
    aceptadas cuando P(e) ≈ 0:
      - Antes del fix → devolvía {1: 0.0, 0: 0.0}, distribución inválida (suma 0).
      - Después del fix → devuelve {1: 0.5, 0: 0.5}, distribución uniforme que
        indica honestamente "no tengo información suficiente".

    Likelihood Weighting y Gibbs Sampling nunca sufren este problema:
      - LW nunca descarta muestras (todas tienen algún peso, aunque pequeño).
      - Gibbs trabaja directamente en el espacio posterior condicionado a la
        evidencia, sin generar muestras inconsistentes.

    Para N grande los tres convergen al mismo valor verdadero.

    Parámetros
    ----------
    scenario_name : str
        Etiqueta descriptiva del escenario.
    query_var : str
        Variable cuya probabilidad posterior se estima.
    evidence : dict {str: int}
        Evidencia combinada de alta improbabilidad prior.
    bn : BayesianNetwork
        Red Bayesiana.
    sample_sizes : list[int]
        Valores de N a evaluar (se usan N pequeños para ver el fallback).
    num_runs : int
        Número de corridas independientes por cada tamaño de muestra.

    Devuelve
    --------
    pd.DataFrame
        Tabla con columnas: Run, N, Algorithm, P(True), Time(s).
    """
    results = []

    print(f"\n--- {scenario_name} ---")
    print(f"Query: P({query_var} | {evidence})")
    print("  [Rejection puede devolver 0.5/0.5 para N pequeños por falta de muestras aceptadas]")

    for run in range(1, num_runs + 1):
        for N in sample_sizes:
            start = time.time()
            res_rej = rejection_sampling(query_var, evidence, bn, N)
            t_rej = time.time() - start

            start = time.time()
            res_lw = likelihood_weighting(query_var, evidence, bn, N)
            t_lw = time.time() - start

            start = time.time()
            res_gibbs = gibbs_sampling(query_var, evidence, bn, N)
            t_gibbs = time.time() - start

            results.append({'Run': run, 'N': N, 'Algorithm': 'Rejection',  'P(True)': round(res_rej[1],   4), 'Time(s)': round(t_rej,   4)})
            results.append({'Run': run, 'N': N, 'Algorithm': 'Likelihood', 'P(True)': round(res_lw[1],    4), 'Time(s)': round(t_lw,    4)})
            results.append({'Run': run, 'N': N, 'Algorithm': 'Gibbs',      'P(True)': round(res_gibbs[1], 4), 'Time(s)': round(t_gibbs, 4)})

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    summary = plot_rare_evidence_results(scenario_name, df)
    print("\nResumen estadístico:")
    print(summary.to_string(index=False))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Punto de entrada principal
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """
    Construye la red de telecomunicaciones y ejecuta los cinco escenarios
    de experimentación en secuencia.
    """
    # Construir la Red Bayesiana con todos sus nodos, aristas y CPTs
    bn = create_telecom_network()

    # Tamaños de muestra altos usados en los escenarios estándar.
    # A mayor N, menor variabilidad Monte Carlo, con mayor costo de ejecución.
    sample_sizes = [1000, 5000, 10000, 20000]
    num_runs = 30

    # ── Escenarios orburn_iginales ──────────────────────────────────────────────────

    # Escenario 1: Diagnóstico con una sola observación (evidencia upstream)
    # Pregunta: si el usuario reporta que no tiene internet, ¿cuál es la
    # probabilidad de que el router haya fallado?
    # Sin_Internet es hijo de Fallo_Router → evidencia downstream del router.
    run_experiment(
        "Escenario 1 (Evidencia Única)",
        query_var="Fallo_Router",
        evidence={"Sin_Internet": 1},
        bn=bn,
        sample_sizes=sample_sizes,
        num_runs=num_runs,
    )

    # Escenario 2: Diagnóstico con dos síntomas simultáneos
    # Pregunta: dado que el usuario no tiene internet Y las páginas no cargan,
    # ¿qué tan probable es que el DNS haya caído?
    # Ambas evidencias son hijos o nietos de Caida_DNS → diagnóstico "hacia arriba".
    run_experiment(
        "Escenario 2 (Evidencia Múltiple)",
        query_var="Caida_DNS",
        evidence={"Sin_Internet": 1, "Pagina_No_Carga": 1},
        bn=bn,
        sample_sizes=sample_sizes,
        num_runs=num_runs,
    )

    # Escenario 3: Propagación hacia abajo desde una causa raíz poco frecuente
    # Pregunta: si hay clima severo, ¿cuál es la probabilidad de perder internet?
    # Clima_Severo (P=0.05) es raíz → la evidencia es "upstream" de la consulta.
    # Rejection Sampling puede tener dificultades porque P(Clima=True)=0.05:
    # incluso con N alto, solo una fracción pequeña de muestras prior coincide.
    run_experiment(
        "Escenario 3 (Causa Rara)",
        query_var="Sin_Internet",
        evidence={"Clima_Severo": 1},
        bn=bn,
        sample_sizes=sample_sizes,
        num_runs=num_runs,
    )

    # ── Escenarios que ejercitan los cambios implementados ─────────────────────

    # Escenario 4: Impacto del burn-in en la precisión de Gibbs Sampling
    # Usa la misma consulta del Escenario 2 con N=10000 pasos de conteo fijos.
    # Se varía el burn-in desde 0 % hasta 50 % de N.
    # Resultado esperado: con burn_in=0 la estimación es más variable porque
    # incluye el transiente inicial; con burn_in ≥ 200 la estimación se estabiliza.
    run_burn_in_experiment(
        "Escenario 4 (Efecto del Burn-in en Gibbs)",
        query_var="Caida_DNS",
        evidence={"Sin_Internet": 1, "Pagina_No_Carga": 1},
        bn=bn,
        N=10000,
        burn_in_values=[0, 250, 500, 1000, 1500, 2000, 2500, 3000, 5000],
        num_runs=num_runs,
    )

    # Escenario 5: Evidencia combinada de probabilidad prior muy baja
    # P(Fallo_Router | Clima_Severo=T, Sin_Internet=T, Pagina_No_Carga=T)
    # La probabilidad prior de esta evidencia conjunta es muy pequeña porque
    # Clima_Severo=True solo ocurre el 5 % del tiempo.
    # Rejection Sampling puede aceptar pocas muestras aun con N alto, por lo que
    # sus estimaciones suelen ser más variables que Likelihood Weighting y Gibbs.
    run_rare_evidence_experiment(
        "Escenario 5 (Evidencia Extrema — Fix Rejection Sampling)",
        query_var="Fallo_Router",
        evidence={"Clima_Severo": 1, "Sin_Internet": 1, "Pagina_No_Carga": 1},
        bn=bn,
        sample_sizes=[1000, 5000, 10000, 20000],
        num_runs=num_runs,
    )


if __name__ == "__main__":
    main()

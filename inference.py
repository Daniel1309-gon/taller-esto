"""
inference.py
============
Implementa los tres algoritmos de inferencia aproximada (Monte Carlo) para
Redes Bayesianas descritos en el Capítulo 13 de Russell & Norvig:

  1. Muestreo por Rechazo   — Sección 13.4.1, Figura 13.17
  2. Ponderación por Verosimilitud — Sección 13.4.1, Figura 13.18 / Ec. 13.9
  3. Muestreo de Gibbs (MCMC)     — Sección 13.4.2, Figura 13.20 / Ec. 13.10

Todos los algoritmos trabajan con variables binarias (0 = False, 1 = True).
La función auxiliar get_prob_true() actúa como interfaz común de consulta
a las CPTs almacenadas en DataFrames de pandas.

Referencia: Russell & Norvig, "Artificial Intelligence: A Modern Approach" (4ª ed.).
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Función auxiliar: consulta de CPT
# ──────────────────────────────────────────────────────────────────────────────

def get_prob_true(node, sample):
    """
    Busca en la CPT del nodo la probabilidad de que sea True (1)
    dadas las asignaciones actuales de sus padres en 'sample'.

    Parámetros
    ----------
    node : BayesianNode
        Nodo cuya probabilidad se quiere consultar.
    sample : dict {str: int}
        Asignación parcial o completa de variables (nombre → 0/1).
        Debe contener los valores de TODOS los padres del nodo.

    Devuelve
    --------
    float
        P(node = True | padres según sample).

    Funcionamiento
    --------------
    - Si el nodo tiene padres: construye una consulta del tipo
      "Padre1 == val1 & Padre2 == val2 & ..." y usa DataFrame.query()
      para encontrar la fila correspondiente en la CPT.
    - Si el nodo es raíz (sin padres): devuelve directamente la única
      probabilidad almacenada en la CPT.

    Lanza
    -----
    ValueError si no existe ninguna fila que coincida con la combinación
    de valores de los padres (indica CPT incompleta).
    """
    cpt = node.cpt

    if node.parents:
        # Construye la cadena de consulta para pandas, p.ej.:
        # "Corte_Electrico == 1 & Mantenimiento_Programado == 0"
        query = " & ".join([f"{p.name} == {sample[p.name]}" for p in node.parents])

        matching_row = cpt.query(query)

        if matching_row.empty:
            raise ValueError(
                f"No se encontró entrada en la CPT de '{node.name}' "
                f"para la combinación de padres: {query}"
            )

        # .iloc[0] devuelve el escalar de la primera (y única) fila coincidente
        return matching_row['prob'].iloc[0]
    else:
        # Nodo raíz: su CPT solo tiene la columna 'prob' con un único valor
        return cpt['prob'].iloc[0]


# ──────────────────────────────────────────────────────────────────────────────
# Muestreo Prior (Ancestral Sampling)   —   Figura 13.16
# ──────────────────────────────────────────────────────────────────────────────

def prior_sample(bn):
    """
    Genera una muestra completa de la red mediante Muestreo Ancestral
    (también llamado Muestreo Prior o Prior Sampling).

    Algoritmo (Figura 13.16)
    ------------------------
    Para cada variable Xi en orden topológico:
        1. Leer P(Xi = True | padres(Xi)) usando las asignaciones ya hechas.
        2. Muestrear Xi ~ Bernoulli(p_true).

    El orden topológico garantiza que los padres de Xi ya tienen valor
    asignado cuando se procesa Xi, lo que hace válida la consulta a la CPT.

    Parámetros
    ----------
    bn : BayesianNetwork
        Red Bayesiana con nodos y CPTs ya definidos.

    Devuelve
    --------
    dict {str: int}
        Muestra completa: cada variable mapeada a 0 o 1.
        La probabilidad de cada muestra es P(x1, ..., xn) según la Ec. 13.2.
    """
    order  = bn.get_topological_order()
    sample = {}

    for node_name in order:
        node   = bn.get_node(node_name)
        p_true = get_prob_true(node, sample)

        # Muestreo de Bernoulli: si u < p_true → True (1), si no → False (0)
        # donde u ~ Uniforme[0,1)
        sample[node_name] = 1 if np.random.random() < p_true else 0

    return sample


# ──────────────────────────────────────────────────────────────────────────────
# Muestreo por Rechazo   —   Figura 13.17
# ──────────────────────────────────────────────────────────────────────────────

def rejection_sampling(X_name, e, bn, N):
    """
    Estima P(X | e) mediante Muestreo por Rechazo (Rejection Sampling).

    Idea (Figura 13.17)
    -------------------
    Genera N muestras con prior_sample().  Solo se conservan las que son
    consistentes con la evidencia 'e' (las demás se descartan, de ahí el nombre).
    La estimación final es la distribución de X en las muestras aceptadas:

        P̂(X | e) = α · N_PS(X, e)     donde α normaliza los conteos.

    La estimación es consistente: converge a la probabilidad verdadera cuando N → ∞.
    Sin embargo, si P(e) es pequeña, la mayoría de muestras se rechazan y la
    convergencia es muy lenta (principal debilidad del algoritmo).

    Caso borde (fix aplicado)
    -------------------------
    Si ninguna de las N muestras es consistente con e, retorna {1: 0.5, 0: 0.5}
    (distribución uniforme = máxima incertidumbre) en lugar de {1: 0.0, 0: 0.0},
    que sería una distribución inválida (no suma 1).

    Parámetros
    ----------
    X_name : str
        Nombre de la variable de consulta.
    e : dict {str: int}
        Evidencia observada: variables fijadas a sus valores (0 o 1).
    bn : BayesianNetwork
        Red Bayesiana.
    N : int
        Número total de muestras a generar (antes del rechazo).

    Devuelve
    --------
    dict {int: float}
        Distribución estimada {1: P̂(X=True|e), 0: P̂(X=False|e)}.
    """
    # Contadores de cuántas muestras aceptadas tienen X=1 y X=0
    counts = {1: 0, 0: 0}

    for _ in range(N):
        # Paso 1: generar muestra de la distribución prior
        sample = prior_sample(bn)

        # Paso 2: verificar consistencia con la evidencia
        # Se abandona el chequeo en cuanto se encuentra una discrepancia (short-circuit)
        consistent = all(sample[var] == val for var, val in e.items())

        # Paso 3: si es consistente, contar el valor de la variable de consulta
        if consistent:
            counts[sample[X_name]] += 1

    # Caso borde: ninguna muestra pasó el filtro → incertidumbre máxima
    total = sum(counts.values())
    if total == 0:
        return {1: 0.5, 0: 0.5}

    # Normalizar los conteos para obtener una distribución de probabilidad
    return normalize(counts)


# ──────────────────────────────────────────────────────────────────────────────
# Ponderación por Verosimilitud   —   Figura 13.18 / Ec. 13.9
# ──────────────────────────────────────────────────────────────────────────────

def weighted_sample(bn, e):
    """
    Genera una muestra ponderada consistente con la evidencia 'e'.

    Algoritmo (Figura 13.18 — WEIGHTED-SAMPLE)
    ------------------------------------------
    Recorre los nodos en orden topológico:
      - Variable de evidencia Ei con valor observado ei:
            Fijar sample[Ei] = ei
            Acumular el peso:  w ← w × P(Ei = ei | padres(Ei))
      - Variable no observada Zi:
            Muestrear normalmente: Zi ~ P(Zi | padres(Zi))

    El peso w al final equivale al producto de las verosimilitudes de la
    evidencia dada la muestra generada (Ec. 13.9):

        w(z) = α ∏_{i=1}^{m} P(ei | padres(Ei))

    Esto corrige el sesgo de fijar la evidencia: las muestras generadas no
    provienen de la distribución posterior verdadera, pero el peso w compensa
    esa desviación al sumarlos en likelihood_weighting().

    Parámetros
    ----------
    bn : BayesianNetwork
        Red Bayesiana.
    e : dict {str: int}
        Evidencia observada.

    Devuelve
    --------
    tuple (dict, float)
        (sample, w) donde sample es la asignación completa y w el peso.
    """
    order  = bn.get_topological_order()
    sample = {}
    w      = 1.0   # Peso inicial: sin evidencia, todas las muestras pesan igual

    for node_name in order:
        node   = bn.get_node(node_name)
        p_true = get_prob_true(node, sample)

        if node_name in e:
            # Variable de evidencia: fijar su valor y multiplicar el peso
            val            = e[node_name]
            sample[node_name] = val
            # Si el valor observado es True → multiplicar por p_true
            # Si es False → multiplicar por (1 - p_true) = P(nodo=False | padres)
            w *= p_true if val == 1 else (1 - p_true)
        else:
            # Variable no observada: muestrear de su distribución condicional
            sample[node_name] = 1 if np.random.random() < p_true else 0

    return sample, w


def likelihood_weighting(X_name, e, bn, N):
    """
    Estima P(X | e) mediante Ponderación por Verosimilitud (Likelihood Weighting).

    Algoritmo (Figura 13.18 — LIKELIHOOD-WEIGHTING)
    ------------------------------------------------
    Genera N muestras ponderadas con weighted_sample().
    Para cada muestra, acumula el peso w en el acumulador del valor de X:

        W[X = xi] += w

    Finalmente normaliza W para obtener la distribución estimada.

    Ventaja sobre Rejection Sampling
    ---------------------------------
    Nunca descarta muestras: todas contribuyen con un peso proporcional a
    cuán compatible es la muestra con la evidencia.  Esto lo hace mucho más
    eficiente cuando P(e) es pequeño.

    Limitación
    ----------
    Si la evidencia está "aguas abajo" (downstream) de las variables no observadas,
    la distribución de muestreo Q_WS ignora esa información y muchas muestras
    tendrán pesos muy bajos, degradando la precisión.  Gibbs Sampling maneja
    mejor este caso (véase la Figura 13.22 del libro).

    Parámetros
    ----------
    X_name : str
        Nombre de la variable de consulta.
    e : dict {str: int}
        Evidencia observada.
    bn : BayesianNetwork
        Red Bayesiana.
    N : int
        Número de muestras ponderadas a generar.

    Devuelve
    --------
    dict {int: float}
        Distribución estimada {1: P̂(X=True|e), 0: P̂(X=False|e)}.
    """
    # Acumuladores de peso para cada posible valor de X
    weights = {1: 0.0, 0: 0.0}

    for _ in range(N):
        sample, w = weighted_sample(bn, e)
        # El peso se acumula en el bin correspondiente al valor actual de X
        weights[sample[X_name]] += w

    # Normalizar los pesos acumulados para obtener probabilidades
    return normalize(weights)


# ──────────────────────────────────────────────────────────────────────────────
# Muestreo de Gibbs (MCMC)   —   Figura 13.20 / Ec. 13.10
# ──────────────────────────────────────────────────────────────────────────────

def get_markov_blanket_prob(node_name, sample, bn):
    """
    Calcula P(Xi = True | mb(Xi)) usando la Ecuación 13.10 del libro.

    El Manto de Markov de Xi (mb(Xi)) incluye: padres, hijos y co-padres
    (otros padres de sus hijos).  Por la propiedad de independencia local
    de las Redes Bayesianas, Xi es condicionalmente independiente de TODOS
    los demás nodos dado su Manto de Markov.

    Ecuación 13.10
    --------------
    P(xi | mb(Xi)) = α · P(xi | padres(Xi)) · ∏_{Yj ∈ Hijos(Xi)} P(yj | padres(Yj))

    donde:
      - P(xi | padres(Xi))           se consulta directamente en la CPT de Xi.
      - P(yj | padres(Yj))           se consulta en la CPT de cada hijo Yj,
                                     con los valores del sample actual (incluido
                                     el valor tentativo de Xi).
      - α                            es la constante de normalización calculada
                                     al evaluar la fórmula para xi=True y xi=False.

    Implementación
    --------------
    Para cada valor posible (val ∈ {0, 1}):
      1. Crear una copia temporal del sample con Xi fijado a 'val'.
      2. Calcular term1 = P(Xi = val | padres(Xi)).
      3. Calcular term2 = ∏ P(yj | padres(Yj))  para todos los hijos.
         La copia temporal garantiza que Xi aparezca con 'val' cuando
         los hijos lo consultan como padre.
      4. Resultado sin normalizar = term1 × term2.

    Finalmente se normaliza dividiendo por la suma de los dos resultados.

    Parámetros
    ----------
    node_name : str
        Nombre de la variable Xi a muestrear.
    sample : dict {str: int}
        Estado actual completo de la cadena de Markov.
    bn : BayesianNetwork
        Red Bayesiana.

    Devuelve
    --------
    float
        P(Xi = True | mb(Xi)) normalizada.  Si la suma es 0 (caso degenerado),
        devuelve 0.5 (máxima incertidumbre).
    """
    node = bn.get_node(node_name)

    def calculate_prob_for_value(val):
        """Calcula el numerador de la Ec. 13.10 para un valor concreto de Xi."""
        # Copia del estado con Xi fijado al valor 'val' a evaluar
        temp_sample = sample.copy()
        temp_sample[node_name] = val

        # Término 1: P(xi | padres(Xi))
        p_xi  = get_prob_true(node, temp_sample)
        term1 = p_xi if val == 1 else (1 - p_xi)

        # Término 2: producto de P(yj | padres(Yj)) para cada hijo Yj
        # temp_sample tiene Xi = val, lo que afecta a P(Yj | padres(Yj))
        # cuando Xi es padre de Yj.
        term2 = 1.0
        for child in node.children:
            p_y_true  = get_prob_true(child, temp_sample)
            child_val = temp_sample[child.name]   # valor actual del hijo en la cadena
            term2 *= p_y_true if child_val == 1 else (1 - p_y_true)

        return term1 * term2

    # Evaluar la fórmula para Xi=True y Xi=False (sin normalizar)
    p_true_unnorm  = calculate_prob_for_value(1)
    p_false_unnorm = calculate_prob_for_value(0)

    # Normalizar: α = 1 / (p_true_unnorm + p_false_unnorm)
    total = p_true_unnorm + p_false_unnorm
    return p_true_unnorm / total if total > 0 else 0.5


def gibbs_sampling(X_name, e, bn, N, burn_in=None):
    """
    Estima P(X | e) mediante Muestreo de Gibbs, un algoritmo MCMC
    (Markov Chain Monte Carlo).

    Algoritmo (Figura 13.20 — GIBBS-ASK)
    -------------------------------------
    1. Inicializar el estado:
       - Variables de evidencia: fijadas a sus valores observados (nunca cambian).
       - Variables no observadas: valores aleatorios 0/1.
    2. Durante burn_in + N iteraciones:
       a. Elegir aleatoriamente una variable no observada Zi.
       b. Muestrear un nuevo valor para Zi de su distribución de Manto de Markov:
              Zi ~ P(Zi | mb(Zi))    [Ec. 13.10]
       c. A partir del paso burn_in: contar el valor actual de X en el estado.
    3. Normalizar los conteos para obtener la distribución estimada.

    Por qué funciona (convergencia)
    --------------------------------
    La distribución estacionaria de la cadena de Markov definida por los
    pasos de Gibbs es exactamente P(Z | e) — la distribución posterior de
    las variables no observadas dada la evidencia.  Por lo tanto, cuando
    la cadena mezcla bien, los conteos de X convergen a P(X | e).
    (Demostración formal: Russell & Norvig, págs. 462–463, Ec. 13.11–13.13.)

    Burn-in (mejora añadida)
    ------------------------
    La cadena parte de un estado aleatorio y puede tardar varias iteraciones
    en alcanzar la distribución estacionaria (período transiente).  Descartar
    las primeras burn_in muestras evita que el transiente sesgue el resultado.
    Por defecto burn_in = max(1, N // 10), es decir el 10 % de N.

    Parámetros
    ----------
    X_name : str
        Nombre de la variable de consulta.
    e : dict {str: int}
        Evidencia observada: variables fijadas a sus valores (0 o 1).
    bn : BayesianNetwork
        Red Bayesiana.
    N : int
        Número de pasos a CONTAR (no incluye el burn-in).
    burn_in : int, opcional
        Pasos iniciales que se descartan antes de empezar a contar.
        Si es None se usa N // 10 (mínimo 1).

    Devuelve
    --------
    dict {int: float}
        Distribución estimada {1: P̂(X=True|e), 0: P̂(X=False|e)}.
    """
    # Valor por defecto del burn-in: 10 % de N, mínimo 1 paso
    if burn_in is None:
        burn_in = max(1, N // 10)

    counts = {1: 0, 0: 0}
    order  = bn.get_topological_order()

    # ── Inicializar el estado de la cadena de Markov ──────────────────────────
    sample = {}
    for node_name in order:
        if node_name in e:
            # Variables de evidencia: siempre fijas, nunca se muestrean
            sample[node_name] = e[node_name]
        else:
            # Variables libres: inicialización aleatoria uniforme
            sample[node_name] = np.random.choice([0, 1])

    # Lista de variables que la cadena puede modificar
    non_evidence = [name for name in order if name not in e]

    # Caso degenerado: todas las variables son evidencia → contar directamente
    if not non_evidence:
        counts[sample[X_name]] = N
        return normalize(counts)

    # ── Ejecutar la cadena de Markov (burn-in + conteo) ──────────────────────
    total_steps = N + burn_in
    for step in range(total_steps):

        # Seleccionar aleatoriamente una variable no observada
        Zi_name = np.random.choice(non_evidence)

        # Calcular P(Zi = True | mb(Zi)) usando la Ec. 13.10
        p_true = get_markov_blanket_prob(Zi_name, sample, bn)

        # Muestrear el nuevo valor de Zi y actualizar el estado
        sample[Zi_name] = 1 if np.random.random() < p_true else 0

        # Solo contar después de haber completado el período de burn-in
        if step >= burn_in:
            counts[sample[X_name]] += 1

    return normalize(counts)


# ──────────────────────────────────────────────────────────────────────────────
# Utilidad: normalización de distribuciones
# ──────────────────────────────────────────────────────────────────────────────

def normalize(dist):
    """
    Normaliza un diccionario {valor: peso} para que sus valores sumen 1.

    Si la suma de todos los pesos es 0 (caso degenerado sin muestras),
    devuelve el diccionario sin modificar para evitar división por cero.

    Parámetros
    ----------
    dist : dict {int: float}
        Distribución no normalizada (conteos o pesos acumulados).

    Devuelve
    --------
    dict {int: float}
        Distribución normalizada donde sum(valores) = 1.
    """
    total = sum(dist.values())
    if total == 0:
        return dist   # Sin muestras: no se puede normalizar
    return {k: v / total for k, v in dist.items()}

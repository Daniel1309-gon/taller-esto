"""
data_cpts.py
============
Construye la Red Bayesiana del dominio de diagnóstico de fallas en una red de
telecomunicaciones.  Define los nodos, las aristas causales y las Tablas de
Probabilidad Condicional (CPTs) de cada variable.

Topología del dominio
---------------------
Causas raíz (variables sin padres):
    Clima_Severo            — condiciones climáticas extremas
    Mantenimiento_Programado — ventana de mantenimiento activa

Fallas internas (variables intermedias):
    Corte_Electrico  — causado por Clima_Severo
    Caida_DNS        — causada por Corte_Electrico
    Fallo_Router     — causado por Corte_Electrico y Mantenimiento_Programado

Síntomas observables (posibles variables de evidencia):
    Latencia_Alta    — síntoma de Fallo_Router
    Sin_Internet     — síntoma de Fallo_Router y Caida_DNS
    Pagina_No_Carga  — síntoma de Caida_DNS y Latencia_Alta

Referencia: Russell & Norvig, "Artificial Intelligence: A Modern Approach" (4ª ed.),
Secciones 13.1–13.2.
"""

import pandas as pd
from models import BayesianNode, BayesianNetwork


def create_telecom_network():
    """
    Instancia y devuelve la Red Bayesiana completa del sistema de telecomunicaciones.

    El proceso sigue tres pasos:
      1. Crear los nodos (variables aleatorias binarias).
      2. Registrar los nodos en la red y definir las aristas causales.
      3. Asignar a cada nodo su CPT como DataFrame de pandas.

    Convención de CPTs
    ------------------
    - Todas las variables son binarias: 1 = True (ocurre), 0 = False (no ocurre).
    - La columna 'prob' almacena P(nodo = True | combinación de padres).
    - P(nodo = False | ...) = 1 − prob (no se almacena explícitamente).
    - Los nodos raíz solo tienen la columna 'prob' (probabilidad marginal).

    Devuelve
    --------
    BayesianNetwork
        Red completamente construida y lista para inferencia.
    """

    # ── 1. Crear los nodos ────────────────────────────────────────────────────

    clima          = BayesianNode("Clima_Severo")
    mantenimiento  = BayesianNode("Mantenimiento_Programado")
    corte          = BayesianNode("Corte_Electrico")
    dns            = BayesianNode("Caida_DNS")
    router         = BayesianNode("Fallo_Router")
    latencia       = BayesianNode("Latencia_Alta")
    internet       = BayesianNode("Sin_Internet")
    pagina         = BayesianNode("Pagina_No_Carga")

    # ── 2. Registrar nodos y definir la topología causal ─────────────────────

    bn = BayesianNetwork()
    for node in [clima, mantenimiento, corte, dns, router, latencia, internet, pagina]:
        bn.add_node(node)

    # Clima extremo puede provocar un corte eléctrico
    bn.add_edge("Clima_Severo",            "Corte_Electrico")

    # Un corte eléctrico puede derribar el servidor DNS o el router
    bn.add_edge("Corte_Electrico",         "Caida_DNS")
    bn.add_edge("Corte_Electrico",         "Fallo_Router")

    # El mantenimiento programado también puede causar falla del router
    bn.add_edge("Mantenimiento_Programado","Fallo_Router")

    # El fallo del router genera alta latencia y pérdida de internet
    bn.add_edge("Fallo_Router",            "Latencia_Alta")
    bn.add_edge("Fallo_Router",            "Sin_Internet")

    # La caída del DNS también provoca pérdida de internet y páginas que no cargan
    bn.add_edge("Caida_DNS",               "Sin_Internet")
    bn.add_edge("Caida_DNS",               "Pagina_No_Carga")

    # La latencia alta también puede impedir que las páginas carguen
    bn.add_edge("Latencia_Alta",           "Pagina_No_Carga")

    # ── 3. Definir las CPTs ───────────────────────────────────────────────────

    # ── Clima_Severo (raíz) ───────────────────────────────────────────────────
    # P(Clima_Severo = True) = 0.05
    # Evento infrecuente: condiciones extremas ocurren el 5 % del tiempo.
    clima.set_cpt(pd.DataFrame({
        'prob': [0.05]
    }))

    # ── Mantenimiento_Programado (raíz) ───────────────────────────────────────
    # P(Mantenimiento_Programado = True) = 0.10
    # Ventanas de mantenimiento representan el 10 % del tiempo operativo.
    mantenimiento.set_cpt(pd.DataFrame({
        'prob': [0.10]
    }))

    # ── Corte_Electrico | Clima_Severo ────────────────────────────────────────
    # P(Corte_Electrico = True | Clima_Severo = 1) = 0.80
    #   → Clima extremo causa corte en el 80 % de los casos.
    # P(Corte_Electrico = True | Clima_Severo = 0) = 0.01
    #   → Sin clima extremo, el corte es casi imposible (fallos de red eléctrica).
    corte.set_cpt(pd.DataFrame({
        'Clima_Severo': [1,    0],
        'prob':         [0.80, 0.01]
    }))

    # ── Caida_DNS | Corte_Electrico ───────────────────────────────────────────
    # P(Caida_DNS = True | Corte_Electrico = 1) = 0.30
    #   → Un corte puede derribar el DNS (servidores con UPS parcial).
    # P(Caida_DNS = True | Corte_Electrico = 0) = 0.02
    #   → El DNS puede caer espontáneamente por otras causas (software, etc.).
    dns.set_cpt(pd.DataFrame({
        'Corte_Electrico': [1,    0],
        'prob':            [0.30, 0.02]
    }))

    # ── Fallo_Router | Corte_Electrico, Mantenimiento_Programado ─────────────
    # La CPT tiene 2² = 4 filas (todas las combinaciones de los dos padres binarios).
    #
    # CE=1, M=1 → 0.95 : corte + mantenimiento simultáneos casi garantizan falla.
    # CE=1, M=0 → 0.50 : solo corte → falla probable (routers con respaldo parcial).
    # CE=0, M=1 → 0.80 : el mantenimiento por sí solo puede tumbar el router.
    # CE=0, M=0 → 0.01 : sin causas externas, falla espontánea muy rara.
    router.set_cpt(pd.DataFrame({
        'Corte_Electrico':          [1,    1,    0,    0],
        'Mantenimiento_Programado': [1,    0,    1,    0],
        'prob':                     [0.95, 0.50, 0.80, 0.01]
    }))

    # ── Latencia_Alta | Fallo_Router ─────────────────────────────────────────
    # P(Latencia_Alta = True | Fallo_Router = 1) = 0.90
    #   → Un router fallido casi siempre genera alta latencia.
    # P(Latencia_Alta = True | Fallo_Router = 0) = 0.05
    #   → Sin falla del router, la latencia alta puede deberse a congestión u otros.
    latencia.set_cpt(pd.DataFrame({
        'Fallo_Router': [1,    0],
        'prob':         [0.90, 0.05]
    }))

    # ── Sin_Internet | Fallo_Router, Caida_DNS ───────────────────────────────
    # FR=1, CD=1 → 0.99 : ambas causas juntas prácticamente garantizan corte total.
    # FR=1, CD=0 → 0.90 : solo falla del router → muy probable perder internet.
    # FR=0, CD=1 → 0.70 : solo caída DNS → probable sin internet (depende del ISP).
    # FR=0, CD=0 → 0.001: sin causas conocidas, pérdida de internet casi imposible.
    internet.set_cpt(pd.DataFrame({
        'Fallo_Router': [1,    1,    0,    0],
        'Caida_DNS':    [1,    0,    1,    0],
        'prob':         [0.99, 0.90, 0.70, 0.001]
    }))

    # ── Pagina_No_Carga | Caida_DNS, Latencia_Alta ───────────────────────────
    # CD=1, LA=1 → 0.99 : DNS caído + alta latencia → página casi nunca carga.
    # CD=1, LA=0 → 0.80 : solo DNS caído → la mayoría de páginas no resuelven.
    # CD=0, LA=1 → 0.60 : solo latencia alta → algunas páginas no cargan (timeout).
    # CD=0, LA=0 → 0.01 : sin causas, fallo de carga muy improbable.
    pagina.set_cpt(pd.DataFrame({
        'Caida_DNS':    [1,    1,    0,    0],
        'Latencia_Alta':[1,    0,    1,    0],
        'prob':         [0.99, 0.80, 0.60, 0.01]
    }))

    return bn


# ── Ejecución directa para verificar la red ───────────────────────────────────
if __name__ == "__main__":
    # Construye la red y muestra el orden topológico junto a cada CPT
    bn = create_telecom_network()
    order = bn.get_topological_order()

    print("Orden Topológico:", order)
    print("(Los padres siempre aparecen antes que sus hijos)\n")

    for name in order:
        node = bn.get_node(name)
        print(f"CPT de {name}:")
        print(node.cpt)
        print()

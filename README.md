# Inferencia Aproximada en Redes Bayesianas

Proyecto académico para el curso de Procesos Estocásticos — Universidad Nacional de Colombia.

Implementa una Red Bayesiana para diagnóstico de fallas en una red de telecomunicaciones y compara tres algoritmos de inferencia aproximada por muestreo Monte Carlo.

## Descripción

El sistema modela un dominio de 8 variables binarias que representan causas, fallas internas y síntomas observables de una red de telecomunicaciones:

```
Clima_Severo ──────────────────────────── Corte_Electrico ──┬── Caida_DNS ──┬── Sin_Internet
                                                             │               └── Pagina_No_Carga
Mantenimiento_Programado ────────────────── Fallo_Router ───┼── Latencia_Alta ─┘
                                                             └── Sin_Internet
```

Se implementan tres algoritmos del Capítulo 13 de *Artificial Intelligence: A Modern Approach* (Russell & Norvig, 4ª ed.):

| Algoritmo | Descripción | Fortaleza |
|-----------|-------------|-----------|
| **Rejection Sampling** | Genera muestras prior y descarta las inconsistentes con la evidencia | Simple, insesgado |
| **Likelihood Weighting** | Fija la evidencia y pondera cada muestra por su verosimilitud | Eficiente cuando P(e) es pequeña |
| **Gibbs Sampling (MCMC)** | Cadena de Markov que muestrea del Manto de Markov de cada variable | Robusto ante evidencia combinada |

## Requisitos

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/) (gestor de dependencias)

## Instalación

```powershell
# Instalar dependencias
uv sync
```

## Ejecución

```powershell
# Ejecutar los cinco escenarios de experimentación
uv run python main.py

# Verificar la topología de la red y sus CPTs
uv run python data_cpts.py
```

## Escenarios de Experimentación

El programa ejecuta cinco escenarios que comparan los algoritmos en distintas condiciones:

1. **Evidencia única** — `P(Fallo_Router | Sin_Internet=T)`: diagnóstico con una sola observación.
2. **Evidencia múltiple** — `P(Caida_DNS | Sin_Internet=T, Pagina_No_Carga=T)`: efecto de combinar dos síntomas.
3. **Causa rara** — `P(Sin_Internet | Clima_Severo=T)`: evidencia upstream con baja probabilidad prior (P=0.05).
4. **Efecto del burn-in** — variación del período de calentamiento en Gibbs Sampling (N fijo, burn-in variable).
5. **Evidencia extrema** — `P(Fallo_Router | Clima_Severo=T, Sin_Internet=T, Pagina_No_Carga=T)`: demuestra el fallback de Rejection Sampling cuando no acepta ninguna muestra.

Cada escenario muestra una tabla con P̂(variable=True | evidencia) y el tiempo de ejecución para N ∈ {100, 1000, 10000}.

## Estructura del Proyecto

```
proyecto/
├── models.py       # BayesianNode y BayesianNetwork (sin lógica de inferencia)
├── data_cpts.py    # Construcción de la red y definición de CPTs
├── inference.py    # Tres algoritmos de inferencia aproximada
├── main.py         # Punto de entrada: cinco escenarios de experimentación
└── pyproject.toml  # Dependencias del proyecto
```

## Dependencias

Solo se usan `numpy` y `pandas`. No se emplean librerías de alto nivel como `pgmpy`, `networkx` ni `scipy`; toda la lógica de grafos y muestreo está implementada desde cero.

## Referencia

Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4ª ed.). Secciones 13.1–13.4.

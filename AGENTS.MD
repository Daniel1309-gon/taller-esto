# Instrucciones de Desarrollo: Sistema de Diagnóstico de Redes con Redes Bayesianas

## 1. Contexto del Proyecto
Eres un agente de desarrollo experto en Python, algoritmos clásicos de Inteligencia Artificial y modelado probabilístico. 
El objetivo de este proyecto es construir un sistema de diagnóstico de fallos en redes de telecomunicaciones utilizando **Redes Bayesianas**, basado estrictamente en los algoritmos de inferencia aproximada del **Capítulo 13 del libro "Artificial Intelligence: A Modern Approach (4th ed.)" de Russell & Norvig**.

## 2. Restricciones ESTRICTAS de Desarrollo
- **Librerías Permitidas:** Solo Python nativo (built-ins), `numpy` (para operaciones matemáticas y generación aleatoria eficiente) y `pandas` (ESTRICTAMENTE solo para la manipulación, lectura y formateo de las Tablas de Probabilidad Condicional).
- **Librerías PROHIBIDAS:** `pgmpy`, `networkx`, `scipy`, `sklearn`, o cualquier otra librería de IA/Grafos/Probabilidad de terceros. Toda la lógica de grafos y muestreo DEBE ser implementada desde cero.

## 3. Uso de Herramientas del Entorno (MCP)
- **MCP context7:** Tienes acceso y **DEBES utilizar la herramienta MCP `context7`**. Utiliza esta herramienta para leer el contexto del proyecto, extraer las tablas de probabilidad (CPTs) iniciales (el INPUT), o consultar cualquier documentación de referencia que el usuario haya proveído en ese contexto. Antes de inventar datos, verifica si `context7` te puede proveer el INPUT base.

## 4. Dominio del Problema (Topología de la Red)
Vamos a modelar una red bayesiana para diagnosticar problemas de telecomunicaciones. Considera las siguientes variables booleanas (Nodos):
- **Causas Raíz (Nodos Padre):** `Clima_Severo`, `Mantenimiento_Programado`
- **Fallos Internos (Nodos Intermedios):** `Corte_Electrico`, `Caida_DNS`, `Fallo_Router`
- **Síntomas/Evidencia (Nodos Hoja):** `Latencia_Alta`, `Sin_Internet`, `Pagina_No_Carga`

## 5. Arquitectura del Software a Implementar
El código debe dividirse de forma modular. Por favor, implementa las siguientes clases y módulos:

### A. Estructura de Datos (Modelado del Grafo)
Crea una clase `BayesianNode` y una clase `BayesianNetwork`.
- `BayesianNode`: Debe almacenar su nombre, sus padres, sus hijos y su Tabla de Probabilidad Condicional (CPT).
- `BayesianNetwork`: Debe almacenar un diccionario de nodos y permitir agregar nodos respetando la topología de un Grafo Acíclico Dirigido (DAG). Debe incluir un método para procesar el orden topológico.

### B. Algoritmos de Inferencia (Módulo de Muestreo)
Debes implementar tres algoritmos de inferencia aproximada desde cero, basándote en el Capítulo 13:
1. **Muestreo por Descarte (Rejection Sampling):** Genera muestras previas de la red y descarta aquellas que no coinciden con la evidencia observada.
2. **Ponderación de Verosimilitud (Likelihood Weighting / Monte Carlo):** Genera muestras fijando los valores de la evidencia y pondera cada muestra por la probabilidad de la evidencia dados sus padres.
3. **Muestreo de Gibbs (Gibbs Sampling):** Inicializa la red con valores aleatorios, fija la evidencia y genera muestras cambiando el valor de una variable no observada a la vez, condicionada a su Manto de Markov (Markov Blanket).

## 6. Fases de Ejecución para el Agente

Por favor, ejecuta el desarrollo en los siguientes pasos interactivos. Pausa y pide mi confirmación al finalizar cada paso antes de continuar:

- **Paso 1:** Utiliza el **MCP `context7`** para consultar el contexto y/o recuperar las tablas de probabilidad (INPUT). Luego, define las estructuras base (`BayesianNode` y `BayesianNetwork`). Crea un script de prueba inicial donde se instancie la red y se carguen las CPTs usando `pandas` para formatearlas.
- **Paso 2:** Implementa la función generadora de muestras previas (Prior Sample) y el algoritmo de **Muestreo por Descarte**. Prueba que funcione con una consulta sencilla.
- **Paso 3:** Implementa el algoritmo de **Ponderación de Verosimilitud (Monte Carlo/Likelihood Weighting)**. Añade casos de prueba.
- **Paso 4:** Implementa el cálculo del Manto de Markov y el algoritmo de **Muestreo de Gibbs**.
- **Paso 5:** Crea un archivo `main.py` que configure 3 escenarios de prueba representativos (Ej: 1. Evidencia única, 2. Evidencia múltiple, 3. Causa rara). Ejecuta los 3 algoritmos en cada escenario con distintos tamaños de muestras (ej. 100, 1000, 10000) e imprime en consola una tabla (usando pandas si lo deseas para la visualización) comparando la precisión y el tiempo de ejecución de cada uno.

## 7. Salida Esperada
El código resultante debe estar fuertemente documentado, explicando qué ecuación o concepto matemático del libro se está aplicando en cada función.
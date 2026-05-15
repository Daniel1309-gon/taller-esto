"""
models.py
=========
Define las estructuras de datos fundamentales para representar una Red Bayesiana:
  - BayesianNode : un nodo del grafo con su Tabla de Probabilidad Condicional (CPT).
  - BayesianNetwork : el grafo dirigido acíclico (DAG) que los conecta.

Estas clases NO contienen lógica de inferencia; solo modelan la estructura
y las probabilidades del dominio.

Referencia: Russell & Norvig, "Artificial Intelligence: A Modern Approach" (4ª ed.),
Sección 13.1–13.2.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Nodo de la Red Bayesiana
# ──────────────────────────────────────────────────────────────────────────────

class BayesianNode:
    """
    Representa un nodo (variable aleatoria binaria) dentro de una Red Bayesiana.

    Atributos
    ---------
    name : str
        Identificador único del nodo; coincide con el nombre de la variable
        aleatoria en el dominio (p.ej. "Fallo_Router").
    parents : list[BayesianNode]
        Lista de nodos padre.  Un nodo sin padres es una variable raíz
        (causa originaria) cuya distribución marginal está en su CPT.
    children : list[BayesianNode]
        Lista de nodos hijo.  Necesaria para calcular la distribución
        del Manto de Markov durante el muestreo de Gibbs (Ec. 13.10).
    cpt : pd.DataFrame
        Tabla de Probabilidad Condicional.  Cada fila representa una
        combinación de valores de los padres y almacena P(nodo=True | padres).
        Para nodos raíz el DataFrame contiene solo la columna 'prob'.
    """

    def __init__(self, name):
        # Nombre de la variable aleatoria
        self.name = name

        # Relaciones en el DAG — se llenan llamando a BayesianNetwork.add_edge()
        self.parents = []
        self.children = []

        # CPT vacía hasta llamar a set_cpt()
        self.cpt = None

    def set_cpt(self, cpt_data):
        """
        Asigna la Tabla de Probabilidad Condicional al nodo.

        Parámetros
        ----------
        cpt_data : pd.DataFrame
            DataFrame cuyas columnas son los nombres de los nodos padre
            (con valores 0/1) más la columna 'prob', que contiene
            P(nodo=True | combinación de padres).

            Ejemplo para un nodo con un padre binario P:
                P  | prob
                1  | 0.8      ← P(nodo=True | P=True)  = 0.8
                0  | 0.05     ← P(nodo=True | P=False) = 0.05

            Para un nodo raíz (sin padres):
                prob
                0.1           ← P(nodo=True) = 0.1
        """
        self.cpt = cpt_data

    def __repr__(self):
        return f"Node({self.name})"


# ──────────────────────────────────────────────────────────────────────────────
# Red Bayesiana (DAG)
# ──────────────────────────────────────────────────────────────────────────────

class BayesianNetwork:
    """
    Representa una Red Bayesiana como un Grafo Dirigido Acíclico (DAG).

    Responsabilidades
    -----------------
    - Mantener el registro de todos los nodos (variables del dominio).
    - Añadir aristas dirigidas que codifican dependencias causales.
    - Calcular un orden topológico válido, requerido por todos los
      algoritmos de muestreo para garantizar que los padres estén
      asignados antes que sus hijos.

    Referencia: Russell & Norvig, Sección 13.1 (Figura 13.2).
    """

    def __init__(self):
        # Diccionario nombre → BayesianNode; conserva el orden de inserción
        self.nodes = {}

    def add_node(self, node):
        """
        Registra un nodo en la red.
        Si el nodo ya existe (mismo nombre) se ignora para evitar duplicados.
        """
        if node.name not in self.nodes:
            self.nodes[node.name] = node

    def add_edge(self, parent_name, child_name):
        """
        Añade una arista dirigida  padre → hijo.

        Esta arista modela que el padre influye directamente en la
        distribución del hijo (P(hijo | padres)).  Internamente actualiza
        las listas 'children' del padre y 'parents' del hijo para que
        los algoritmos de inferencia puedan recorrer el grafo en ambas
        direcciones.

        Parámetros
        ----------
        parent_name : str
            Nombre del nodo origen de la arista.
        child_name : str
            Nombre del nodo destino de la arista.

        Lanza
        -----
        ValueError si alguno de los dos nodos no ha sido registrado antes.
        """
        if parent_name not in self.nodes or child_name not in self.nodes:
            raise ValueError("Ambos nodos deben existir en la red antes de añadir la arista.")

        parent = self.nodes[parent_name]
        child  = self.nodes[child_name]

        # Evitar aristas duplicadas
        if child not in parent.children:
            parent.children.append(child)
        if parent not in child.parents:
            child.parents.append(parent)

    def get_topological_order(self):
        """
        Devuelve una lista con los nombres de los nodos en orden topológico
        usando el algoritmo de Kahn (basado en grados de entrada).

        El orden topológico garantiza que, al recorrer la lista de izquierda
        a derecha, cada nodo aparece DESPUÉS de todos sus padres.  Esto es
        esencial para el muestreo ancestral (prior_sample) y para
        weighted_sample: al llegar a un nodo, los valores de sus padres
        ya han sido asignados en el diccionario 'sample'.

        Referencia: Russell & Norvig, Sección 13.4.1 (Figura 13.16).

        Devuelve
        --------
        list[str]
            Nombres de los nodos en un orden topológico válido.

        Lanza
        -----
        ValueError si el grafo contiene un ciclo (no es un DAG).
        """
        # Calcula el grado de entrada (número de padres) de cada nodo
        in_degree = {name: len(node.parents) for name, node in self.nodes.items()}

        # Cola inicial: nodos sin padres (grado de entrada 0 = variables raíz)
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []

        while queue:
            # Extrae un nodo con grado 0 y lo añade al orden
            u_name = queue.pop(0)
            order.append(u_name)

            # Reduce el grado de entrada de sus hijos; si llegan a 0 entran a la cola
            for child in self.nodes[u_name].children:
                in_degree[child.name] -= 1
                if in_degree[child.name] == 0:
                    queue.append(child.name)

        # Si no se procesaron todos los nodos existe un ciclo en el grafo
        if len(order) != len(self.nodes):
            raise ValueError("El grafo contiene un ciclo; no es un DAG válido.")

        return order

    def get_node(self, name):
        """Devuelve el BayesianNode con el nombre dado, o None si no existe."""
        return self.nodes.get(name)

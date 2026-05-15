# Project Instructions: Network Diagnosis System with Bayesian Networks

## Project Overview
This project involves building a telecommunications network failure diagnosis system using **Bayesian Networks**. The implementation must strictly follow the approximate inference algorithms described in **Chapter 13 of "Artificial Intelligence: A Modern Approach (4th ed.)" by Russell & Norvig**.

### Core Technologies & Constraints
- **Language:** Python 3.x
- **Math & Randomness:** `numpy` (for efficient array operations and random sampling).
- **Data Handling:** `pandas` (strictly for reading, manipulating, and displaying CPTs).
- **STRICT PROHIBITION:** Do NOT use `pgmpy`, `networkx`, `scipy`, `sklearn`, or any other high-level AI/Graph/Probabilistic libraries. All logic for graph management and sampling must be implemented from scratch.

## Implementation Details

### 1. Data Structures (`BayesianNode`, `BayesianNetwork`)
- **`BayesianNode`**: Must store its name, list of parents, list of children, and its Conditional Probability Table (CPT).
- **`BayesianNetwork`**:
  - Manages the collection of nodes.
  - Ensures the graph is a Directed Acyclic Graph (DAG).
  - Implements **topological ordering** (essential for sampling).

### 2. Inference Algorithms (Approximate Inference)
You must implement the following three algorithms from scratch:

#### A. Rejection Sampling
1.  **`PriorSample(bn)`**: Generates a sample $[x_1, \dots, x_n]$ by sampling each variable in topological order, conditioned on its parents' values.
2.  **`RejectionSampling(X, e, bn, N)`**:
    - Call `PriorSample`.
    - If the sample matches the evidence $e$, keep it; otherwise, discard it.
    - Estimate $\hat{P}(X|e)$ by normalizing the counts of $X$'s values in the accepted samples.

#### B. Likelihood Weighting
1.  **`WeightedSample(bn, e)`**:
    - Initialize weight $w = 1.0$.
    - For each variable in topological order:
      - If it is an evidence variable $E_i$ with value $e_i$: $w \leftarrow w \times P(e_i | parents(E_i))$.
      - Else: sample its value from $P(X_i | parents(X_i))$.
    - Return the sample and its weight $w$.
2.  **`LikelihoodWeighting(X, e, bn, N)`**:
    - Accumulate the weights $w$ for each value of the query variable $X$.
    - Normalize the resulting distribution.

#### C. Gibbs Sampling (MCMC)
1.  **`GibbsSampling(X, e, bn, N)`**:
    - Initialize the state with random values for non-evidence variables, fixing evidence $e$.
    - In each iteration, pick a non-evidence variable $X_i$.
    - Sample its new value from its **Markov Blanket** distribution:
      $P(x_i | mb(X_i)) = \alpha P(x_i | parents(X_i)) \prod_{Y_j \in Children(X_i)} P(y_j | parents(Y_j))$
    - Estimate $\hat{P}(X|e)$ by counting the values of $X$ visited during the walk and normalizing.

## Domain Topology (Network Diagnosis)
- **Root Causes:** `Clima_Severo`, `Mantenimiento_Programado`
- **Internal Failures:** `Corte_Electrico`, `Caida_DNS`, `Fallo_Router`
- **Symptoms (Evidence):** `Latencia_Alta`, `Sin_Internet`, `Pagina_No_Carga`

## Execution Plan
1.  **Phase 1: Modeling**: Create classes and load CPTs (using `pandas`). Verify topological order.
2.  **Phase 2: Rejection & Likelihood**: Implement and test both algorithms.
3.  **Phase 3: Gibbs**: Implement Markov Blanket logic and the Gibbs sampler.
4.  **Phase 4: Evaluation**: Use `main.py` to compare precision and time across algorithms for $N \in \{100, 1000, 10000\}$ in multiple scenarios.

## Documentation
Each function must be documented with its corresponding mathematical formula or concept from Russell & Norvig's Chapter 13.

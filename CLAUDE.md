# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telecommunications network failure diagnosis system using Bayesian Networks with approximate inference. Implements algorithms from Chapter 13 of *Artificial Intelligence: A Modern Approach (4th ed.)* by Russell & Norvig. Academic project for a Stochastic Processes course at Universidad Nacional de Colombia.

## Commands

This project uses `uv` for dependency management (Python 3.13+).

```powershell
# Run the main experiment
uv run python main.py

# Verify network topology and CPTs
uv run python data_cpts.py

# Install dependencies
uv sync
```

## Architecture

The code is split across three modules plus an entry point:

- **`models.py`** — Core data structures: `BayesianNode` (name, parents, children, CPT as a `pd.DataFrame`) and `BayesianNetwork` (node registry + topological sort via Kahn's algorithm). No inference logic here.

- **`data_cpts.py`** — Defines the telecom network: 8 binary nodes (`Clima_Severo`, `Mantenimiento_Programado`, `Corte_Electrico`, `Caida_DNS`, `Fallo_Router`, `Latencia_Alta`, `Sin_Internet`, `Pagina_No_Carga`) with their CPTs as DataFrames. Root causes → internal failures → observable symptoms.

- **`inference.py`** — Three approximate inference algorithms, all operating on binary variables (0/1):
  - `rejection_sampling`: prior-samples then discards inconsistent samples
  - `likelihood_weighting`: fixes evidence and accumulates weights
  - `gibbs_sampling`: MCMC walk sampling from Markov Blanket distribution (`get_markov_blanket_prob`)
  - Shared helper `get_prob_true(node, sample)` looks up CPT rows via `DataFrame.query()`

- **`main.py`** — Runs three query scenarios at N ∈ {100, 1000, 10000} samples, comparing all three algorithms on precision and wall-clock time. Output is a pandas DataFrame table per scenario.

## Hard Constraints

**Do NOT use** `pgmpy`, `networkx`, `scipy`, `sklearn`, or any other high-level probabilistic/graph library. All graph management and sampling logic must remain implemented from scratch using only `numpy` and `pandas`.

## CPT Convention

All nodes are binary. CPT DataFrames use columns named after parent nodes (values 0 or 1) plus a `prob` column storing P(node=True | parent combination). `get_prob_true` queries these DataFrames using pandas `.query()` string syntax.

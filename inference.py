import numpy as np
import pandas as pd

def get_prob_true(node, sample):
    """
    Looks up the probability of node being True given the parent values in the sample.
    """
    cpt = node.cpt
    if node.parents:
        # Build query for the CPT
        query = " & ".join([f"{p.name} == {sample[p.name]}" for p in node.parents])
        # Find the matching row
        matching_row = cpt.query(query)
        if matching_row.empty:
            raise ValueError(f"No CPT entry found for {node.name} with parents {query}")
        return matching_row['prob'].iloc[0]
    else:
        # Root node
        return cpt['prob'].iloc[0]

def prior_sample(bn):
    """
    Generates a single sample from the network using prior sampling (Ancestral Sampling).
    """
    order = bn.get_topological_order()
    sample = {}
    
    for node_name in order:
        node = bn.get_node(node_name)
        p_true = get_prob_true(node, sample)
        # Sample True (1) or False (0)
        sample[node_name] = 1 if np.random.random() < p_true else 0
        
    return sample

def rejection_sampling(X_name, e, bn, N):
    """
    Estimates P(X|e) using Rejection Sampling.
    X_name: Name of the query variable.
    e: Evidence dictionary {node_name: value}.
    bn: BayesianNetwork object.
    N: Total number of samples to generate.
    """
    counts = {1: 0, 0: 0} # Counts for True and False
    
    for _ in range(N):
        sample = prior_sample(bn)
        
        # Check if sample is consistent with evidence
        consistent = True
        for node_name, val in e.items():
            if sample[node_name] != val:
                consistent = False
                break
        
        if consistent:
            counts[sample[X_name]] += 1
            
    total = sum(counts.values())
    if total == 0:
        return {1: 0.5, 0: 0.5}  # No consistent samples: uniform (maximum uncertainty)

    return normalize(counts)

def weighted_sample(bn, e):
    """
    Generates a sample and its weight based on evidence e.
    """
    order = bn.get_topological_order()
    sample = {}
    w = 1.0
    
    for node_name in order:
        node = bn.get_node(node_name)
        p_true = get_prob_true(node, sample)
        
        if node_name in e:
            # Evidence variable: fix value and update weight
            val = e[node_name]
            sample[node_name] = val
            # If evidence is True, weight *= p_true, else weight *= (1 - p_true)
            w *= p_true if val == 1 else (1 - p_true)
        else:
            # Non-evidence variable: sample as usual
            sample[node_name] = 1 if np.random.random() < p_true else 0
            
    return sample, w

def likelihood_weighting(X_name, e, bn, N):
    """
    Estimates P(X|e) using Likelihood Weighting.
    """
    weights = {1: 0.0, 0: 0.0}
    
    for _ in range(N):
        sample, w = weighted_sample(bn, e)
        weights[sample[X_name]] += w
        
    return normalize(weights)

def get_markov_blanket_prob(node_name, sample, bn):
    """
    Calculates P(X_i = True | mb(X_i)) using the formula from Equation 13.10.
    P(x_i | mb(X_i)) = alpha * P(x_i | parents(X_i)) * product(P(y_j | parents(Y_j)))
    """
    node = bn.get_node(node_name)
    
    def calculate_prob_for_value(val):
        temp_sample = sample.copy()
        temp_sample[node_name] = val
        
        # 1. P(x_i | parents(X_i))
        p_xi = get_prob_true(node, temp_sample)
        term1 = p_xi if val == 1 else (1 - p_xi)
        
        # 2. product(P(y_j | parents(Y_j)))
        term2 = 1.0
        for child in node.children:
            p_y_true = get_prob_true(child, temp_sample)
            child_val = temp_sample[child.name]
            term2 *= p_y_true if child_val == 1 else (1 - p_y_true)
            
        return term1 * term2

    p_true_unnorm = calculate_prob_for_value(1)
    p_false_unnorm = calculate_prob_for_value(0)
    
    # Normalize to get alpha
    total = p_true_unnorm + p_false_unnorm
    return p_true_unnorm / total if total > 0 else 0.5

def gibbs_sampling(X_name, e, bn, N):
    """
    Estimates P(X|e) using Gibbs Sampling (MCMC).
    """
    counts = {1: 0, 0: 0}
    order = bn.get_topological_order()
    
    # Initialize sample with evidence fixed and others random
    sample = {}
    for node_name in order:
        if node_name in e:
            sample[node_name] = e[node_name]
        else:
            sample[node_name] = np.random.choice([0, 1])
            
    # List of non-evidence variables to sample
    non_evidence = [name for name in order if name not in e]
    
    if not non_evidence:
        # All variables are evidence!
        counts[sample[X_name]] = N
        return normalize(counts)

    for _ in range(N):
        # Pick a random non-evidence variable
        Zi_name = np.random.choice(non_evidence)
        
        # Sample new value from Markov Blanket distribution
        p_true = get_markov_blanket_prob(Zi_name, sample, bn)
        sample[Zi_name] = 1 if np.random.random() < p_true else 0
        
        # Count current state of X
        counts[sample[X_name]] += 1
        
    return normalize(counts)

def normalize(dist):
    """Normalizes a dictionary of {val: weight}."""
    total = sum(dist.values())
    if total == 0: return dist
    return {k: v / total for k, v in dist.items()}

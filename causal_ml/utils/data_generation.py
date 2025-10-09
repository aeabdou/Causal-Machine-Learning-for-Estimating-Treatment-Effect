"""Data generation utilities for causal inference pipelines."""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def generate_synthetic_data(
    n_samples: int = 10000,
    n_features: int = 10,
    treatment_effect_heterogeneity: bool = True,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Generate synthetic data for causal inference with heterogeneous treatment effects.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    treatment_effect_heterogeneity : bool
        Whether to include heterogeneous treatment effects
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    T : pd.Series
        Treatment assignment (binary)
    Y : pd.Series
        Observed outcomes
    tau : pd.Series
        True individual treatment effects (for evaluation)
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])
    
    # Generate propensity score (probability of treatment)
    # Depends on X0 and X1
    propensity = 1 / (1 + np.exp(-(0.5 * X[:, 0] + 0.3 * X[:, 1])))
    propensity = np.clip(propensity, 0.1, 0.9)  # Ensure overlap
    
    # Generate treatment assignment
    T = np.random.binomial(1, propensity)
    
    # Generate baseline outcomes (depends on features)
    baseline = (2 * X[:, 0] + X[:, 1] + 0.5 * X[:, 2] + 
                0.3 * X[:, 0] * X[:, 1])
    
    # Generate heterogeneous treatment effects
    if treatment_effect_heterogeneity:
        # Treatment effect depends on X0 and X3
        # High effects for X0 > 0 and X3 > 0
        tau = 5 + 3 * X[:, 0] + 2 * X[:, 3] + X[:, 0] * X[:, 3]
    else:
        # Constant treatment effect
        tau = np.ones(n_samples) * 5.0
    
    # Generate observed outcomes
    # Y = baseline + treatment_effect * T + noise
    noise = np.random.randn(n_samples)
    Y = baseline + tau * T + noise
    
    return (
        X_df,
        pd.Series(T, name='treatment'),
        pd.Series(Y, name='outcome'),
        pd.Series(tau, name='true_effect')
    )


def create_policy_relevant_data(
    n_samples: int = 100000,
    random_state: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Generate synthetic data that mimics a policy intervention scenario.
    
    This simulates a scenario where policymakers want to target households
    with the highest benefit from an intervention.
    
    Parameters
    ----------
    n_samples : int
        Number of households (default: 100,000)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    X : pd.DataFrame
        Household characteristics
    T : pd.Series
        Treatment assignment (program enrollment)
    Y : pd.Series
        Outcome (e.g., income, welfare improvement)
    tau : pd.Series
        True individual treatment effects
    """
    np.random.seed(random_state)
    
    # Generate household characteristics
    income = np.random.lognormal(mean=10, sigma=0.8, size=n_samples)
    education = np.random.randint(0, 20, size=n_samples)
    household_size = np.random.poisson(lam=3, size=n_samples) + 1
    age = np.random.randint(18, 75, size=n_samples)
    urban = np.random.binomial(1, 0.6, size=n_samples)
    
    # Additional features
    employment_status = np.random.binomial(1, 0.7, size=n_samples)
    health_index = np.random.beta(2, 2, size=n_samples) * 100
    
    X = pd.DataFrame({
        'income': income,
        'education': education,
        'household_size': household_size,
        'age': age,
        'urban': urban,
        'employment_status': employment_status,
        'health_index': health_index
    })
    
    # Normalize features for propensity model
    X_norm = (X - X.mean()) / X.std()
    
    # Propensity score (probability of receiving treatment)
    # Lower-income, less educated households more likely to be treated
    propensity_logit = (-0.5 * X_norm['income'] + 
                        0.3 * X_norm['education'] + 
                        0.2 * X_norm['household_size'])
    propensity = 1 / (1 + np.exp(-propensity_logit))
    propensity = np.clip(propensity, 0.05, 0.95)
    
    # Treatment assignment
    T = np.random.binomial(1, propensity)
    
    # Heterogeneous treatment effects
    # Higher effects for:
    # - Lower income households
    # - Higher education (can better utilize the program)
    # - Larger households
    tau = (5000 - 0.1 * income + 
           200 * education + 
           500 * household_size +
           20 * age * (age < 50).astype(int))  # Younger households benefit more
    
    # Add some non-linearity
    tau = tau + 1000 * (education > 12).astype(int) * (income < np.median(income)).astype(int)
    
    # Baseline outcome
    baseline = (0.5 * income + 
                500 * education + 
                300 * household_size + 
                50 * employment_status)
    
    # Observed outcome
    noise = np.random.randn(n_samples) * 2000
    Y = baseline + tau * T + noise
    
    return (
        X,
        pd.Series(T, name='treatment'),
        pd.Series(Y, name='outcome'),
        pd.Series(tau, name='true_effect')
    )

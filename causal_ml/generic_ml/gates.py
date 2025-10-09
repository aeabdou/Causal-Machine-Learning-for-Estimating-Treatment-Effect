"""
GATES: Group Average Treatment Effects

GATES analyzes heterogeneous treatment effects by grouping observations
based on predicted treatment effects and estimating average effects within groups.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import warnings


class GATES:
    """
    Group Average Treatment Effects (GATES) for heterogeneity analysis.
    
    GATES groups observations based on predicted baseline treatment effects
    and estimates the average treatment effect within each group. This helps
    identify which subpopulations benefit most from treatment.
    
    References
    ----------
    Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2020).
    Generic machine learning inference on heterogeneous treatment effects 
    in randomized experiments.
    
    Parameters
    ----------
    n_groups : int
        Number of groups to create (default: 5 quintiles)
    ml_model : BaseEstimator, optional
        ML model for predicting baseline outcomes
    n_folds : int
        Number of folds for cross-fitting (default: 5)
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_groups: int = 5,
        ml_model: Optional[BaseEstimator] = None,
        n_folds: int = 5,
        random_state: Optional[int] = 42
    ):
        self.n_groups = n_groups
        self.ml_model = ml_model or RandomForestRegressor(
            n_estimators=100, random_state=random_state
        )
        self.n_folds = n_folds
        self.random_state = random_state
        self.gates_ = None
        self.gates_se_ = None
        self.group_labels_ = None
        self.fitted_ = False
        
    def fit(
        self, 
        X: pd.DataFrame, 
        T: pd.Series, 
        Y: pd.Series
    ) -> 'GATES':
        """
        Fit the GATES model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        T : pd.Series
            Treatment assignment
        Y : pd.Series
            Observed outcomes
            
        Returns
        -------
        self : GATES
            Fitted model
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        T_array = T.values if isinstance(T, pd.Series) else T
        Y_array = Y.values if isinstance(Y, pd.Series) else Y
        
        n_samples = len(X_array)
        
        # Step 1: Predict baseline outcomes using cross-fitting
        baseline_scores = np.zeros(n_samples)
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, test_idx in kf.split(X_array):
            # Use only control group for baseline prediction
            control_idx = train_idx[T_array[train_idx] == 0]
            
            if len(control_idx) > 0:
                X_control = X_array[control_idx]
                Y_control = Y_array[control_idx]
                
                model = clone(self.ml_model)
                model.fit(X_control, Y_control)
                baseline_scores[test_idx] = model.predict(X_array[test_idx])
            else:
                warnings.warn("No control observations in fold, using mean")
                baseline_scores[test_idx] = np.mean(Y_array[train_idx])
        
        # Step 2: Create groups based on predicted baseline scores
        # Higher baseline scores might indicate different treatment effects
        self.group_labels_ = pd.qcut(
            baseline_scores, 
            q=self.n_groups, 
            labels=[f'Group {i+1}' for i in range(self.n_groups)],
            duplicates='drop'
        )
        
        # Step 3: Estimate treatment effects within each group
        self.gates_ = {}
        self.gates_se_ = {}
        
        for group in self.group_labels_.unique():
            group_mask = (self.group_labels_ == group)
            
            # Treated and control in this group
            treated_mask = group_mask & (T_array == 1)
            control_mask = group_mask & (T_array == 0)
            
            if treated_mask.sum() > 0 and control_mask.sum() > 0:
                # Estimate GATE as difference in means
                y_treated = Y_array[treated_mask]
                y_control = Y_array[control_mask]
                
                gate = np.mean(y_treated) - np.mean(y_control)
                
                # Standard error using Neyman's formula
                var_treated = np.var(y_treated) / len(y_treated)
                var_control = np.var(y_control) / len(y_control)
                se = np.sqrt(var_treated + var_control)
                
                self.gates_[group] = gate
                self.gates_se_[group] = se
            else:
                self.gates_[group] = np.nan
                self.gates_se_[group] = np.nan
        
        self.fitted_ = True
        return self
    
    def get_gates(self) -> pd.DataFrame:
        """
        Get GATES estimates for all groups.
        
        Returns
        -------
        gates_df : pd.DataFrame
            DataFrame with GATES estimates, standard errors, and confidence intervals
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        results = []
        for group in sorted(self.gates_.keys()):
            gate = self.gates_[group]
            se = self.gates_se_[group]
            
            if not np.isnan(gate):
                ci_lower = gate - 1.96 * se
                ci_upper = gate + 1.96 * se
                p_value = 2 * (1 - self._normal_cdf(abs(gate / se)))
            else:
                ci_lower = ci_upper = p_value = np.nan
            
            results.append({
                'Group': group,
                'GATE': gate,
                'Std. Error': se,
                'CI Lower': ci_lower,
                'CI Upper': ci_upper,
                'P-value': p_value
            })
        
        return pd.DataFrame(results)
    
    def identify_high_benefit_groups(
        self, 
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Identify groups with treatment effects above a threshold.
        
        Parameters
        ----------
        threshold : float, optional
            Minimum treatment effect threshold. If None, uses median GATE.
            
        Returns
        -------
        high_benefit_groups : list
            List of group labels with high treatment effects
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        gates_df = self.get_gates()
        
        if threshold is None:
            threshold = gates_df['GATE'].median()
        
        high_benefit = gates_df[gates_df['GATE'] > threshold]['Group'].tolist()
        return high_benefit
    
    def get_group_membership(self) -> pd.Series:
        """
        Get group membership for all observations.
        
        Returns
        -------
        groups : pd.Series
            Group labels for each observation
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        return self.group_labels_
    
    @staticmethod
    def _normal_cdf(x):
        """Approximate standard normal CDF."""
        from scipy import stats
        return stats.norm.cdf(x)
    
    def summary(self) -> str:
        """
        Print a summary of GATES results.
        
        Returns
        -------
        summary : str
            Formatted summary string
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        gates_df = self.get_gates()
        high_benefit = self.identify_high_benefit_groups()
        
        summary = "=" * 60 + "\n"
        summary += "GATES: Group Average Treatment Effects\n"
        summary += "=" * 60 + "\n\n"
        summary += gates_df.to_string(index=False) + "\n\n"
        summary += f"High-benefit groups (above median): {', '.join(high_benefit)}\n"
        summary += "=" * 60 + "\n"
        
        return summary

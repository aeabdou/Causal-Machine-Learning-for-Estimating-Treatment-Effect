"""Double Machine Learning (DML) pipeline for causal inference."""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV
import warnings


class DoubleMLPipeline:
    """
    Double Machine Learning (DML) for robust treatment effect estimation.
    
    DML uses cross-fitting and orthogonalization to provide robust estimates
    of treatment effects while allowing flexible machine learning models for
    nuisance parameters.
    
    References
    ----------
    Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., 
    Newey, W., & Robins, J. (2018). Double/debiased machine learning for 
    treatment and structural parameters.
    
    Parameters
    ----------
    ml_model_outcome : BaseEstimator, optional
        ML model for outcome prediction (default: Random Forest)
    ml_model_treatment : BaseEstimator, optional
        ML model for treatment prediction (default: Random Forest)
    n_folds : int
        Number of folds for cross-fitting (default: 5)
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        ml_model_outcome: Optional[BaseEstimator] = None,
        ml_model_treatment: Optional[BaseEstimator] = None,
        n_folds: int = 5,
        random_state: Optional[int] = 42
    ):
        self.ml_model_outcome = ml_model_outcome or RandomForestRegressor(
            n_estimators=100, random_state=random_state
        )
        self.ml_model_treatment = ml_model_treatment or RandomForestRegressor(
            n_estimators=100, random_state=random_state
        )
        self.n_folds = n_folds
        self.random_state = random_state
        self.theta_ = None
        self.theta_se_ = None
        self.fitted_ = False
        
    def fit(
        self, 
        X: pd.DataFrame, 
        T: pd.Series, 
        Y: pd.Series
    ) -> 'DoubleMLPipeline':
        """
        Fit the Double ML model.
        
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
        self : DoubleMLPipeline
            Fitted model
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        T_array = T.values if isinstance(T, pd.Series) else T
        Y_array = Y.values if isinstance(Y, pd.Series) else Y
        
        n_samples = len(X_array)
        
        # Initialize arrays for residuals
        residual_Y = np.zeros(n_samples)
        residual_T = np.zeros(n_samples)
        
        # Cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, test_idx in kf.split(X_array):
            # Split data
            X_train, X_test = X_array[train_idx], X_array[test_idx]
            T_train, T_test = T_array[train_idx], T_array[test_idx]
            Y_train, Y_test = Y_array[train_idx], Y_array[test_idx]
            
            # Fit outcome model (Y ~ X)
            model_Y = clone(self.ml_model_outcome)
            model_Y.fit(X_train, Y_train)
            Y_pred = model_Y.predict(X_test)
            residual_Y[test_idx] = Y_test - Y_pred
            
            # Fit treatment model (T ~ X)
            model_T = clone(self.ml_model_treatment)
            model_T.fit(X_train, T_train)
            T_pred = model_T.predict(X_test)
            residual_T[test_idx] = T_test - T_pred
        
        # Estimate treatment effect using orthogonalized residuals
        # theta = E[residual_Y * residual_T] / E[residual_T^2]
        self.theta_ = np.sum(residual_Y * residual_T) / np.sum(residual_T ** 2)
        
        # Estimate standard error
        score = residual_Y - self.theta_ * residual_T
        var_theta = np.mean(score ** 2) / (np.mean(residual_T ** 2) ** 2)
        self.theta_se_ = np.sqrt(var_theta / n_samples)
        
        self.fitted_ = True
        return self
    
    def predict_cate(
        self, 
        X: pd.DataFrame, 
        T: pd.Series, 
        Y: pd.Series,
        return_std: bool = False
    ) -> np.ndarray:
        """
        Predict Conditional Average Treatment Effects (CATE).
        
        This uses a second-stage model to estimate heterogeneous effects.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        T : pd.Series
            Treatment assignment
        Y : pd.Series
            Observed outcomes
        return_std : bool
            Whether to return standard errors
            
        Returns
        -------
        cate : np.ndarray
            Conditional average treatment effects
        """
        # For simplicity, we return the average treatment effect
        # In a full implementation, this would fit a second-stage model
        cate = np.ones(len(X)) * self.theta_
        
        if return_std:
            std = np.ones(len(X)) * self.theta_se_
            return cate, std
        return cate
    
    def get_ate(self) -> Dict[str, float]:
        """
        Get Average Treatment Effect (ATE) estimate.
        
        Returns
        -------
        results : dict
            Dictionary with 'ate', 'se', 'ci_lower', 'ci_upper'
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        # 95% confidence interval
        ci_lower = self.theta_ - 1.96 * self.theta_se_
        ci_upper = self.theta_ + 1.96 * self.theta_se_
        
        return {
            'ate': self.theta_,
            'se': self.theta_se_,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': 2 * (1 - self._normal_cdf(abs(self.theta_ / self.theta_se_)))
        }
    
    @staticmethod
    def _normal_cdf(x):
        """Approximate standard normal CDF."""
        from scipy import stats
        return stats.norm.cdf(x)
    
    def summary(self) -> pd.DataFrame:
        """
        Get a summary table of results.
        
        Returns
        -------
        summary : pd.DataFrame
            Summary statistics
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        results = self.get_ate()
        
        summary_data = {
            'Estimate': [results['ate']],
            'Std. Error': [results['se']],
            'z-value': [results['ate'] / results['se']],
            'P>|z|': [results['p_value']],
            '[0.025': [results['ci_lower']],
            '0.975]': [results['ci_upper']]
        }
        
        return pd.DataFrame(summary_data, index=['ATE'])

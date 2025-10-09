"""
CLAN: Classification Analysis

CLAN uses machine learning to classify observations into groups with
different treatment effects, helping identify the most responsive subgroups.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV
import warnings


class CLAN:
    """
    Classification Analysis (CLAN) for identifying high-benefit subgroups.
    
    CLAN uses classification to identify observable characteristics that
    predict whether an observation will have above-median treatment effects.
    This helps policymakers target interventions to those who benefit most.
    
    References
    ----------
    Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2020).
    Generic machine learning inference on heterogeneous treatment effects 
    in randomized experiments.
    
    Parameters
    ----------
    ml_model_cate : BaseEstimator, optional
        ML model for predicting treatment effects
    ml_model_classifier : BaseEstimator, optional  
        ML model for classification into high/low benefit groups
    n_folds : int
        Number of folds for cross-fitting (default: 5)
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        ml_model_cate: Optional[BaseEstimator] = None,
        ml_model_classifier: Optional[BaseEstimator] = None,
        n_folds: int = 5,
        random_state: Optional[int] = 42
    ):
        self.ml_model_cate = ml_model_cate or RandomForestRegressor(
            n_estimators=100, random_state=random_state
        )
        self.ml_model_classifier = ml_model_classifier or RandomForestClassifier(
            n_estimators=100, random_state=random_state
        )
        self.n_folds = n_folds
        self.random_state = random_state
        self.classifier_ = None
        self.cate_estimates_ = None
        self.high_benefit_threshold_ = None
        self.high_benefit_labels_ = None
        self.fitted_ = False
        
    def fit(
        self, 
        X: pd.DataFrame, 
        T: pd.Series, 
        Y: pd.Series
    ) -> 'CLAN':
        """
        Fit the CLAN model.
        
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
        self : CLAN
            Fitted model
        """
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        T_array = T.values if isinstance(T, pd.Series) else T
        Y_array = Y.values if isinstance(Y, pd.Series) else Y
        
        n_samples = len(X_array)
        
        # Step 1: Estimate CATE using cross-fitting
        self.cate_estimates_ = self._estimate_cate(X_array, T_array, Y_array)
        
        # Step 2: Define high-benefit threshold (median)
        self.high_benefit_threshold_ = np.median(self.cate_estimates_)
        
        # Step 3: Create binary labels for high/low benefit
        self.high_benefit_labels_ = (
            self.cate_estimates_ > self.high_benefit_threshold_
        ).astype(int)
        
        # Step 4: Train classifier to predict high-benefit group
        self.classifier_ = clone(self.ml_model_classifier)
        self.classifier_.fit(X_array, self.high_benefit_labels_)
        
        self.fitted_ = True
        return self
    
    def _estimate_cate(
        self, 
        X: np.ndarray, 
        T: np.ndarray, 
        Y: np.ndarray
    ) -> np.ndarray:
        """
        Estimate CATE using S-learner approach with cross-fitting.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        T : np.ndarray
            Treatment assignment
        Y : np.ndarray
            Observed outcomes
            
        Returns
        -------
        cate : np.ndarray
            Estimated conditional average treatment effects
        """
        n_samples = len(X)
        cate = np.zeros(n_samples)
        
        # Create augmented features with treatment indicator
        X_with_T = np.column_stack([X, T])
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for train_idx, test_idx in kf.split(X):
            # Train model on training fold
            model = clone(self.ml_model_cate)
            model.fit(X_with_T[train_idx], Y[train_idx])
            
            # Predict with treatment on and off for test fold
            X_test = X[test_idx]
            X_test_treated = np.column_stack([X_test, np.ones(len(X_test))])
            X_test_control = np.column_stack([X_test, np.zeros(len(X_test))])
            
            Y_pred_treated = model.predict(X_test_treated)
            Y_pred_control = model.predict(X_test_control)
            
            # CATE is the difference
            cate[test_idx] = Y_pred_treated - Y_pred_control
        
        return cate
    
    def predict_high_benefit(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict whether observations are in the high-benefit group.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        predictions : np.ndarray
            Binary predictions (1 = high benefit, 0 = low benefit)
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.classifier_.predict(X_array)
    
    def predict_proba_high_benefit(
        self, 
        X: pd.DataFrame
    ) -> np.ndarray:
        """
        Predict probability of being in the high-benefit group.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        probabilities : np.ndarray
            Probability of high benefit
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        return self.classifier_.predict_proba(X_array)[:, 1]
    
    def get_targeting_strategy(
        self, 
        X: pd.DataFrame,
        budget: Optional[int] = None,
        percentile: Optional[float] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get targeting strategy for policy intervention.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix for population
        budget : int, optional
            Number of units that can be treated (e.g., 100,000 households)
        percentile : float, optional
            Top percentile to target (e.g., 0.2 for top 20%)
            
        Returns
        -------
        should_target : np.ndarray
            Binary array indicating which units to target
        stats : dict
            Statistics about the targeting strategy
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        # Get predicted probabilities
        proba = self.predict_proba_high_benefit(X)
        
        # Determine targeting
        if budget is not None:
            # Target top N by probability
            threshold_idx = max(0, len(proba) - budget)
            threshold = np.sort(proba)[threshold_idx]
            should_target = (proba >= threshold).astype(int)
            
            stats = {
                'budget': budget,
                'targeted': should_target.sum(),
                'threshold_probability': threshold,
                'avg_probability_targeted': proba[should_target == 1].mean(),
                'avg_probability_not_targeted': proba[should_target == 0].mean() if (should_target == 0).any() else 0
            }
        elif percentile is not None:
            # Target top percentile
            threshold = np.percentile(proba, (1 - percentile) * 100)
            should_target = (proba >= threshold).astype(int)
            
            stats = {
                'percentile': percentile,
                'targeted': should_target.sum(),
                'threshold_probability': threshold,
                'avg_probability_targeted': proba[should_target == 1].mean(),
                'avg_probability_not_targeted': proba[should_target == 0].mean() if (should_target == 0).any() else 0
            }
        else:
            # Use classifier's default threshold (0.5)
            should_target = self.predict_high_benefit(X)
            
            stats = {
                'targeted': should_target.sum(),
                'threshold_probability': 0.5,
                'avg_probability_targeted': proba[should_target == 1].mean(),
                'avg_probability_not_targeted': proba[should_target == 0].mean() if (should_target == 0).any() else 0
            }
        
        return should_target, stats
    
    def get_feature_importance(
        self, 
        feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get feature importance for the classification model.
        
        Parameters
        ----------
        feature_names : list, optional
            Names of features
            
        Returns
        -------
        importance_df : pd.DataFrame
            Feature importance rankings
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        if hasattr(self.classifier_, 'feature_importances_'):
            importances = self.classifier_.feature_importances_
        elif hasattr(self.classifier_, 'coef_'):
            importances = np.abs(self.classifier_.coef_[0])
        else:
            raise ValueError("Classifier does not have feature importance")
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    
    def summary(
        self, 
        X: Optional[pd.DataFrame] = None,
        budget: Optional[int] = None
    ) -> str:
        """
        Print a summary of CLAN results.
        
        Parameters
        ----------
        X : pd.DataFrame, optional
            Feature matrix for generating targeting strategy
        budget : int, optional
            Budget for targeting strategy
            
        Returns
        -------
        summary : str
            Formatted summary string
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        summary = "=" * 70 + "\n"
        summary += "CLAN: Classification Analysis for Targeting\n"
        summary += "=" * 70 + "\n\n"
        
        summary += f"High-benefit threshold (CATE): {self.high_benefit_threshold_:.4f}\n"
        summary += f"Proportion classified as high-benefit: {self.high_benefit_labels_.mean():.2%}\n\n"
        
        if X is not None and budget is not None:
            should_target, stats = self.get_targeting_strategy(X, budget=budget)
            summary += f"\nTargeting Strategy (Budget: {budget:,}):\n"
            summary += f"  - Units to target: {stats['targeted']:,}\n"
            summary += f"  - Threshold probability: {stats['threshold_probability']:.4f}\n"
            summary += f"  - Avg prob (targeted): {stats['avg_probability_targeted']:.4f}\n"
            summary += f"  - Avg prob (not targeted): {stats['avg_probability_not_targeted']:.4f}\n"
        
        summary += "\n" + "=" * 70 + "\n"
        
        return summary

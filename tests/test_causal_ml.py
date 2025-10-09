"""Tests for the causal ML pipelines."""

import numpy as np
import pandas as pd
import pytest
from causal_ml import DoubleMLPipeline, GATES, CLAN
from causal_ml.utils import generate_synthetic_data, create_policy_relevant_data


class TestDoubleML:
    """Tests for Double ML pipeline."""
    
    def test_basic_fit(self):
        """Test basic fitting functionality."""
        X, T, Y, tau = generate_synthetic_data(n_samples=1000, random_state=42)
        
        dml = DoubleMLPipeline(n_folds=3, random_state=42)
        dml.fit(X, T, Y)
        
        assert dml.fitted_ is True
        assert dml.theta_ is not None
        assert dml.theta_se_ is not None
    
    def test_ate_estimation(self):
        """Test ATE estimation is reasonable."""
        X, T, Y, tau = generate_synthetic_data(
            n_samples=5000, 
            treatment_effect_heterogeneity=False,
            random_state=42
        )
        
        dml = DoubleMLPipeline(n_folds=5, random_state=42)
        dml.fit(X, T, Y)
        
        ate_results = dml.get_ate()
        
        # With constant effect of 5, should be close
        assert 3 < ate_results['ate'] < 7
        assert ate_results['se'] > 0
    
    def test_summary(self):
        """Test summary generation."""
        X, T, Y, tau = generate_synthetic_data(n_samples=1000, random_state=42)
        
        dml = DoubleMLPipeline(n_folds=3, random_state=42)
        dml.fit(X, T, Y)
        
        summary = dml.summary()
        assert isinstance(summary, pd.DataFrame)
        assert 'Estimate' in summary.columns


class TestGATES:
    """Tests for GATES."""
    
    def test_basic_fit(self):
        """Test basic fitting functionality."""
        X, T, Y, tau = generate_synthetic_data(n_samples=1000, random_state=42)
        
        gates = GATES(n_groups=5, n_folds=3, random_state=42)
        gates.fit(X, T, Y)
        
        assert gates.fitted_ is True
        assert gates.gates_ is not None
    
    def test_gates_estimation(self):
        """Test GATES produces valid results."""
        X, T, Y, tau = generate_synthetic_data(n_samples=2000, random_state=42)
        
        gates = GATES(n_groups=3, n_folds=3, random_state=42)
        gates.fit(X, T, Y)
        
        gates_df = gates.get_gates()
        
        assert len(gates_df) <= 3  # Should have up to 3 groups
        assert 'GATE' in gates_df.columns
        assert 'Std. Error' in gates_df.columns
    
    def test_high_benefit_identification(self):
        """Test identification of high-benefit groups."""
        X, T, Y, tau = generate_synthetic_data(n_samples=2000, random_state=42)
        
        gates = GATES(n_groups=5, n_folds=3, random_state=42)
        gates.fit(X, T, Y)
        
        high_benefit = gates.identify_high_benefit_groups()
        assert isinstance(high_benefit, list)
        assert len(high_benefit) > 0


class TestCLAN:
    """Tests for CLAN."""
    
    def test_basic_fit(self):
        """Test basic fitting functionality."""
        X, T, Y, tau = generate_synthetic_data(n_samples=1000, random_state=42)
        
        clan = CLAN(n_folds=3, random_state=42)
        clan.fit(X, T, Y)
        
        assert clan.fitted_ is True
        assert clan.classifier_ is not None
    
    def test_prediction(self):
        """Test prediction functionality."""
        X, T, Y, tau = generate_synthetic_data(n_samples=1000, random_state=42)
        
        clan = CLAN(n_folds=3, random_state=42)
        clan.fit(X, T, Y)
        
        predictions = clan.predict_high_benefit(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_targeting_strategy(self):
        """Test targeting strategy generation."""
        X, T, Y, tau = generate_synthetic_data(n_samples=1000, random_state=42)
        
        clan = CLAN(n_folds=3, random_state=42)
        clan.fit(X, T, Y)
        
        should_target, stats = clan.get_targeting_strategy(X, budget=500)
        
        assert len(should_target) == len(X)
        assert stats['targeted'] <= 500
        assert 'threshold_probability' in stats
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        X, T, Y, tau = generate_synthetic_data(n_samples=1000, random_state=42)
        
        clan = CLAN(n_folds=3, random_state=42)
        clan.fit(X, T, Y)
        
        importance = clan.get_feature_importance(X.columns.tolist())
        assert isinstance(importance, pd.DataFrame)
        assert 'Feature' in importance.columns
        assert 'Importance' in importance.columns


class TestDataGeneration:
    """Tests for data generation utilities."""
    
    def test_synthetic_data(self):
        """Test synthetic data generation."""
        X, T, Y, tau = generate_synthetic_data(n_samples=1000, random_state=42)
        
        assert len(X) == 1000
        assert len(T) == 1000
        assert len(Y) == 1000
        assert len(tau) == 1000
        assert isinstance(X, pd.DataFrame)
    
    def test_policy_data(self):
        """Test policy-relevant data generation."""
        X, T, Y, tau = create_policy_relevant_data(n_samples=10000, random_state=42)
        
        assert len(X) == 10000
        assert 'income' in X.columns
        assert 'education' in X.columns
        assert 'household_size' in X.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

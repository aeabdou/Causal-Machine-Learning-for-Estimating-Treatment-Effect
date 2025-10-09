"""
Validation Script: Verify causal ML pipeline implementation

This script runs comprehensive checks to ensure all components
are working correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from causal_ml import DoubleMLPipeline, GATES, CLAN
from causal_ml.utils import generate_synthetic_data, create_policy_relevant_data


def test_imports():
    """Test that all modules import correctly."""
    print("✓ All imports successful")
    return True


def test_data_generation():
    """Test data generation utilities."""
    try:
        X, T, Y, tau = generate_synthetic_data(n_samples=100)
        assert len(X) == 100
        assert len(T) == 100
        assert len(Y) == 100
        assert len(tau) == 100
        
        X, T, Y, tau = create_policy_relevant_data(n_samples=100)
        assert len(X) == 100
        
        print("✓ Data generation working correctly")
        return True
    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        return False


def test_double_ml():
    """Test Double ML pipeline."""
    try:
        X, T, Y, tau = generate_synthetic_data(n_samples=500)
        
        dml = DoubleMLPipeline(n_folds=3, random_state=42)
        dml.fit(X, T, Y)
        
        ate = dml.get_ate()
        assert 'ate' in ate
        assert 'se' in ate
        assert 'ci_lower' in ate
        assert 'ci_upper' in ate
        
        summary = dml.summary()
        assert summary is not None
        
        print("✓ Double ML pipeline working correctly")
        return True
    except Exception as e:
        print(f"✗ Double ML failed: {e}")
        return False


def test_gates():
    """Test GATES implementation."""
    try:
        X, T, Y, tau = generate_synthetic_data(n_samples=500)
        
        gates = GATES(n_groups=3, n_folds=3, random_state=42)
        gates.fit(X, T, Y)
        
        gates_df = gates.get_gates()
        assert len(gates_df) > 0
        
        high_benefit = gates.identify_high_benefit_groups()
        assert isinstance(high_benefit, list)
        
        groups = gates.get_group_membership()
        assert len(groups) == len(X)
        
        print("✓ GATES working correctly")
        return True
    except Exception as e:
        print(f"✗ GATES failed: {e}")
        return False


def test_clan():
    """Test CLAN implementation."""
    try:
        X, T, Y, tau = generate_synthetic_data(n_samples=500)
        
        clan = CLAN(n_folds=3, random_state=42)
        clan.fit(X, T, Y)
        
        predictions = clan.predict_high_benefit(X)
        assert len(predictions) == len(X)
        
        proba = clan.predict_proba_high_benefit(X)
        assert len(proba) == len(X)
        
        should_target, stats = clan.get_targeting_strategy(X, budget=250)
        assert len(should_target) == len(X)
        assert stats['targeted'] <= 250
        
        importance = clan.get_feature_importance(X.columns.tolist())
        assert len(importance) > 0
        
        print("✓ CLAN working correctly")
        return True
    except Exception as e:
        print(f"✗ CLAN failed: {e}")
        return False


def test_integration():
    """Test full pipeline integration."""
    try:
        X, T, Y, tau = create_policy_relevant_data(n_samples=1000)
        
        # Full pipeline
        dml = DoubleMLPipeline(n_folds=3, random_state=42)
        dml.fit(X, T, Y)
        ate = dml.get_ate()
        
        gates = GATES(n_groups=3, n_folds=3, random_state=42)
        gates.fit(X, T, Y)
        gates_results = gates.get_gates()
        
        clan = CLAN(n_folds=3, random_state=42)
        clan.fit(X, T, Y)
        should_target, stats = clan.get_targeting_strategy(X, budget=500)
        
        # Validate results
        targeted_effect = tau[should_target == 1].mean()
        not_targeted_effect = tau[should_target == 0].mean()
        
        assert targeted_effect > not_targeted_effect, "CLAN should target higher effect individuals"
        
        print("✓ Full pipeline integration successful")
        print(f"  - ATE: {ate['ate']:.2f}")
        print(f"  - GATES groups: {len(gates_results)}")
        print(f"  - Targeted effect: {targeted_effect:.2f}")
        print(f"  - Non-targeted effect: {not_targeted_effect:.2f}")
        print(f"  - Gain from targeting: {targeted_effect - not_targeted_effect:.2f}")
        return True
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("CAUSAL ML PIPELINE VALIDATION")
    print("="*60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Data Generation", test_data_generation),
        ("Double ML", test_double_ml),
        ("GATES", test_gates),
        ("CLAN", test_clan),
        ("Integration", test_integration),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        print("-" * 60)
        success = test_func()
        results.append((name, success))
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60 + "\n")
    
    all_passed = all(success for _, success in results)
    
    for name, success in results:
        status = "PASS" if success else "FAIL"
        symbol = "✓" if success else "✗"
        print(f"{symbol} {name}: {status}")
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED - Pipeline is ready for use!")
    else:
        print("SOME TESTS FAILED - Please review errors above")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())

"""
Quick Demo: Causal ML Pipeline

A faster version of the policy targeting example with a smaller dataset
for quick demonstration and testing.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from causal_ml import DoubleMLPipeline, GATES, CLAN
from causal_ml.utils import generate_synthetic_data

np.random.seed(42)


def main():
    """Quick demonstration of causal ML pipelines."""
    
    print("\n" + "="*70)
    print("QUICK DEMO: Causal ML Pipeline")
    print("="*70 + "\n")
    
    # Generate synthetic data
    print("Generating synthetic data (10,000 samples)...")
    X, T, Y, true_tau = generate_synthetic_data(
        n_samples=10000, 
        treatment_effect_heterogeneity=True,
        random_state=42
    )
    print(f"✓ Generated {len(X):,} observations\n")
    
    # Double ML
    print("1. Double ML - Average Treatment Effect")
    print("-" * 70)
    dml = DoubleMLPipeline(n_folds=5, random_state=42)
    dml.fit(X, T, Y)
    
    ate = dml.get_ate()
    print(f"   ATE: {ate['ate']:.3f} ± {ate['se']:.3f}")
    print(f"   95% CI: [{ate['ci_lower']:.3f}, {ate['ci_upper']:.3f}]")
    print(f"   True ATE: {true_tau.mean():.3f}\n")
    
    # GATES
    print("2. GATES - Heterogeneous Treatment Effects")
    print("-" * 70)
    gates = GATES(n_groups=5, n_folds=3, random_state=42)
    gates.fit(X, T, Y)
    
    gates_df = gates.get_gates()
    print(gates_df.to_string(index=False))
    
    high_benefit = gates.identify_high_benefit_groups()
    print(f"\n   High-benefit groups: {', '.join(high_benefit)}\n")
    
    # CLAN
    print("3. CLAN - Targeting Strategy")
    print("-" * 70)
    clan = CLAN(n_folds=3, random_state=42)
    clan.fit(X, T, Y)
    
    # Target top 5000
    should_target, stats = clan.get_targeting_strategy(X, budget=5000)
    print(f"   Budget: {stats['targeted']:,} households")
    print(f"   Targeting threshold (probability): {stats['threshold_probability']:.4f}")
    
    # Evaluate
    targeted_effect = true_tau[should_target == 1].mean()
    not_targeted_effect = true_tau[should_target == 0].mean()
    print(f"\n   True effect (targeted): {targeted_effect:.3f}")
    print(f"   True effect (not targeted): {not_targeted_effect:.3f}")
    print(f"   Gain from targeting: {targeted_effect - not_targeted_effect:.3f}\n")
    
    # Feature importance
    print("4. Feature Importance")
    print("-" * 70)
    importance = clan.get_feature_importance(X.columns.tolist())
    print(importance.head().to_string(index=False))
    
    print("\n" + "="*70)
    print("Demo complete! See examples/policy_targeting_example.py for full version.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

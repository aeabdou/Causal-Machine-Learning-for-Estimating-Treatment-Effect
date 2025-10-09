"""
Example: Policy Targeting for Household Intervention Program

This example demonstrates how to use Double ML, GATES, and CLAN to:
1. Estimate average treatment effects robustly
2. Identify heterogeneous treatment effects across groups
3. Target the highest-benefit households for a policy intervention

Scenario: A government wants to implement a household assistance program
but has budget to serve only 100,000 households. We use causal ML to
identify which households would benefit most.
"""

import sys
from pathlib import Path

# Add parent directory to path to import causal_ml
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our causal ML modules
from causal_ml import DoubleMLPipeline, GATES, CLAN
from causal_ml.utils import create_policy_relevant_data

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("CAUSAL ML PIPELINE FOR POLICY TARGETING")
    print("=" * 80)
    print()
    
    # =========================================================================
    # Step 1: Generate synthetic policy-relevant data
    # =========================================================================
    print("Step 1: Generating synthetic household data (n=100,000)...")
    print("-" * 80)
    
    X, T, Y, true_tau = create_policy_relevant_data(
        n_samples=100000,
        random_state=42
    )
    
    print(f"Generated {len(X):,} households with {X.shape[1]} characteristics")
    print(f"Treatment rate: {T.mean():.2%}")
    print(f"Average outcome: ${Y.mean():,.2f}")
    print()
    
    # Display sample data
    print("Sample of household characteristics:")
    print(X.head())
    print()
    
    # =========================================================================
    # Step 2: Double ML for Average Treatment Effect
    # =========================================================================
    print("\nStep 2: Estimating Average Treatment Effect using Double ML...")
    print("-" * 80)
    
    dml = DoubleMLPipeline(n_folds=5, random_state=42)
    dml.fit(X, T, Y)
    
    ate_results = dml.get_ate()
    print(f"\nAverage Treatment Effect (ATE): ${ate_results['ate']:,.2f}")
    print(f"Standard Error: ${ate_results['se']:,.2f}")
    print(f"95% CI: [${ate_results['ci_lower']:,.2f}, ${ate_results['ci_upper']:,.2f}]")
    print(f"P-value: {ate_results['p_value']:.4f}")
    print()
    
    print("Detailed Summary:")
    print(dml.summary())
    print()
    
    # =========================================================================
    # Step 3: GATES for Group Heterogeneity Analysis
    # =========================================================================
    print("\nStep 3: Analyzing heterogeneous effects using GATES...")
    print("-" * 80)
    
    gates = GATES(n_groups=5, n_folds=5, random_state=42)
    gates.fit(X, T, Y)
    
    print(gates.summary())
    
    gates_results = gates.get_gates()
    print("\nHigh-benefit groups identified:")
    high_benefit_groups = gates.identify_high_benefit_groups()
    print(f"Groups with above-median treatment effects: {', '.join(high_benefit_groups)}")
    print()
    
    # =========================================================================
    # Step 4: CLAN for Targeted Policy Recommendations
    # =========================================================================
    print("\nStep 4: Developing targeting strategy using CLAN...")
    print("-" * 80)
    
    clan = CLAN(n_folds=5, random_state=42)
    clan.fit(X, T, Y)
    
    print(clan.summary(X, budget=100000))
    
    # Get targeting strategy
    should_target, targeting_stats = clan.get_targeting_strategy(X, budget=100000)
    
    print(f"\nTargeting Recommendation:")
    print(f"  - Total households: {len(X):,}")
    print(f"  - Households to target: {targeting_stats['targeted']:,}")
    print(f"  - Targeting rate: {targeting_stats['targeted']/len(X):.2%}")
    print()
    
    # =========================================================================
    # Step 5: Evaluate targeting strategy against true effects
    # =========================================================================
    print("\nStep 5: Evaluating targeting strategy performance...")
    print("-" * 80)
    
    # Compare targeted vs non-targeted groups
    targeted_mask = should_target == 1
    
    true_effect_targeted = true_tau[targeted_mask].mean()
    true_effect_not_targeted = true_tau[~targeted_mask].mean()
    
    print(f"Average true treatment effect (targeted group): ${true_effect_targeted:,.2f}")
    print(f"Average true treatment effect (non-targeted group): ${true_effect_not_targeted:,.2f}")
    print(f"Gain from targeting: ${true_effect_targeted - true_effect_not_targeted:,.2f}")
    print()
    
    # Calculate total benefit
    if T.sum() > 0:
        # If we had targeted the top 100k
        top_100k_idx = np.argsort(true_tau)[-100000:]
        optimal_benefit = true_tau.iloc[top_100k_idx].sum()
        
        # Our targeting benefit (simulated)
        our_targeting_benefit = true_tau[targeted_mask].sum() if targeted_mask.sum() > 0 else 0
        
        # Random targeting benefit
        random_benefit = true_tau.sample(min(100000, len(true_tau)), random_state=42).sum()
        
        print(f"Total benefit comparison:")
        print(f"  - Optimal targeting (oracle): ${optimal_benefit:,.2f}")
        print(f"  - Our targeting (CLAN): ${our_targeting_benefit:,.2f}")
        print(f"  - Random targeting: ${random_benefit:,.2f}")
        print(f"  - Efficiency vs optimal: {our_targeting_benefit/optimal_benefit:.2%}")
        print(f"  - Improvement vs random: {(our_targeting_benefit - random_benefit)/random_benefit:.2%}")
    print()
    
    # =========================================================================
    # Step 6: Feature importance for targeting
    # =========================================================================
    print("\nStep 6: Key characteristics for identifying high-benefit households...")
    print("-" * 80)
    
    feature_importance = clan.get_feature_importance(X.columns.tolist())
    print("\nTop 5 most important features:")
    print(feature_importance.head())
    print()
    
    # =========================================================================
    # Step 7: Visualizations
    # =========================================================================
    print("\nStep 7: Creating visualizations...")
    print("-" * 80)
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: GATES results
    fig, ax = plt.subplots(figsize=(10, 6))
    gates_plot = gates_results.copy()
    gates_plot = gates_plot.sort_values('GATE', ascending=False)
    
    ax.barh(range(len(gates_plot)), gates_plot['GATE'], 
            xerr=1.96*gates_plot['Std. Error'], capsize=5)
    ax.set_yticks(range(len(gates_plot)))
    ax.set_yticklabels(gates_plot['Group'])
    ax.set_xlabel('Group Average Treatment Effect ($)')
    ax.set_title('GATES: Treatment Effects by Group')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'gates_results.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gates_results.png'}")
    
    # Plot 2: Feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features = feature_importance.head(10)
    ax.barh(range(len(top_features)), top_features['Importance'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance for Targeting High-Benefit Households')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'feature_importance.png'}")
    
    # Plot 3: Distribution of treatment effects
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(true_tau[targeted_mask], bins=50, alpha=0.7, label='Targeted', color='green')
    axes[0].hist(true_tau[~targeted_mask], bins=50, alpha=0.7, label='Not Targeted', color='red')
    axes[0].set_xlabel('True Treatment Effect ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Treatment Effects')
    axes[0].legend()
    axes[0].axvline(x=true_effect_targeted, color='green', linestyle='--', label='Mean (Targeted)')
    axes[0].axvline(x=true_effect_not_targeted, color='red', linestyle='--', label='Mean (Not Targeted)')
    
    # Income distribution by targeting
    axes[1].hist(X.loc[targeted_mask, 'income'], bins=50, alpha=0.7, 
                 label='Targeted', color='green', density=True)
    axes[1].hist(X.loc[~targeted_mask, 'income'], bins=50, alpha=0.7, 
                 label='Not Targeted', color='red', density=True)
    axes[1].set_xlabel('Income ($)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Income Distribution by Targeting Decision')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'targeting_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'targeting_analysis.png'}")
    
    print()
    
    # =========================================================================
    # Step 8: Save results
    # =========================================================================
    print("\nStep 8: Saving results...")
    print("-" * 80)
    
    # Save targeting recommendations
    targeting_df = X.copy()
    targeting_df['should_target'] = should_target
    targeting_df['predicted_benefit_prob'] = clan.predict_proba_high_benefit(X)
    targeting_df['true_treatment_effect'] = true_tau
    
    targeting_df.to_csv(output_dir / 'targeting_recommendations.csv', index=False)
    print(f"Saved: {output_dir / 'targeting_recommendations.csv'}")
    
    # Save summary statistics
    summary_stats = {
        'ate': ate_results['ate'],
        'ate_se': ate_results['se'],
        'ate_ci_lower': ate_results['ci_lower'],
        'ate_ci_upper': ate_results['ci_upper'],
        'n_households': len(X),
        'treatment_rate': T.mean(),
        'targeted_households': targeting_stats['targeted'],
        'targeting_threshold': targeting_stats['threshold_probability'],
        'true_effect_targeted': true_effect_targeted,
        'true_effect_not_targeted': true_effect_not_targeted,
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
    print(f"Saved: {output_dir / 'summary_statistics.csv'}")
    
    # Save GATES results
    gates_results.to_csv(output_dir / 'gates_results.csv', index=False)
    print(f"Saved: {output_dir / 'gates_results.csv'}")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Successfully identified {targeting_stats['targeted']:,} high-benefit households")
    print(f"  - Expected gain from targeting: ${true_effect_targeted - true_effect_not_targeted:,.2f} per household")
    print(f"  - All results saved to '{output_dir}' directory")
    print()


if __name__ == "__main__":
    main()

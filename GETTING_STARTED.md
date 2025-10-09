# Getting Started with Causal ML

This guide will help you get up and running with the causal ML pipelines in 5 minutes.

## Installation (1 minute)

```bash
# Clone the repository (if not already done)
git clone https://github.com/aeabdou/Causal-Machine-Learning-for-Estimating-Treatment-Effect.git
cd Causal-Machine-Learning-for-Estimating-Treatment-Effect

# Install dependencies
pip install -r requirements.txt

# Verify installation
python validate.py
```

Expected output: "ALL TESTS PASSED - Pipeline is ready for use!"

## Quick Start (2 minutes)

### Run the Quick Demo

```bash
python examples/quick_demo.py
```

This will:
- Generate 10,000 synthetic observations
- Estimate treatment effects with Double ML
- Identify heterogeneous effects with GATES
- Create targeting strategy with CLAN
- Show feature importance

### Expected Output

```
QUICK DEMO: Causal ML Pipeline

1. Double ML - Average Treatment Effect
   ATE: 4.937 ± 0.090
   95% CI: [4.760, 5.114]

2. GATES - Heterogeneous Treatment Effects
   Group 1 effect: 1.57
   Group 5 effect: 9.10
   High-benefit groups: 4 & 5

3. CLAN - Targeting Strategy
   Gain from targeting: 5.78 (58% improvement!)

4. Feature Importance
   X0: 54.3% (most important)
```

## Use with Your Own Data (2 minutes)

### Prepare Your Data

Your data should have three components:

```python
import pandas as pd

# 1. Features (covariates)
X = pd.DataFrame({
    'age': [25, 30, 35, ...],
    'income': [50000, 60000, 45000, ...],
    'education': [12, 16, 14, ...]
})

# 2. Treatment (0 = control, 1 = treated)
T = pd.Series([0, 1, 0, 1, ...])

# 3. Outcome (the variable you want to improve)
Y = pd.Series([1000, 1500, 900, ...])
```

### Analyze Your Data

```python
from causal_ml import DoubleMLPipeline, GATES, CLAN

# Step 1: Estimate average effect
dml = DoubleMLPipeline(n_folds=5, random_state=42)
dml.fit(X, T, Y)
print(dml.summary())

# Step 2: Find which groups benefit most
gates = GATES(n_groups=5, n_folds=5, random_state=42)
gates.fit(X, T, Y)
print(gates.summary())

# Step 3: Identify individuals to target
clan = CLAN(n_folds=5, random_state=42)
clan.fit(X, T, Y)

# Target top 10,000 (or whatever your budget allows)
should_target, stats = clan.get_targeting_strategy(X, budget=10000)

# Save results
targeting_df = X.copy()
targeting_df['should_target'] = should_target
targeting_df.to_csv('targeting_recommendations.csv')
```

## Common Use Cases

### 1. Should we run this program?

**Use Double ML** to get the Average Treatment Effect (ATE):

```python
dml = DoubleMLPipeline()
dml.fit(X, T, Y)
ate = dml.get_ate()

if ate['ci_lower'] > 0:
    print(f"✓ Program is effective! Effect: {ate['ate']:.2f}")
else:
    print("⚠ No significant effect detected")
```

### 2. Who benefits most from the program?

**Use GATES** to find high-benefit groups:

```python
gates = GATES(n_groups=5)
gates.fit(X, T, Y)
gates_df = gates.get_gates()

# Groups are sorted by baseline characteristics
# Look for groups with highest GATE values
high_benefit = gates.identify_high_benefit_groups()
print(f"Focus on: {', '.join(high_benefit)}")
```

### 3. We can only serve 100,000 people - who should get it?

**Use CLAN** for targeted selection:

```python
clan = CLAN()
clan.fit(X, T, Y)

# Get targeting recommendations
should_target, stats = clan.get_targeting_strategy(X, budget=100000)

# See what predicts high benefit
importance = clan.get_feature_importance(X.columns.tolist())
print("\nMost important characteristics:")
print(importance.head())

# Save targeting list
X['priority_score'] = clan.predict_proba_high_benefit(X)
X['should_enroll'] = should_target
X.sort_values('priority_score', ascending=False).to_csv('enrollment_priority.csv')
```

## Next Steps

### Learn More
- Read [TUTORIAL.md](TUTORIAL.md) for detailed explanations
- See [README.md](README.md) for API reference
- Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for technical details

### Run Full Example
```bash
python examples/policy_targeting_example.py
```

This creates:
- Visualizations in `outputs/` directory
- CSV files with results
- Comprehensive analysis of 100,000 households

### Run Tests
```bash
# Unit tests
python -m pytest tests/ -v

# Validation checks
python validate.py
```

## Troubleshooting

### "No module named 'causal_ml'"

Add the directory to Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

Or install the package:
```bash
pip install -e .
```

### "Standard errors very large"

- Increase sample size (need more data)
- Reduce number of features (try feature selection)
- Check for strong confounding (may need better data)

### "All groups have similar effects"

- Effects may truly be homogeneous (constant effect)
- Try different number of groups
- Check if you have sufficient treatment effect variation

### "CLAN targets everyone / no one"

- May need to adjust threshold
- Check if there's real heterogeneity in your data
- Try: `clan.get_targeting_strategy(X, percentile=0.2)` for top 20%

## Getting Help

1. Check the [TUTORIAL.md](TUTORIAL.md) for detailed guidance
2. Review example output in [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
3. Open an issue on GitHub with:
   - Your code
   - Error message
   - Sample data (if possible)

## Quick Reference

### Key Parameters

**n_folds**: Number of cross-validation folds
- Small data (< 5K): use 3
- Medium data (5K-50K): use 5
- Large data (> 50K): use 10

**random_state**: Set to any integer (e.g., 42) for reproducibility

**n_groups** (GATES): Number of groups to create
- Common: 3, 5, or 10
- More groups = more granular but less stable

**budget** (CLAN): Number of units to target
- Must be ≤ sample size
- Can also use `percentile` instead

### Quick Commands

```bash
# Validate installation
python validate.py

# Quick demo (fast)
python examples/quick_demo.py

# Full example (slower, more comprehensive)
python examples/policy_targeting_example.py

# Run tests
python -m pytest tests/ -v
```

## That's It!

You're now ready to use causal ML for your policy targeting and treatment effect estimation needs. Happy analyzing! 🎉

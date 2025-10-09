# Tutorial: Causal ML for Policy Targeting

This tutorial demonstrates how to use the causal ML pipelines to identify and target the highest-benefit households for policy interventions.

## Table of Contents

1. [Installation](#installation)
2. [Basic Concepts](#basic-concepts)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Real-World Example](#real-world-example)
5. [Interpreting Results](#interpreting-results)

## Installation

```bash
# Install the package
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Basic Concepts

### Average Treatment Effect (ATE)

The ATE is the average causal effect of treatment across the entire population:
- **Double ML** provides robust ATE estimates using machine learning
- Handles confounding by using cross-fitting and orthogonalization

### Heterogeneous Treatment Effects

Different individuals may experience different treatment effects:
- **GATES** reveals how effects vary across groups
- **CLAN** identifies characteristics that predict high benefits

### Policy Targeting

When resources are limited, we want to target those who benefit most:
- CLAN provides actionable targeting recommendations
- Handles budget constraints (e.g., 100,000 households)

## Step-by-Step Guide

### Step 1: Load Your Data

```python
import pandas as pd
from causal_ml import DoubleMLPipeline, GATES, CLAN

# Your data should have:
# - X: Features/covariates (DataFrame)
# - T: Treatment assignment (0/1, Series)
# - Y: Outcome of interest (Series)

X = pd.read_csv('features.csv')
T = pd.read_csv('treatment.csv')['treatment']
Y = pd.read_csv('outcomes.csv')['outcome']
```

### Step 2: Estimate Average Treatment Effect

```python
# Initialize and fit Double ML
dml = DoubleMLPipeline(n_folds=5, random_state=42)
dml.fit(X, T, Y)

# Get ATE estimate
ate_results = dml.get_ate()
print(f"ATE: {ate_results['ate']:.2f}")
print(f"95% CI: [{ate_results['ci_lower']:.2f}, {ate_results['ci_upper']:.2f}]")
```

**Interpretation:**
- If ATE = 5000 with CI [4500, 5500], the treatment increases outcomes by $5000 on average
- The confidence interval tells us we're 95% confident the true effect is between $4500-$5500

### Step 3: Identify Heterogeneous Effects with GATES

```python
# Initialize and fit GATES
gates = GATES(n_groups=5, n_folds=5, random_state=42)
gates.fit(X, T, Y)

# Get group-level effects
gates_df = gates.get_gates()
print(gates_df)

# Identify high-benefit groups
high_benefit = gates.identify_high_benefit_groups()
print(f"High-benefit groups: {high_benefit}")
```

**Interpretation:**
- GATES divides the population into groups (e.g., quintiles)
- Each group has its own treatment effect estimate
- High-benefit groups are those with above-median effects

### Step 4: Develop Targeting Strategy with CLAN

```python
# Initialize and fit CLAN
clan = CLAN(n_folds=5, random_state=42)
clan.fit(X, T, Y)

# Get targeting recommendations for budget
should_target, stats = clan.get_targeting_strategy(X, budget=100000)

print(f"Targeting {stats['targeted']:,} households")
print(f"Threshold probability: {stats['threshold_probability']:.4f}")

# Get feature importance
importance = clan.get_feature_importance(X.columns.tolist())
print(importance.head())
```

**Interpretation:**
- CLAN identifies which households to target given budget constraints
- Feature importance shows which characteristics predict high benefits
- Use this to design eligibility criteria for programs

## Real-World Example

### Scenario: Cash Transfer Program

A government wants to implement a cash transfer program but can only serve 100,000 out of 500,000 eligible households.

```python
from causal_ml.utils import create_policy_relevant_data

# Generate synthetic data (or use real data)
X, T, Y, true_tau = create_policy_relevant_data(n_samples=500000)

# Step 1: Verify program effectiveness
dml = DoubleMLPipeline()
dml.fit(X, T, Y)
ate = dml.get_ate()

if ate['ci_lower'] > 0:
    print("✓ Program is effective!")
else:
    print("⚠ Program may not be effective")

# Step 2: Find which groups benefit most
gates = GATES(n_groups=5)
gates.fit(X, T, Y)
print(gates.summary())

# Step 3: Target the 100,000 highest-benefit households
clan = CLAN()
clan.fit(X, T, Y)
should_target, stats = clan.get_targeting_strategy(X, budget=100000)

# Create targeting dataset
targeting_df = X.copy()
targeting_df['should_target'] = should_target
targeting_df['benefit_score'] = clan.predict_proba_high_benefit(X)

# Save for implementation
targeting_df.to_csv('targeting_list.csv')
```

### Implementation Steps

1. **Program Design**: Use ATE to justify program funding
2. **Eligibility Criteria**: Use GATES to understand which groups benefit
3. **Selection Process**: Use CLAN predictions to rank applicants
4. **Evaluation**: Compare actual outcomes to predictions

## Interpreting Results

### Double ML Output

```
ATE: 6545.45
Std. Error: 34.54
95% CI: [6477.76, 6613.14]
P-value: 0.0000
```

- **ATE**: Average effect ($6545 benefit per household)
- **Std. Error**: Uncertainty in the estimate
- **95% CI**: Range where true effect likely lies
- **P-value**: Statistical significance (< 0.05 is significant)

### GATES Output

```
Group       GATE    Std. Error  CI Lower    CI Upper
Group 1     7787.67    45.16    7699.16    7876.19
Group 2     8582.94    40.34    8503.87    8662.00
Group 3     8220.11    41.50    8138.78    8301.44
Group 4     6563.24    46.46    6472.18    6654.30
Group 5    -3784.33   216.70   -4209.06   -3359.59
```

- **Group 5 has negative effect**: Program actually harms this group!
- **Groups 1-4 benefit**: But Groups 2 & 3 benefit most
- **Policy implication**: Exclude Group 5, prioritize Groups 2 & 3

### CLAN Output

```
Targeting 100,000 households
Threshold probability: 0.5900
Avg prob (targeted): 0.75
Avg prob (not targeted): 0.25
```

- **Threshold**: Only target households with >59% probability of high benefit
- **Model confidence**: Targeted group has 75% avg probability vs 25% for others
- **Expected impact**: Targeted group should have much higher effects

### Feature Importance

```
Feature          Importance
income           0.51
education        0.21
household_size   0.10
health_index     0.09
age             0.07
```

- **Income** is most important predictor (51% of importance)
- **Education** second most important (21%)
- **Policy design**: Focus outreach on low-income, educated households

## Best Practices

### 1. Always Check Overlap

Ensure treated and control groups are comparable:
```python
import matplotlib.pyplot as plt

# Check propensity score overlap
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X, T)
propensity = lr.predict_proba(X)[:, 1]

plt.hist(propensity[T==1], alpha=0.5, label='Treated')
plt.hist(propensity[T==0], alpha=0.5, label='Control')
plt.legend()
plt.title('Propensity Score Overlap')
plt.show()
```

### 2. Use Cross-Validation

More folds = more robust but slower:
```python
# For small datasets (< 5000)
dml = DoubleMLPipeline(n_folds=3)

# For medium datasets (5000-50000)
dml = DoubleMLPipeline(n_folds=5)

# For large datasets (> 50000)
dml = DoubleMLPipeline(n_folds=10)
```

### 3. Validate on Holdout Set

If you have true effects (from RCT or simulation):
```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test = train_test_split(X, test_size=0.3)
T_train, T_test = train_test_split(T, test_size=0.3)
Y_train, Y_test = train_test_split(Y, test_size=0.3)

# Fit on training
clan.fit(X_train, T_train, Y_train)

# Evaluate on test
should_target = clan.predict_high_benefit(X_test)
actual_high = (true_tau[X_test.index] > median_effect)
accuracy = (should_target == actual_high).mean()
print(f"Targeting accuracy: {accuracy:.2%}")
```

### 4. Consider Ethical Implications

- **Fairness**: Check if targeting excludes protected groups
- **Transparency**: Document why certain groups are prioritized
- **Evaluation**: Plan to evaluate actual outcomes
- **Adaptation**: Be ready to adjust based on results

## Troubleshooting

### Issue: Very high standard errors
**Solution**: Increase sample size or reduce number of features

### Issue: GATES groups all similar
**Solution**: Try different number of groups or check if effects are truly homogeneous

### Issue: CLAN predicts everyone as high benefit
**Solution**: Check if there's real heterogeneity, adjust threshold

### Issue: Negative treatment effects
**Solution**: This is informative! Some groups may be harmed—exclude them from treatment

## Next Steps

1. **Explore the examples**: Run `python examples/quick_demo.py`
2. **Try with your data**: Adapt the code to your use case
3. **Read the papers**: See references in README.md
4. **Contribute**: Submit issues or PRs on GitHub

## Additional Resources

- [Double ML Paper](https://arxiv.org/abs/1608.00060)
- [GenericML Paper](https://arxiv.org/abs/2004.14497)
- [Causal Inference Book](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

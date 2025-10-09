# Causal Machine Learning for Estimating Treatment Effects

A reproducible Python library for causal inference using state-of-the-art machine learning methods. This package enables policymakers and researchers to identify heterogeneous treatment effects and target interventions to the highest-benefit populations.

## Features

- **Double ML (Double Machine Learning)**: Robust treatment effect estimation using cross-fitting and orthogonalization
- **GATES (Group Average Treatment Effects)**: Identify heterogeneous effects across subpopulations
- **CLAN (Classification Analysis)**: Machine learning-based targeting to identify high-benefit individuals
- **Policy-Relevant Applications**: Designed for real-world scenarios like targeting 100,000+ households

## Installation

```bash
# Clone the repository
git clone https://github.com/aeabdou/Causal-Machine-Learning-for-Estimating-Treatment-Effect.git
cd Causal-Machine-Learning-for-Estimating-Treatment-Effect

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from causal_ml import DoubleMLPipeline, GATES, CLAN
from causal_ml.utils import create_policy_relevant_data

# Generate synthetic data (100,000 households)
X, T, Y, true_tau = create_policy_relevant_data(n_samples=100000)

# 1. Estimate Average Treatment Effect with Double ML
dml = DoubleMLPipeline(n_folds=5, random_state=42)
dml.fit(X, T, Y)
print(dml.get_ate())

# 2. Analyze heterogeneity with GATES
gates = GATES(n_groups=5, n_folds=5, random_state=42)
gates.fit(X, T, Y)
print(gates.summary())

# 3. Target high-benefit households with CLAN
clan = CLAN(n_folds=5, random_state=42)
clan.fit(X, T, Y)
should_target, stats = clan.get_targeting_strategy(X, budget=100000)
print(f"Targeting {stats['targeted']:,} households")
```

## Complete Example

Run the comprehensive policy targeting example:

```bash
python examples/policy_targeting_example.py
```

This example demonstrates:
- Generating realistic household intervention data
- Estimating robust treatment effects with Double ML
- Identifying high-benefit groups with GATES
- Creating an optimal targeting strategy with CLAN
- Evaluating targeting performance
- Visualizing results

## Methods Overview

### Double ML

Double Machine Learning provides robust estimates of treatment effects by:
1. Using machine learning to predict outcomes and treatment assignment
2. Employing cross-fitting to avoid overfitting bias
3. Orthogonalizing the estimation to reduce regularization bias

**Key benefits:**
- Robust to model misspecification
- Allows flexible ML models
- Valid inference with confidence intervals

### GATES (Group Average Treatment Effects)

GATES analyzes heterogeneous treatment effects by:
1. Grouping observations based on baseline characteristics
2. Estimating average effects within each group
3. Identifying which groups benefit most from treatment

**Key benefits:**
- Reveals heterogeneity patterns
- Easy to interpret for policymakers
- Highlights high-gain segments

### CLAN (Classification Analysis)

CLAN uses machine learning to identify high-benefit individuals by:
1. Estimating individual treatment effects
2. Training a classifier to predict high vs. low benefit
3. Providing targeting recommendations with budget constraints

**Key benefits:**
- Actionable targeting recommendations
- Handles budget constraints (e.g., 100,000 households)
- Identifies key characteristics of high-benefit individuals

## Project Structure

```
.
├── causal_ml/              # Main package
│   ├── __init__.py
│   ├── double_ml/          # Double ML implementation
│   │   ├── __init__.py
│   │   └── double_ml_pipeline.py
│   ├── generic_ml/         # GenericML (GATES, CLAN)
│   │   ├── __init__.py
│   │   ├── gates.py
│   │   └── clan.py
│   └── utils/              # Utilities
│       ├── __init__.py
│       └── data_generation.py
├── examples/               # Example scripts
│   └── policy_targeting_example.py
├── tests/                  # Unit tests
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Use Cases

### Policy Intervention Targeting

Identify which households would benefit most from assistance programs when budget constraints exist.

```python
# Target top 100,000 households
should_target, stats = clan.get_targeting_strategy(X, budget=100000)
```

### Personalized Medicine

Determine which patients would benefit most from a new treatment.

### Marketing Optimization

Identify customer segments most responsive to marketing campaigns.

### Education Policy

Target educational interventions to students who would benefit most.

## API Reference

### DoubleMLPipeline

```python
DoubleMLPipeline(
    ml_model_outcome=None,      # ML model for outcome prediction
    ml_model_treatment=None,    # ML model for treatment prediction
    n_folds=5,                  # Cross-fitting folds
    random_state=42             # Random seed
)
```

**Methods:**
- `fit(X, T, Y)`: Fit the model
- `get_ate()`: Get average treatment effect
- `summary()`: Print summary table

### GATES

```python
GATES(
    n_groups=5,                 # Number of groups
    ml_model=None,              # ML model for predictions
    n_folds=5,                  # Cross-fitting folds
    random_state=42             # Random seed
)
```

**Methods:**
- `fit(X, T, Y)`: Fit the model
- `get_gates()`: Get group-level estimates
- `identify_high_benefit_groups()`: Identify top groups
- `summary()`: Print summary

### CLAN

```python
CLAN(
    ml_model_cate=None,         # ML model for CATE estimation
    ml_model_classifier=None,   # ML model for classification
    n_folds=5,                  # Cross-fitting folds
    random_state=42             # Random seed
)
```

**Methods:**
- `fit(X, T, Y)`: Fit the model
- `predict_high_benefit(X)`: Predict high-benefit individuals
- `get_targeting_strategy(X, budget)`: Get targeting recommendations
- `get_feature_importance()`: Get important features
- `summary()`: Print summary

## References

1. Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). **Double/debiased machine learning for treatment and structural parameters.** *The Econometrics Journal*, 21(1), C1-C68.

2. Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2020). **Generic machine learning inference on heterogeneous treatment effects in randomized experiments.** *arXiv preprint arXiv:2004.14497*.

3. Athey, S., & Imbens, G. W. (2016). **Recursive partitioning for heterogeneous causal effects.** *Proceedings of the National Academy of Sciences*, 113(27), 7353-7360.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Citation

If you use this code in your research, please cite:

```bibtex
@software{causal_ml_treatment_effects,
  author = {Abdou, AE},
  title = {Causal Machine Learning for Estimating Treatment Effects},
  year = {2025},
  url = {https://github.com/aeabdou/Causal-Machine-Learning-for-Estimating-Treatment-Effect}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.
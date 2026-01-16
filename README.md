# HTE-Pipeline-Walkthrough

This repository contains a comprehensive framework for estimating **Heterogeneous Treatment Effects (HTE)** using machine learning.

The project is structured as both a **functional tool** for researchers and a **step-by-step educational guide** that deconstructs the methodology. It implements a split-sample (cross-fitting) approach to estimate Conditional Average Treatment Effects (CATE) and performs statistical inference using BLP, GATES, and CLAN.

## Repository Structure

This repository consists of two main Jupyter Notebooks:

### 1. `Research_Core.ipynb` (The Step-by-Step Guide)
**Start here for the explanation.** This notebook serves as the documentation and logic breakdown for the project. It walks through the entire process in a linear, readable format:
* **Data Preparation:** Detailed cleaning steps, handling missing values, and encoding specific covariates (e.g., `totalincome`, `ladder`).
* **Proxy Construction:** Explains how to use ML models (like XGBoost) to create a proxy for the individual treatment effect score, $S(Z)$.
* **Visualizations:** Includes initial plots to visualize the distribution of treatment effects before running the full loop.

### 2. `Research_Loop.ipynb` (The Main Function)
**Use this for production/analysis.**
This notebook contains the aggregated `estimate_heterogeneity` function. It takes the logic established in the *Core* file and operationalizes it:
* **Automated Loop:** Iteratively splits the data into "Auxiliary" (training) and "Main" (inference) folds.
* **Stability:** Averages results across multiple random splits to ensure robust estimates.
* **Inference Outputs:** Automatically calculates and returns:
    * **BLP (Best Linear Predictor):** Tests if heterogeneity exists.
    * **GATES (Group Average Treatment Effects):** Treatment effects sorted by groups (e.g., top 20% vs bottom 20%).
    * **CLAN (Classification Analysis):** Characteristics of the most/least affected units.

## Methodology

The pipeline relies on a **Cross-Fitting** strategy to prevent overfitting:

1.  **Split Data:** The dataset is divided into two random folds: *Auxiliary* and *Main*.
2.  **Train Models (Auxiliary):** We train two machine learning models (e.g., XGBoost, CatBoost, or Neural Nets) on the auxiliary fold:
    * $\hat{\mu}_1(Z)$: Predicted outcome *with* treatment.
    * $\hat{\mu}_0(Z)$: Predicted outcome *without* treatment.
3.  **Compute Proxy (Main):** We apply these models to the *Main* fold to generate a predicted treatment effect score for each unit:
    $$S(Z) = \hat{\mu}_1(Z) - \hat{\mu}_0(Z)$$
4.  **Estimate Effects:** We sort units in the *Main* fold based on this score to identify which subgroups benefit most or least from the intervention.

## Dependencies
* Python 3.x
* `pandas`, `numpy`
* `statsmodels`
* `scikit-learn`
* `xgboost`, `catboost`
* `tensorflow` (if using Neural Net options)

# HTE-Pipeline-Walkthrough

This repository contains a comprehensive framework for estimating **Heterogeneous Treatment Effects (HTE)** using machine learning.

The project is designed to be both a reusable tool and an educational resource. It provides a standalone function for causal inference, a step-by-step logic walkthrough, and a full real-world application analyzing poverty alleviation interventions.

## Repository Structure

### 1. `hte_function.py` (The Tool)
* **Contains:** The standalone, reusable function for estimating heterogeneity.
* **Usage:** Import this file to apply the BLP, GATES, and CLAN methodologies to your own datasets without re-writing the pipeline.

### 2. `Research_Loop.ipynb` (The Application & Case Study)
* **Context:** A complete end-to-end analysis focused on **Poverty Alleviation**.
* **Workflow:**
    * **EDA:** Exploratory Data Analysis of the poverty indicators.
    * **Application:** Applies the core HTE function to the data.
    * **Predictions & Policy:** Analyzes the results to predict intervention impacts and derive policy insights.
* **Advanced Features:** Includes the **Automated Covariate Ranking Algorithm** to diagnose drivers of heterogeneity.

### 3. `Research_Core.ipynb` (The Educational Guide)
* **Context:** A "textbook" style breakdown of the methodology.
* **Content:** Deconstructs the main function into a linear, step-by-step explanation.
* **Key Concepts:** Explains the data cleaning logic, the construction of the ML proxy ($S(Z)$), and the cross-fitting strategy used to avoid overfitting.

## Methodology

The pipeline implements a **Cross-Fitting T-Learner** strategy to prevent overfitting and robustly identify treatment heterogeneity.

### 1. The T-Learner Approach
Instead of using a single model with a treatment dummy (S-Learner), this pipeline fits two separate machine learning models:
* **Split Data:** The dataset is divided into *Auxiliary* (training) and *Main* (inference) folds.
* **Train Models:** We train ML models (XGBoost, CatBoost, or Neural Nets) on the auxiliary fold:
    * $\hat{\mu}_1(Z)$: Learns the mapping from covariates to the outcome for the **treated** group.
    * $\hat{\mu}_0(Z)$: Learns the mapping from covariates to the outcome for the **control** group.
* **Compute Proxy:** We apply these models to the *Main* fold to generate a predicted treatment effect score for each unit:
    $$S(Z) = \hat{\mu}_1(Z) - \hat{\mu}_0(Z)$$

### 2. Inference Metrics
* **BLP (Best Linear Predictor):** Tests if the heterogeneity found is statistically significant.
* **GATES (Group Average Treatment Effects):** Sorts units by their $S(Z)$ score and compares the Average Treatment Effect (ATE) of the top 20% vs. the bottom 20%.
* **CLAN (Classification Analysis):** Compares the average characteristics of the most and least affected groups.

### 3. Automated Driver Analysis (Ranking Algorithm & Stability)
A key feature of the `Research_Loop` is the **Covariate Importance Ranking Algorithm**. This goes beyond simple detection to explain *why* heterogeneity exists and ensures the findings are robust:
1.  **Sort:** Units are classified into "Most Affected" and "Least Affected" groups based on their T-Learner score.
2.  **Compare:** The algorithm iterates through every covariate to calculate the difference in means between these two groups.
3.  **Test & Rank:** It performs hypothesis testing (p-value < 0.05) to filter significant drivers and ranks them by magnitude.
4.  **Stability Tests:** To ensure the results are not statistical noise, the pipeline iterates this entire process across multiple random splits (e.g., 100+ runs). It aggregates the results to identify drivers that are **consistently significant** across different data partitions, filtering out unstable predictors.

## Dependencies
* Python 3.x
* `pandas`, `numpy`
* `statsmodels`
* `scikit-learn`
* `xgboost`, `catboost`
* `tensorflow` (if using Neural Net options)

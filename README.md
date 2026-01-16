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
* **Goal:** Demonstrates how the tool works in a complex, real-world research scenario.

### 3. `Research_Core.ipynb` (The Educational Guide)
* **Context:** A "textbook" style breakdown of the methodology.
* **Content:** Deconstructs the main function into a linear, step-by-step explanation.
* **Key Concepts:** Explains the data cleaning logic, the construction of the ML proxy ($S(Z)$), and the cross-fitting strategy used to avoid overfitting.

## Methodology

The pipeline implements a **Cross-Fitting** strategy to prevent overfitting:

1.  **Split Data:** The dataset is divided into *Auxiliary* (training) and *Main* (inference) folds.
2.  **Train Models:** We train ML models (XGBoost, CatBoost, etc.) on the auxiliary fold to predict outcomes with ($\hat{\mu}_1$) and without ($\hat{\mu}_0$) treatment.
3.  **Compute Proxy:** We generate a predicted treatment effect score for the main fold:
    $$S(Z) = \hat{\mu}_1(Z) - \hat{\mu}_0(Z)$$
4.  **Inference:**
    * **BLP:** Best Linear Predictor for heterogeneity.
    * **GATES:** Group Average Treatment Effects (e.g., top 20% vs bottom 20%).
    * **CLAN:** Characteristics of the most/least affected groups.

## Dependencies
* Python 3.x
* `pandas`, `numpy`
* `statsmodels`
* `scikit-learn`
* `xgboost`, `catboost`
* `tensorflow` (if using Neural Net options)  

# HTE-Inference-Engine: Poverty Alleviation Analysis

This repository contains an advanced causal machine learning framework developed for the **"Door of Hope" (Bab Amal)** poverty alleviation project at the University of Minnesota.

[cite_start]The system implements the "Generic Machine Learning" (GML) inference strategy proposed by **Chernozhukov, Demirer, Duflo, and Fernández-Val (2023)**[cite: 2, 88]. [cite_start]It is designed to move beyond simple Average Treatment Effects (ATE) to rigorously identify, validate, and explain **Heterogeneous Treatment Effects (HTE)** in high-dimensional experimental data[cite: 71, 72, 88].

---

## The Pipeline Logic

[cite_start]The engine follows a four-stage process to ensure that the discovered heterogeneity is a robust economic signal rather than statistical noise[cite: 31, 116].

### 1. Multi-Model Estimation (Cross-Fitting)
The framework architects a **Cross-Fitted T-Learner** structure. [cite_start]By splitting the data into auxiliary and main folds, the system prevents overfitting while benchmarking diverse base estimators—including **CatBoost, XGBoost, and Neural Networks**[cite: 31, 116, 117]. [cite_start]Each model estimates the Conditional Average Treatment Effect (CATE) by decoupling the treatment and control response functions to calculate the proxy score $S(Z)$[cite: 94, 231, 232].



### 2. Signal Validation (BLP & GATES)
Before a specific estimator is selected, the system benchmarks them using two primary diagnostics:
* [cite_start]**Best Linear Predictor (BLP):** The engine regresses actual outcomes on the ML-generated proxy scores to test for the **presence of heterogeneity**[cite: 95, 309]. [cite_start]This determines if the ML model has captured a statistically significant signal where $\beta_2 \neq 0$[cite: 320].
* [cite_start]**Sorted Group Average Treatment Effects (GATES):** The population is stratified into quintiles based on predicted responsiveness[cite: 96, 449]. [cite_start]This allows for a comparison of the impact on the "Most Affected" vs. "Least Affected" groups to identify which ML architecture maximizes causal robustness[cite: 105, 451].

### 3. Segment Characterization (CLAN)
[cite_start]Once the optimal estimator is identified, the system applies **Classification Analysis (CLAN)**[cite: 106, 511]. [cite_start]This step isolates the key drivers of intervention effectiveness by characterizing the divergence in baseline covariates between the top and bottom quintiles[cite: 512, 514]. [cite_start]It answers the critical policy question: *What socio-economic traits define the households that benefit most?*[cite: 514].

### 4. Novel Stability-Based Feature Ranking
This repository extends the Chernozhukov et al. framework by introducing a **Novel T-Learner Ranking Algorithm**. This system:
* Distills high-dimensional CATE scores into a simplified proxy model.
* Aggregates feature importance and rankings across hundreds of cross-fitting folds.
* Isolates robust drivers by filtering for variables that maintain structural stability across different random partitions of the data, ensuring findings are not driven by a single "lucky" split.

---

## Repository Structure

### `Methodology_Walkthrough.ipynb`
A linear, educational breakdown of the research logic. It covers data engineering for complex poverty indicators (e.g., assets, education) and the manual construction of the HTE proxy $S(Z)$.

### `End_to_End_Analysis.ipynb`
The production-grade research loop. This notebook automates the multi-model benchmarking, stability testing, and final driver ranking used for the "Door of Hope" project.

### `causal_ml_hte.py`
The core inference engine. A modular Python library containing the T-Learner architecture and the statistical functions for BLP, GATES, and CLAN inference.

---

## Methodology Summary

[cite_start]The pipeline implements a **Cross-Fitting T-Learner** strategy to achieve validity[cite: 31, 116]:

1.  [cite_start]**Split Data:** Divide dataset into *Auxiliary* (training) and *Main* (inference) folds[cite: 229].
2.  [cite_start]**Train Models:** Train ML models on the auxiliary fold to predict outcomes $\hat{\mu}_1$ and $\hat{\mu}_0$[cite: 231, 232].
3.  [cite_start]**Compute Proxy:** Generate the predicted treatment effect score $S(Z) = \hat{\mu}_1(Z) - \hat{\mu}_0(Z)$[cite: 232].
4.  [cite_start]**Inference:** Execute BLP, GATES, and CLAN to validate and profile heterogeneity[cite: 235, 236, 237, 238].

---

## References
* **Chernozhukov, V., Demirer, M., Duflo, E., & Fernández-Val, I. (2023).** "Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments, with an Application to Immunization in India." [cite_start]*NBER Working Paper Series*[cite: 2, 21].

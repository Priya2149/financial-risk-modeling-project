# Financial Risk Modeling Project

This repository showcases my hands-on work in **quantitative research and financial risk modeling**, inspired by the **JPMorgan Chase Quantitative Research Virtual Experience Program (Forage)**.  

The project demonstrates practical applications of data science and machine learning in finance, covering:
- Time series forecasting of commodity prices  
- Pricing of storage contracts under operational constraints  
- Credit risk modeling (probability of default and expected loss)  
- FICO score bucketing (quantization) for credit ratings  

---

## Project Overview

### Task 1 – Natural Gas Price Forecasting
- Forecast monthly natural gas prices and extrapolate 12 months ahead.  
- Models: OLS regression with seasonal dummies, Holt-Winters Exponential Smoothing (ETS).  
- Includes backtesting (RMSE, MAPE), automatic cleaning/interpolation, and an `estimate(date)` function.  
- Visualizations: history, seasonality, backtest curve, and forecast fan chart.  

### Task 2 – Gas Storage Contract Pricing
- Prototype pricer for a gas storage contract under constraints.  
- Inputs: injection/withdrawal dates, max rates, storage capacity, storage cost.  
- Outputs: daily cashflow ledger + total contract value (PV).  
- Integrates with Task 1 estimator to price gas on arbitrary dates.  

### Task 3 – Loan Default Probability & Expected Loss
- Predict probability of default (PD) from borrower features.  
- Models: Logistic Regression (baseline) and Decision Tree (comparison).  
- Evaluation: AUC and classification metrics.  
- Function `expected_loss(features, exposure)` → returns PD and EL.  
- Formula:  
  \[
  \text{Expected Loss} = PD \times (1 - \text{Recovery Rate}) \times \text{Exposure}, \quad \text{with Recovery Rate = 10%}
  \]  

### Task 4 – FICO Score Quantization & Rating Map
- Group borrower FICO scores into discrete rating buckets.  
- Methods: Dynamic Programming with two objectives:  
  - **MSE** – minimize within-bucket squared error.  
  - **Log-Likelihood** – maximize fit of defaults per bucket.  
- Output: bucket boundaries, summary table (counts, defaults, PD), and a function `map_fico_to_rating(fico)`.  
- Ratings: **1 = best (highest FICO), N = worst**.  

---

## Libraries Used  

<p align="left">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="pandas"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="numpy"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white" alt="matplotlib"/>
</p>

---

### Key Skills
- Time series forecasting & backtesting
- Storage contract pricing under real-world constraints
- Credit risk modeling (PD, EL, LGD)
- Score quantization & dynamic programming

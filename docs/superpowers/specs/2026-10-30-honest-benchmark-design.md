# Honest Time-Series Forecasting Benchmark — Fairness, Protocol & Model Lineup Redesign

**Date:** 2026-10-30  
**Scope:** `README.md`, `UCI-Bike-Sharing.ipynb`, `Electricity-Load.ipynb`, `M5-Forecasting.ipynb`, and Google Drive notebook sync (`Time-Series-Forecasting-Honest-Comparison`, Folder ID `1MLr_Hw_r0A3nLIsEY4mkJtWIsqvd4UaS`).

---

## 1. Problem Statement & Motivation

An external methodological review and a full codebase audit surfaced four concrete gaps between `README.md` and the three benchmark notebooks:

1. **Misstated "Identical Feature Inputs & Tuning Budget" Claim:**  
   `README.md` claims every model gets *"the exact same feature inputs"* and *"the same tuning budget"*. Methodologically, forcing a single tabular feature matrix onto heterogeneous model families is counterproductive: GBDTs require explicit lag/calendar encodings, statistical state-space models decompose raw series natively, and neural sequence architectures expect contiguous chronological windows. Worse, taking "identical feature inputs" literally in the notebooks led to feeding a 14–19 column tabular feature row `X_train` (`hour`, `is_weekend`, `lag_1`, `sin_hour`, etc.) into inline PyTorch `DLinear`, `N-BEATS`, and `PatchTST` modules — causing `DLinear`'s `AvgPool1d` to smooth across unrelated feature columns rather than time.
2. **Validation-Peeking Asymmetry & Unused `test_df`:**  
   All three notebooks partition data chronologically into `train_df` (70%), `val_df` (15%), and `test_df` (15%), yet previously evaluated leaderboards on `val_df` — the exact split GBDTs and neural models monitored for early stopping — while `test_df` sat unused.
3. **Hardcoded SARIMAX Orders & Inconsistent Rolling Evaluation:**  
   Despite headings referencing `AutoARIMA`, all three notebooks hardcoded `statsmodels` `SARIMAX` orders (`(1,0,1)` or `(1,1,1)`). Furthermore, while `UCI-Bike-Sharing.ipynb` used Kalman state filtering (`.extend()`) for 1-step-ahead rolling parity, `Electricity-Load.ipynb` and `M5-Forecasting.ipynb` ran unassisted multi-step `predict()` across thousands of steps against 1-step-ahead GBDTs and neural models.
4. **Lineup & Documentation Drift:**  
   `NHITS` and `N-BEATSx` (critical for multi-rate seasonal pooling and exogenous covariates) were missing from the notebooks (`NHITS` was claimed in `README.md` but absent in code); `UCI-Bike-Sharing.ipynb` omitted `LightGBM` and `CatBoost`; `Chronos-Bolt` cells in `Electricity-Load.ipynb` and `M5-Forecasting.ipynb` were unexecuted; and `README.md` referenced unimplemented components (`darts`, `Prophet`, `TBATS`, `TimesFM`, YAML CLI, Streamlit/FastAPI demo).

---

## 2. Core Fairness Contract

Instead of claiming identical feature matrices or identical hyperparameter search workloads across incompatible paradigms, the benchmark enforces four strict invariants:

1. **Identical Causal Information Cutoff ($\mathcal{F}_{t-1}$):**  
   When predicting target $y_t$ at any test timestamp $t \in \text{test\_df}$, every model has access strictly to target history $y_{1:t-1}$ and known calendar/exogenous covariates up to the causal cutoff, with **zero parameter refitting** on `test_df`.
2. **Paradigm-Native Input Representations:**  
   Each model tier consumes $\mathcal{F}_{t-1}$ in the representation its architecture was designed for:
   - *Statistical state-space / decomposition (`statsforecast`):* Raw continuous series $y$ with native seasonal cycle length(s) (plus exogenous regressors $X$ for `AutoARIMA`).
   - *Regularized linear & GBDTs:* Engineered autoregressive lags, shifted rolling statistics, cyclical Fourier/trigonometric calendar encodings, and exogenous features.
   - *Sequence deep learning (`neuralforecast`):* Contiguous raw lookback windows $y_{t-L:t-1}$ (`DLinear`, `PatchTST`, `NHITS`, `NBEATSx`), augmented with exogenous covariates (`futr_exog_list`) on architectures that natively support them (`NBEATSx`, `NHITS`).
   - *Zero-shot foundation (`Chronos-Bolt`):* Raw univariate context window $y_{t-512:t-1}$.
3. **Strict Three-Way Chronological Split (`train_df` 70% / `val_df` 15% / `test_df` 15%):**  
   - `train_df` (70%) + `val_df` (15%) form the pre-test development horizon (85%).
   - `val_df` serves exclusively as the internal validation tail for iterative early stopping (`LightGBM`, `XGBoost`, `CatBoost`, and `NeuralForecast`) and Ridge `alpha` selection.
   - **`test_df` (final 15%)** is the sole blind evaluation split reported on the leaderboard. Every model observes true target values from `val_df` right up to $t-1$ across the `val_df` $\rightarrow$ `test_df` boundary.
4. **Disclosed Selection & Compute Budget:**  
   Every tier runs within the repository's 120-second per-model wall-clock budget on Colab GPU/CPU, using its standard selection mechanism rather than asymmetric manual tuning:
   - `AutoARIMA`: Hyndman–Khandakar stepwise AICc search over $(p, d, q)(P, D, Q)_s$ on pre-test data (printing selected orders).
   - `AutoETS` / `AutoTheta` / `MSTL`: Maximum-likelihood state-space & LOESS decomposition on pre-test data.
   - `RidgeCV`: Closed-form LOO/GCV selection over `alphas=[0.1, 1.0, 10.0, 100.0]`.
   - `GBDTs`: Fixed standard capacity (`learning_rate=0.03`, `num_leaves=31` / `max_depth=6`, `n_estimators=1000`) with validation early stopping (`stopping_rounds=30` on `val_df`).
   - `NeuralForecast`: Fixed optimizer/step budget (`max_steps=500`, `val_check_steps=25`, `early_stop_patience_steps=3` on `val_size=len(val_df)`).
   - `Chronos-Bolt`: Zero-shot pretrained inference (`amazon/chronos-bolt-tiny`, zero fine-tuning).

---

## 3. Standardized Four-Tier Model Lineup

All three notebooks evaluate the exact same 17-model lineup across four tiers at horizon $h=1$ (rolling 1-step-ahead over `test_df` with `refit=False`):

| Tier | Model | Library / Implementation | Input Representation | Selection / Regularization |
|---|---|---|---|---|
| **Tier 1: Classical & Statistical** | `Naive (t-1)` | Vectorized NumPy | $y_{t-1}$ | Parameter-free |
| | `Smoothed Naive (2-Step)` | Vectorized NumPy | $(y_{t-1} + y_{t-2}) / 2$ | Parameter-free |
| | `Seasonal Naive (s1)` | Vectorized NumPy | $y_{t-24}$ (hourly) / $y_{t-7}$ (daily) | Parameter-free |
| | `Seasonal Naive (s2)` | Vectorized NumPy | $y_{t-168}$ (hourly) / $y_{t-28}$ (daily) | Parameter-free |
| | `Ridge Regression` | `sklearn.linear_model.RidgeCV` | Tabular `X` (`train + val` fit, `test` eval) | `alphas=[0.1, 1.0, 10.0, 100.0]` + `StandardScaler` |
| | `AutoETS` | `statsforecast.models.AutoETS` | Univariate $y$ (`season_length=s1`) | AICc state-space selection (`refit=False` forward filter) |
| | `AutoTheta` | `statsforecast.models.AutoTheta` | Univariate $y$ (`season_length=s1`) | MLE Theta decomposition (`refit=False` forward filter) |
| | `MSTL` | `statsforecast.models.MSTL` | Univariate $y$ (`season_length=[s1, s2]`, `AutoETS("ZZN")` trend) | Multi-seasonal LOESS + ETS (`refit=False` forward filter) |
| | `AutoARIMA` | `statsforecast.models.AutoARIMA` | $y$ + dataset `exog_cols` (`season_length=s1`) | Stepwise AICc order search (`refit=False` Kalman forward filter) |
| **Tier 2: Tabular GBDTs** | `LightGBM Regressor` | `lightgbm.LGBMRegressor` | Tabular `X` | `early_stopping(30)` on `val_df` |
| | `XGBoost Regressor` | `xgboost.XGBRegressor` | Tabular `X` (`tree_method="hist"`) | `early_stopping_rounds=30` on `val_df` |
| | `CatBoost Regressor` | `catboost.CatBoostRegressor` | Tabular `X` | `early_stopping_rounds=30` on `val_df` |
| **Tier 3: Sequence Deep Learning** | `DLinear` | `neuralforecast.models.DLinear` | Raw window $y_{t-L:t-1}$ (univariate) | Moving-avg decomp + `early_stop_patience_steps=3` on `val_df` |
| | `PatchTST` | `neuralforecast.models.PatchTST` | Raw window $y_{t-L:t-1}$ (univariate) | Subseries patching + `early_stop_patience_steps=3` on `val_df` |
| | `NBEATSx` | `neuralforecast.models.NBEATSx` | Window $y_{t-L:t-1}$ + `futr_exog_list` | Doubly-residual basis + exog + early stopping on `val_df` |
| | `NHITS` | `neuralforecast.models.NHITS` | Window $y_{t-L:t-1}$ + `futr_exog_list` | Multi-rate pooling + interpolation + early stopping on `val_df` |
| **Tier 4: Foundation Model** | `Chronos-Bolt (Zero-Shot)` | `chronos.ChronosBoltPipeline` | Raw window $y_{t-512:t-1}$ | Zero-shot median quantile (`amazon/chronos-bolt-tiny`) |

*(Total: 17 rows on each dataset leaderboard: 9 Tier 1, 3 Tier 2, 4 Tier 3, 1 Tier 4.)*

### Scope Boundaries Documented in `README.md`
* **Included neural families:** Linear decomposition (`DLinear`), basis expansion with exogenous drivers (`NBEATSx`), multi-rate hierarchical interpolation (`NHITS`), channel-independent patched self-attention (`PatchTST`), and pretrained tokenized foundation models (`Chronos-Bolt`).
* **Excluded neural families (with explicit rationale):**
  * *GluonTS `DeepAR` / `DeepState`:* Originating from Amazon Research's GluonTS ecosystem (not Meta's Prophet or PyMC), these autoregressive RNN / linear-Gaussian state-space architectures target global probabilistic training across thousands of disaggregate series; for single-series / aggregate point-forecasting leaderboards, they add heavy framework friction without outperforming `NHITS`, `PatchTST`, or GBDTs.
  * *Temporal Convolutional Networks (`TCN`):* Dilated causal convolutions are a classic sequence modeling family, but post-2023 time-series benchmarks establish that `NHITS` (multi-rate pooling) and `PatchTST` / `DLinear` dominate the accuracy-compute frontier that TCNs previously occupied.

---

## 4. Dataset-Specific Parameters & API Safeguards

### 4.1 Dataset Parameter Matrix

| Parameter | `UCI-Bike-Sharing.ipynb` | `Electricity-Load.ipynb` | `M5-Forecasting.ipynb` |
|---|---|---|---|
| **Time Column / `freq`** | `timestamp` / `"h"` | `timestamp` / `"h"` | `date` / `"D"` |
| **Target Column** | `cnt` | `load` | `national_sales` |
| **Seasonalities `[s1, s2]`** | `[24, 168]` | `[24, 168]` | `[7, 28]` |
| **`statsforecast` Pre-Test Fit Tail** | Last `2,016` hrs (12 weeks) of `train+val` | Last `2,016` hrs (12 weeks) of `train+val` | Full `1,626` days of `train+val` |
| **`AutoARIMA` Exogenous (`exog_cols`)** | `["workingday", "holiday", "temp", "hum", "sin_hr", "cos_hr"]` | `["is_weekend", "sin_hour", "cos_hour", "sin_wday", "cos_wday"]` | `["snap_active_states", "any_snap", "sin_wday", "cos_wday"]` |
| **`neuralforecast` `input_size` ($L$)** | `168` (1 week) | `168` (1 week) | `56` (8 weeks) |
| **`NBEATSx` / `NHITS` `futr_exog_list`** | `["workingday", "holiday", "weathersit", "temp", "atemp", "hum", "windspeed", "sin_hr", "cos_hr"]` | `["is_weekend", "sin_hour", "cos_hour", "sin_wday", "cos_wday"]` | `["snap_active_states", "any_snap", "sin_wday", "cos_wday", "sin_month", "cos_month"]` |
| **Primary Sort Metric** | `MAE`, `MASE` (added for parity) | `MAE (kW)`, `MASE` | `RMSSE`, `MAE` |

*(Note on Bike Sharing weather covariates: Because `Ridge`, `AutoARIMA`, and Tier 2 GBDTs treat meteorological columns (`temp`, `atemp`, `hum`, `windspeed`, `weathersit`) in `X_test` as hourly weather-forecast covariates known for target hour $t$, passing those exact same covariates in `futr_exog_list` for `NBEATSx` and `NHITS` guarantees strict covariate parity across Tiers 1–3. In `UCI-Bike-Sharing.ipynb`, we also add `MASE` alongside `MAE`, `RMSE`, and `sMAPE (%)` so all three notebooks report a scale-free metric.)*

### 4.2 Critical Implementation & Compatibility Safeguards
1. **Separate Univariate vs. Exogenous `StatsForecast` Calls:**  
   `AutoETS`, `AutoTheta`, and `MSTL` are strictly univariate (`X` unsupported) and will error if extra exogenous columns exist in `df`. Each statistical model is executed in its own timed `StatsForecast` wrapper: `AutoETS`, `AutoTheta`, and `MSTL` receive `['unique_id', 'ds', 'y']`, while `AutoARIMA` receives `['unique_id', 'ds', 'y'] + exog_cols`. To log the selected ARIMA order reliably across `statsforecast` versions, we call `.fit()` on the pre-test window to log `arima_string(sf.fitted_[0, 0].model_)` and run `.cross_validation(..., h=1, n_windows=len(test_df), step_size=1, refit=False)` on `pre_test + test` to generate the 1-step rolling predictions.
2. **Separate Per-Model `NeuralForecast` Calls & Exogenous Flags:**  
   In `neuralforecast`, `DLinear` and `PatchTST` set `EXOGENOUS_FUTR = False` (strictly univariate), whereas `NBEATSx` and `NHITS` support `futr_exog_list` and `hist_exog_list`. Each of the four neural models is instantiated and evaluated in an individual timed loop (`nf = NeuralForecast(models=[model], freq=freq); cv_df = nf.cross_validation(df=..., val_size=len(val_df), test_size=len(test_df), step_size=1, refit=False)`) so per-model `Fit Time (s)` is measured accurately and univariate models receive only `['unique_id', 'ds', 'y']`.
3. **Pandas 2.2+ Frequency Compatibility:**  
   Use lowercase `"h"` (never deprecated `"H"`) for hourly datasets and `"D"` for daily M5 so `StatsForecast` and `NeuralForecast` run cleanly on current Google Colab runtimes.
4. **Non-Negative Clipping & Denominator Scaling:**  
   All demand/load predictions apply `np.maximum(0.0, y_pred)` prior to metric calculation. Scale-free denominators (`in_sample_mae_naive` for `MASE`, `naive_denom` for `RMSSE`) are computed strictly on pre-test history (`train_df + val_df`).
5. **Clean Pre-Execution State:**  
   Setup cells (`Cell 1`) in all three notebooks are updated to install missing packages cleanly (`statsforecast`, `neuralforecast`, `catboost`, `chronos-forecasting`, `accelerate`). Modified Phase 4 code cells have stale `val_df` outputs cleared (`outputs: []`, `execution_count: null`) so there is zero mismatch between code (`test_df` evaluation) and legacy cached outputs prior to re-running in Colab.

---

## 5. `README.md` Structure & Honest Alignment

`README.md` will be rewritten to match the repository accurately:
1. **The Big Picture: When Baselines Win and When They Don't** — Retain clear practical framing while grounding claims in the 4-tier architecture.
2. **How This Benchmark Defines Fairness** — Explain the $\mathcal{F}_{t-1}$ causal cutoff, paradigm-native input representations, 70/15/15 chronological split (`val_df` for early stopping, `test_df` for blind 1-step rolling evaluation with `refit=False`), and exact per-tier selection protocols.
3. **Datasets** — Accurately describe all three datasets as implemented (UCI Hourly Bike Sharing with leak-free gap imputation; Portuguese Electricity Load hourly demand; M5 Walmart daily national retail sales aggregate with SNAP/calendar drivers — explicitly noting national aggregate vs. 30,490-series hierarchical M5).
4. **Four-Tier Model Lineup & Scope Boundaries** — Detail Tiers 1–4 (`Naive`/`Seasonal Naive`/`RidgeCV`/`AutoETS`/`AutoTheta`/`MSTL`/`AutoARIMA`, `LightGBM`/`XGBoost`/`CatBoost`, `DLinear`/`PatchTST`/`NBEATSx`/`NHITS`, `Chronos-Bolt`), plus explicit rationale for excluding GluonTS `DeepAR`/`DeepState` and `TCN`.
5. **Evaluation Strategy, Contamination Disclosures & Roadmap** — Document `MASE` / `RMSSE` / `MAE` / `RMSE` / `sMAPE`, Monash pretraining overlap disclosure for `Chronos-Bolt`, and move unbuilt extensions (multi-horizon decay curves, Diebold–Mariano tests, item-level hierarchical M5, modular `src/` package refactor) into a concise *Roadmap & Planned Extensions* section rather than presenting them as already-shipped features.

---

## 6. Verification & Delivery Checklist

1. **Notebook JSON & AST Validation:** Parse every `.ipynb` with `json` and `ast.parse` (stripping IPython `!` / `%%time` magics) to guarantee zero syntax errors, undefined variable names, or broken cell ordering.
2. **Cross-Notebook Invariant Audit:** Verify programmatically that all 3 notebooks define all 17 benchmark models, evaluate on `y_test` (`len(test_df)`), clear stale Phase 4 outputs, and save `<dataset>_test_predictions.npz` and `<dataset>_benchmark_leaderboard.csv`.
3. **Google Drive Sync:** Upload the updated `UCI-Bike-Sharing.ipynb`, `Electricity-Load.ipynb`, and `M5-Forecasting.ipynb` to Google Drive folder `Time-Series-Forecasting-Honest-Comparison` (`1MLr_Hw_r0A3nLIsEY4mkJtWIsqvd4UaS`).

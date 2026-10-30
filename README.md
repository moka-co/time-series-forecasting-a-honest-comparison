# Time Series Forecasting: An Honest Benchmark

> [!NOTE]
> **Project Status:** Active empirical benchmark across three self-contained Polars + PyTorch/Nixtla notebooks (`UCI-Bike-Sharing.ipynb`, `Electricity-Load.ipynb`, `M5-Forecasting.ipynb`). Design specification: [`docs/superpowers/specs/2026-10-30-honest-benchmark-design.md`](docs/superpowers/specs/2026-10-30-honest-benchmark-design.md).

---

## The Big Picture: When Baselines Win and When They Don't

The central question behind this repository is practical: **when do complex neural architectures and pretrained foundation models genuinely beat classical statistical methods and gradient boosted trees, and at what compute cost?**

In practice:
- **Strong statistical and linear baselines** (`MSTL`, `AutoARIMA`, `AutoETS`, `AutoTheta`, `RidgeCV`) are tough to beat when demand follows stable repeating cycles and historical memory is clean. They train in seconds on CPU, require minimal operational upkeep, and provide transparent trend/seasonal components.
- **Gradient Boosted Decision Trees** (`LightGBM`, `XGBoost`, `CatBoost`) excel on single and tabularized series when engineered autoregressive lags, rolling statistics, calendar harmonics, and external drivers (weather, holidays, SNAP schedules) interact non-linearly.
- **Sequence deep learning and foundation models** (`DLinear`, `PatchTST`, `NBEATSx`, `NHITS`, `Chronos-Bolt`) earn their keep when multi-rate temporal patterns, long lookback horizons, or zero-shot cold-start requirements outweigh GPU training and inference overhead.

For engineering teams, the takeaway is simple: benchmark strong statistical and GBDT baselines under strict leak-free rules before deploying heavy neural pipelines.

---

## How This Benchmark Defines Fairness

Time series comparisons often fail in one of two ways: either classical baselines are run with naive defaults (`SARIMAX(1,1,1)` or free-running multi-step rollouts pitted against 1-step teacher-forced neural nets), or benchmarks claim *"every model gets the exact same feature matrix and tuning budget"* — an impossible standard across fundamentally different paradigms. Feeding a 15-column tabular lag/calendar vector into `DLinear` or `PatchTST` breaks sequence pooling, just as stripping lag features cripples `LightGBM`.

Instead, this benchmark enforces **four concrete fairness invariants** across every notebook:

1. **Identical Causal Information Cutoff ($\mathcal{F}_{t-1}$):**  
   When predicting target $y_t$ at any test step $t$, every model observes target history strictly up to $t-1$ plus causally valid calendar/exogenous covariates, with **zero parameter refitting** on the test split (`refit=False` 1-step-ahead rolling evaluation). Statistical state-space models update their Kalman/smoothing state with $y_{t-1}$ via `.forward()` rather than decaying across a blind multi-month forecast, matching the $y_{t-1}$ lag/context access of GBDTs, neural sequence models, and `Chronos-Bolt`.
2. **Paradigm-Native Input Representations:**  
   Every model family consumes $\mathcal{F}_{t-1}$ in the representation it was designed for:
   - *Statistical decomposition & state-space (`statsforecast`):* Raw continuous series $y$ with native seasonal cycle lengths (plus exogenous regressors $X$ for `AutoARIMA`).
   - *Regularized linear & GBDTs:* Leak-free autoregressive lags, shifted rolling statistics (`shift(1)` prior to windowing), cyclical trigonometric calendar encodings, and exogenous covariates.
   - *Sequence deep learning (`neuralforecast`):* Contiguous raw chronological lookback windows $y_{t-L:t-1}$ (`L=168` hourly / `L=56` daily), plus `futr_exog_list` covariates on exogenous-capable architectures (`NBEATSx`, `NHITS`).
   - *Zero-shot foundation (`Chronos-Bolt`):* Raw univariate context windows $y_{t-512:t-1}$.
3. **Strict Three-Way Chronological Split (`70% train / 15% val / 15% test`):**  
   `val_df` is reserved strictly for iterative early stopping (`LightGBM`, `XGBoost`, `CatBoost`, `NeuralForecast`) and `RidgeCV` regularization selection. **All leaderboard metrics are reported exclusively on the untouched final 15% `test_df`**, preventing early-stopping validation leakage while allowing every model to observe history right up to the `val_df` $\rightarrow$ `test_df` boundary.
4. **Disclosed Selection Protocols Under a 120s Per-Model Budget:**  
   Rather than asymmetric manual grid tuning, each tier uses its standard automated selection mechanism:
   - `AutoARIMA`: Hyndman–Khandakar stepwise AICc search over $(p, d, q)(P, D, Q)_s$ on pre-test history, logging the exact selected order.
   - `AutoETS` / `AutoTheta` / `MSTL`: Maximum-likelihood state-space selection and multi-seasonal LOESS decomposition on pre-test history.
   - `RidgeCV`: Closed-form generalized cross-validation over `alphas=[0.1, 1.0, 10.0, 100.0]`.
   - `GBDTs`: Standardized capacity (`n_estimators=1000`, `learning_rate=0.03`, `num_leaves=31` / `max_depth=6`) with `early_stopping_rounds=30` monitored on `val_df`.
   - `NeuralForecast`: Standardized optimizer/step budget (`max_steps=500`, `val_check_steps=25`, `early_stop_patience_steps=3` on `val_df`).
   - `Chronos-Bolt`: Out-of-the-box zero-shot inference (`amazon/chronos-bolt-tiny`, zero fine-tuning).

---

## Datasets

Each dataset is implemented as a standalone, reproducible notebook using `Polars` for leak-free temporal preprocessing:

1. **UCI Capital Bikeshare (`UCI-Bike-Sharing.ipynb`):**  
   `17,376` continuous hourly observations (2011–2012) of Washington, D.C. bike rentals (`cnt`). Missing hourly timestamps are reconstructed and imputed strictly from causal expanding weekday-hour historical means (zero future leakage), paired with meteorological drivers (`temp`, `atemp`, `hum`, `windspeed`, `weathersit`) and holiday/workday flags.
2. **Portuguese Electricity Load (`Electricity-Load.ipynb`):**  
   `26,136` continuous hourly observations (2012–2015) of aggregate active power demand (`load` in kW) from the UCI / Monash Electricity Load archive. Tests overlapping diurnal ($s_1=24$) and weekly ($s_2=168$) seasonalities.
3. **M5 Walmart National Retail Demand (`M5-Forecasting.ipynb`):**  
   `1,913` continuous daily observations (2011–2016) of aggregated U.S. Walmart unit sales (`national_sales`) enriched with state SNAP benefit schedules and weekly/monthly cycles ($s_1=7, s_2=28$). *(Note: This notebook benchmarks continuous aggregate retail volume and exogenous SNAP effects; item-level intermittent 30,490-series hierarchical WRMSSE is tracked under Planned Extensions.)*

---

## Four-Tier Model Lineup (17 Standardized Models)

Every notebook evaluates the exact same 17 models across four architectural tiers:

### Tier 1: Classical & Multi-Seasonal Statistical Models (`statsforecast` + `scikit-learn`)
- **Parameter-Free Baselines:** `Naive (t-1)`, `Smoothed Naive (2-Step)`, and dual `Seasonal Naive` (`t-24` & `t-168` hourly; `t-7` & `t-28` daily) setting the minimum accuracy floor.
- **Regularized Linear:** `RidgeCV` with `StandardScaler` and median imputation over engineered lag/calendar/exogenous features.
- **State-Space & Multi-Seasonal Statistical (`Nixtla StatsForecast`):**
  - `AutoETS` and `AutoTheta` (`season_length=s1`) for exponential smoothing and Theta decomposition.
  - `MSTL` (`season_length=[s1, s2]` with `AutoETS("ZZN")` trend forecaster) to handle overlapping diurnal/weekly and weekly/monthly cycles (addressing the single-seasonality limitation highlighted by Bergmeir et al.).
  - `AutoARIMA` (`season_length=s1` + dataset exogenous covariates) selected via stepwise AICc and rolled 1-step-ahead via state-space forward filtering (`refit=False`).

### Tier 2: Tabular Gradient Boosted Decision Trees
- `LightGBM Regressor` (GOSS/histogram leaf-wise trees), `XGBoost Regressor` (`tree_method="hist"` depth-wise trees), and `CatBoost Regressor` (symmetric oblivious trees), all regularized via validation early stopping on `val_df`.

### Tier 3: Sequence Deep Learning (`Nixtla NeuralForecast`)
- `DLinear` (*Zeng et al., AAAI 2023*): Moving-average trend/seasonal decomposition with independent linear projections on raw univariate lookback windows.
- `PatchTST` (*Nie et al., ICLR 2023*): Channel-independent subseries patching Transformer on raw univariate lookback windows.
- `NBEATSx` (*Olivares et al., IJF 2023*): Doubly-residual neural basis expansion incorporating exogenous calendar/weather/SNAP covariates (`futr_exog_list`).
- `NHITS` (*Challu et al., AAAI 2023*): Multi-rate hierarchical pooling and temporal interpolation with exogenous covariates (`futr_exog_list`).

### Tier 4: Zero-Shot Time-Series Foundation Model
- `Amazon Chronos-Bolt` (`amazon/chronos-bolt-tiny`, ~9M parameters, *Ansari et al., 2024*): Patch-tokenized zero-shot median quantile forecasts over rolling 512-step historical contexts.

### Architectural Scope Boundaries
To keep notebook runtimes practical (<3 minutes on a Colab T4 GPU) while covering the modern Pareto frontier, two architecture families are intentionally omitted from the core notebooks:
- **`DeepAR` / `DeepState` (Amazon Research / GluonTS):** Sometimes conflated with the Meta Prophet or PyMC ecosystems, `DeepAR` and `DeepState` are GluonTS probabilistic autoregressive RNN / state-space models built for global cross-series training across thousands of disaggregate items. On single-series point-forecasting tasks, they introduce heavy dependency overhead without outperforming `NHITS`, `PatchTST`, or GBDTs.
- **Temporal Convolutional Networks (`TCN`):** Dilated causal CNNs are an important historical sequence family, but in post-2023 forecasting benchmarks their multi-scale receptive field role is more effectively and efficiently represented by `NHITS` (multi-rate pooling) alongside `DLinear` and `PatchTST`.

---

## Metrics & Pretraining Contamination Disclosure

- **Scale-Dependent & Scale-Free Metrics:** Every notebook reports `MAE`, `RMSE`, `sMAPE (%)`, wall-clock `Fit Time (s)`, and scale-free error relative to pre-test naive persistence (`MASE` for Bike Sharing and Electricity Load; `RMSSE` for M5 Retail Sales).
- **Foundation Model Contamination Disclosure:** The official `Chronos` pretraining corpus includes the Monash Time Series Forecasting Archive (which contains Portuguese Electricity Load, Capital Bikeshare subsets, and M5 subsets). Accordingly, `Chronos-Bolt` results are explicitly labeled in every notebook as **in-distribution zero-shot recall / adaptation** rather than strictly blind out-of-distribution transfer.

---

## Roadmap & Planned Extensions

- **Codebase Modularization:** Extract shared Tier 1–4 training, evaluation, and plotting routines from the three standalone Colab notebooks into a reusable `src/` Python package.
- **Multi-Step Horizon Decay ($h > 1$):** Add free-running multi-horizon evaluation ($h=24$ hours / $h=28$ days without intermediate $y_{t-1}$ updates) to map where model error converges toward `Seasonal Naive`.
- **Disaggregate Cross-Series M5:** Extend M5 from national aggregate demand to global cross-series training (`LightGBM`, `NHITS`, `DeepAR`) across item/store hierarchies evaluated with official `WRMSSE` and quantile pinball loss.
- **Statistical Significance & Ablations:** Add Diebold–Mariano tests, lookback sensitivity curves, and input-shuffling ablations.

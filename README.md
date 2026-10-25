# Time Series Forecasting: An Honest Benchmark

> [!NOTE]
> **Project Status:** Work in progress. Benchmarks, backtesting runs, and the live demo are actively being developed and updated.

---

## The Big Picture: When Baselines Win and When They Don't

The main question behind this project is simple: do we actually need complex deep learning models for time series, or do basic baselines do the job just as well for a fraction of the cost? 

In practice, simple baselines win when data has clean, predictable seasonality and a single repeating cycle. If you are forecasting a regular hourly or daily pattern with low noise over short horizons, Seasonal Naive or a simple ensemble of ETS, Theta, and ARIMA will often match or beat a neural network. They are fast, reliable, run in milliseconds on CPU, and cannot overfit.

Complex models pull ahead when two things happen: cross-series learning and rich external drivers. When you have thousands of related items like in retail demand (such as the M5 dataset), models like LightGBM or deep architectures can learn shared patterns across the entire catalog that no single-series model can catch. They also shine when predictions rely heavily on external variables like weather, prices, or marketing promotions, and when data has multiple overlapping seasons (like hourly electricity demand with both daily and weekly rhythms).

For engineering and data science teams, this benchmark provides a practical rule of thumb: always start with simple baselines. If a complex neural model is only two percent better but takes fifty times longer to train and requires a dedicated GPU in production, it is rarely worth the maintenance headache.

---

## Why This Benchmark Matters

Time series research is full of exciting new deep learning architectures and foundation models. Yet practical industry papers—including the LTSF-Linear critique—regularly show that simple linear baselines often beat multi-layer Transformers when tested properly.

A major reason for this gap is benchmark bias. In many academic comparisons, complex neural models receive extensive hyperparameter tuning, GPU compute, and carefully engineered features, while classical baselines are run with out-of-the-box defaults. This project levels the playing field: every model gets the exact same feature inputs, the same tuning budget, and the same evaluation setup.

---

## Datasets

This benchmark uses three datasets that reflect very different real-world challenges.

The M5 Walmart Competition dataset covers daily retail sales across thousands of products. It is dominated by intermittent demand, promotions, price changes, and lots of zeros. It tests how well models handle sparse data, hierarchical structures, and cross-series learning.

The Electricity Load dataset tracks continuous hourly power consumption over several years. It features strong multiple seasonalities, including daily routines, workweek versus weekend differences, and seasonal temperature swings.

The Bike Sharing dataset tracks hourly bike rentals alongside continuous weather measurements like temperature, humidity, and wind speed. It tests how effectively models use external drivers to improve forecast accuracy.

---

## Model Lineup Across Three Tiers

To make the comparison fair, models are grouped into three distinct tiers rather than just pitting naive methods directly against massive neural nets.

### Simple Baselines
This tier sets the minimum performance standard that any useful model must beat. It includes Seasonal Naive and an equally weighted ensemble of classic statistical methods: ETS, Theta, and AutoARIMA. These run quickly on any laptop and give a fast sanity check on basic trends and seasonal cycles.

### Strong Statistical Models
Comparing basic ARIMA or ETS to neural networks on hourly data is fundamentally unfair because standard ARIMA and ETS only handle a single seasonal period (a point emphasized by Christoph Bergmeir). To fix this, this tier includes models built specifically for multiple seasonalities: MSTL (Multiple Seasonal-Trend decomposition using LOESS), TBATS, and Dynamic Harmonic Regression using ARIMA with Fourier terms.

### Machine Learning and Deep Learning
This tier includes gradient boosting and modern neural architectures. LightGBM is trained with calendar encodings and rolling lag features. Prophet is included for decomposable trend-season modeling. For neural approaches, the benchmark tests N-BEATS and NHITS for hierarchical basis expansion, DLinear and NLinear to test simple linear projections against attention mechanisms, and a lightweight Temporal Transformer. Pre-trained zero-shot foundation models such as Chronos and TimesFM are also evaluated.

---

## Evaluation Strategy and Practical Metrics

Evaluating time series requires strict rules to avoid leaking future data into the past.

Every model is tested using expanding-window backtesting across rolling forecast origins. Standard random k-fold cross-validation is strictly avoided because shuffling time series records leaks future information into the training set, giving artificially optimistic results that fail in production.

Metrics are chosen based on the distribution of each dataset. For M5, performance is measured using Root Mean Squared Scaled Error (RMSSE) and Weighted RMSSE (WRMSSE), the competition's official metrics. Standard metrics like sMAPE are avoided on M5 because dividing by near-zero values distorts the score when demand is intermittent. For electricity load and bike sharing, models are evaluated using MASE and sMAPE to balance scale invariance with easy business interpretation.

To confirm that small differences between models are real rather than random luck, model rankings are backed by Diebold-Mariano tests and Friedman-Nemenyi critical difference diagrams. Every run also tracks practical production costs: training time, inference latency, memory footprint, and required hardware.

---

## Diagnostic Experiments

Beyond standard leaderboards, this project runs targeted experiments to uncover how models actually behave under the hood.

### Error by Forecast Horizon
Forecast accuracy naturally drops as the prediction window gets longer. By tracking error from one hour out to several days, these experiments show where each model's curve converges to Seasonal Naive. That convergence point marks the empirical limit of forecastability, showing exactly where extra model complexity stops helping.

### Lookback Window Sensitivity
Instead of picking an arbitrary historical window like 96 steps, the benchmark tests models across varying lookback lengths. Plotting error against lookback depth tests the LTSF-Linear finding: checking whether linear models continue to improve with longer histories while Transformers plateau or overfit.

### Input Shuffling Test
To test whether deep sequence models actually use chronological order, an ablation test feeds randomly shuffled input windows to trained models. If a complex neural model performs just as well on shuffled data as it does on ordered data, it is acting like a basic regression model rather than truly understanding temporal structure.

### Data Leakage Demonstration
To show why proper backtesting matters, this experiment benchmarks the same models under both random cross-validation and rolling-origin backtesting. The difference in error scores gives a clear demonstration of how temporal leakage creates misleading benchmark claims.

---

## Foundation Models and Extra Analyses

To keep the study relevant for modern engineering workflows, the benchmark explores several practical extensions.

Zero-shot foundation models like Chronos and TimesFM are tested straight out of the box, with careful checks for pretraining contamination on public datasets like electricity load. Top-performing models are evaluated on prediction intervals and quantile loss to gauge uncertainty rather than just point forecasts. Systematic failure cases around holidays and anomalous events are broken down, alongside feature importance for LightGBM and component breakdowns for ETS. Finally, compute-versus-accuracy plots map out the actual trade-off between cloud costs and forecast quality.

---

## Codebase and Deliverables

The repository is built around reproducibility, clean structure, and practical engineering tools.

Code is organized using `darts` to maintain a consistent interface across classical and neural models, alongside `neuralforecast` for fast, GPU-accelerated training of DLinear, N-BEATS, and NHITS. All runs are controlled by YAML configuration files with fixed random seeds for easy reproduction. The deliverables include the core Python benchmarking pipeline, an interactive results notebook, and a lightweight Streamlit and FastAPI demo for running live forecasts and inspecting forecast plots and latency numbers.

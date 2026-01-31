# particle-filter-rs

**Rust + PyO3 particle filter core** — a compiled, zero-copy, GIL-free sequential Monte Carlo library for regime-switching state estimation.

All 19 numerical core functions ported to safe Rust. Parallel prediction via rayon. Bit-exact parity with the Python originals (verified to `1e-10` tolerance). One-line import swap to activate.

---

## A Brief History of the Particle Filter

The particle filter has a lineage stretching back six decades — from Cold War ballistics to modern autonomous vehicles. Understanding where it came from illuminates why the algorithm looks the way it does.

**1960 — The Kalman filter**
Rudolf Kalman publishes *"A New Approach to Linear Filtering and Prediction Problems"*. For linear systems with Gaussian noise, his filter is provably optimal — it yields exact state estimates with minimal computation. It becomes the backbone of Apollo navigation, GPS, and every autopilot on earth. But it has a hard constraint: **linearity**. Real systems are rarely linear.

**1969 — First Monte Carlo attempt**
Handschin and Mayne propose using random samples (Monte Carlo simulation) to approximate the state distribution for nonlinear systems. The idea is sound, but 1960s hardware cannot sustain it — a few hundred samples per update is all a mainframe can manage, and without resampling the estimates degenerate catastrophically as particles drift away from the true state.

**1993 — The bootstrap particle filter**
Neil Gordon, David Salmond, and Adrian Smith at the UK Defence Evaluation and Research Agency (DERA) publish *"Novel approach to nonlinear/non-Gaussian Bayesian state estimation"*. The breakthrough: **resampling**. After each observation, clone the particles that explain the data well and discard the ones that don't. This simple operation prevents degeneracy and makes the algorithm practical. The original application: **tracking military aircraft and missiles** from noisy radar returns — targets that maneuver unpredictably, exactly the kind of nonlinear, multi-modal problem the Kalman filter cannot handle.

> The `systematic_resample` function in this library implements the same algorithmic idea from 1993 — the two-pointer O(N) variant introduced by Carpenter, Clifford, and Fearnhead in 1999.

**1996 — The name "particle filter"**
Pierre Del Moral coins the term *particle filter* and develops the rigorous mathematical framework (Feynman-Kac formulations) that proves convergence guarantees. Independently, Genshiro Kitagawa develops a closely related "Monte Carlo filter" in the statistics community. The two research streams converge.

**1998 — Computer vision discovers particles**
Michael Isard and Andrew Blake publish the CONDENSATION algorithm (Conditional Density Propagation), bringing particle filters into visual object tracking. Suddenly a webcam can follow a hand, a face, or a bouncing ball through clutter and occlusion — problems where Kalman filters fail because the target can be in multiple plausible locations simultaneously.

**2000s — Explosion of applications**
Particle filters spread into robotics (SLAM — simultaneous localization and mapping), finance (stochastic volatility models), speech recognition, bioinformatics, weather prediction, and ocean current estimation. The regime-switching variant — where particles carry both continuous state and a discrete mode label — emerges as the standard approach for systems that switch between qualitatively different behaviors.

**2010s–present — Hardware catches up**
GPUs and multi-core CPUs make it feasible to run tens of thousands of particles in real time. The algorithm's embarrassingly parallel structure (each particle is independent during prediction) maps naturally to SIMD, thread pools, and GPU warps. Libraries like this one exploit that structure via rayon's work-stealing scheduler.

### The core insight, then and now

A particle filter maintains a **population of hypotheses** about the hidden state of a system. Each hypothesis (particle) is a concrete guess — a point in state space — weighted by how well it explains the observed data. The population evolves through three operations that have remained essentially unchanged since 1993:

1. **Predict** — propagate each particle forward through the dynamics model
2. **Update** — re-weight each particle by the likelihood of the new observation
3. **Resample** — clone the fit, cull the unfit

This library implements exactly that loop, plus regime transitions (Markov switching between dynamics models), a companion Kalman smoother, and auxiliary signal processors — all in compiled Rust with zero-copy NumPy interop.

---

## Why Rust + Python

Python excels at prototyping, data wrangling, and orchestration. Rust excels at the thing Python cannot do: sustained, predictable, low-latency number crunching without a garbage collector. Combining them via PyO3 gives you both — and specifically for this library:

| Dimension | Python alone (Numba JIT) | Rust via PyO3 |
|---|---|---|
| **First-call latency** | 2-5 s JIT warm-up per function | Zero — compiled ahead of time |
| **GIL** | Held during Numba execution | Released (`py.detach`) — other Python threads run freely |
| **Parallelism** | `prange` limited to simple loops | Full work-stealing (rayon) with safe shared-nothing concurrency |
| **Memory safety** | Runtime bounds checks | Compile-time guarantees — no segfaults, no buffer overruns |
| **Dependency weight** | `numba` + `llvmlite` (~150 MB) | Single `.so` file (~2 MB), no LLVM runtime |
| **Reproducibility** | JIT output can vary across LLVM versions | Deterministic binary — same result on every machine |
| **Distribution** | Requires Numba installed everywhere | `pip install *.whl` — self-contained, no compiler needed |

The result: you keep writing your application logic in Python, but the inner hot loop — the one that runs thousands of times per second — executes at native speed with zero interpreter overhead.

---

## Core Concepts — A Practical Glossary

Quick reference for the key ideas behind every function in this library.

| Term | What it means | Where it appears |
|---|---|---|
| **Particle** | A single hypothesis about the system state — here a pair `[log_price, velocity]`. The filter maintains N particles (typically 500–2000) as a swarm of guesses. | `predict_particles`, `systematic_resample`, `estimate` |
| **Weight** | A probability score `wᵢ ∈ [0,1]` attached to each particle. High weight = this hypothesis explains the observed data well. All weights sum to 1. | `update_weights`, `effective_sample_size`, `estimate` |
| **Regime** | A discrete market micro-state. Three regimes model qualitatively different dynamics: **RANGE** (mean-reverting), **TREND** (directional), **PANIC** (high-volatility random walk). Each particle carries its own regime label. | `transition_regimes`, `predict_particles` |
| **Transition matrix** | A 3×3 row-stochastic matrix governing how regimes switch over time. Entry `T[i,j]` = probability of jumping from regime `i` to regime `j`. | `transition_regimes` |
| **Resampling** | When most weight concentrates on few particles (degeneracy), clone the winners and discard the losers. Systematic resampling does this in O(N) with low variance. | `systematic_resample` |
| **ESS** | Effective Sample Size = `1 / Σwᵢ²`. Measures particle diversity. ESS ≈ N means all particles are equally useful; ESS ≈ 1 means one particle dominates. Resample when ESS < N/2. | `effective_sample_size` |
| **Log-price space** | All prices are stored as `ln(price)`. This makes additive noise scale-invariant — the same filter parameters work for a $0.50 asset and a $50,000 asset. | All functions |
| **Kalman filter** | A closed-form optimal estimator for linear-Gaussian systems. Here used as a 2D `[level, slope]` smoother that complements the particle filter's nonlinear estimation. | `kalman_update` |
| **VWAP** | Volume-Weighted Average Price — the "fair value" consensus weighted by traded volume. The σ-bands mark statistical extremes around it. | `calculate_vwap_bands` |
| **Momentum score** | A bounded `[-1, +1]` measure of recent directional tendency. Uses `tanh` compression so the signal saturates gracefully instead of exploding. | `calculate_momentum_score` |
| **Hurst exponent** | A measure of long-term memory via R/S analysis. H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk. | `hurst_exponent` |
| **CUSUM** | Page's Cumulative Sum test — detects structural breaks (mean-shift reversals) in a time series by accumulating deviations from a reference mean. | `cusum_test` |
| **Kurtosis** | Excess kurtosis of recent values. Positive = fat tails (widen bands), negative = thin tails (tighten bands), zero = normal. | `rolling_kurtosis` |
| **Volatility compression** | Ratio of short-term to long-term volatility. Ratio < 0.5 = compressed (pause entries), ratio > 1.5 = expanding (breakout). | `volatility_compression` |

---

## Application Domains

This library is a general-purpose **regime-switching particle filter**. Anywhere you have noisy sequential observations and suspect the underlying system switches between qualitatively different behaviors, this toolkit applies:

**Financial signal processing**
- Real-time price level / trend / volatility-regime estimation
- Order-book imbalance → velocity alignment scoring
- Adaptive VWAP bands that respond to regime shifts

**Robotics & navigation**
- Multi-modal position tracking (GPS + IMU + wheel odometry)
- Terrain-adaptive motion models (road vs off-road vs water)
- Sensor fusion with regime-dependent noise profiles

**IoT & anomaly detection**
- Industrial sensor streams with normal / degraded / failure modes
- Network traffic classification (idle / burst / attack)
- Energy grid load-state estimation with weather-regime switching

**Climate & geophysics**
- Regime-switching weather pattern tracking
- Seismic signal filtering with quiet / active / aftershock modes
- Ocean current state estimation under seasonal regime changes

**Biomedical**
- EEG/ECG state estimation with wake / sleep / seizure regimes
- Drug concentration tracking with absorption / distribution / elimination phases

The 3-regime model (RANGE / TREND / PANIC) maps naturally to any domain where the system alternates between **stable**, **directional**, and **chaotic** behavior.

---

## Architecture

```
                        Python (application layer)
                 ┌──────────────────────────────────────┐
                 │         Your Python code              │
                 │      import particle_filter_rs        │
                 └──────────────────┬───────────────────┘
                                   │
                                   ▼
              ┌─────────────────────────────────────┐
              │      particle_filter_rs (Rust)      │
              │                                     │
              │  ┌───────────┐   ┌───────────────┐  │
              │  │ predict   │──▶│ GIL released  │  │
              │  │ (rayon)   │   │ parallel loop │  │
              │  └───────────┘   └───────────────┘  │
              │  ┌───────────┐   ┌───────────────┐  │
              │  │ update    │   │ resample      │  │
              │  │ weights   │   │ + ESS         │  │
              │  └───────────┘   └───────────────┘  │
              │  ┌───────────┐   ┌───────────────┐  │
              │  │ regime    │   │ estimate      │  │
              │  │ transition│   │ weighted mean │  │
              │  └───────────┘   └───────────────┘  │
              │  ┌───────────┐   ┌───────────────┐  │
              │  │ kalman    │   │ vwap + bands  │  │
              │  │ 2D update │   │ momentum score│  │
              │  └───────────┘   └───────────────┘  │
              │  ┌───────────┐   ┌───────────────┐  │
              │  │ hurst     │   │ cusum         │  │
              │  │ R/S expon │   │ break detect  │  │
              │  └───────────┘   └───────────────┘  │
              │  ┌───────────┐   ┌───────────────┐  │
              │  │ volatility│   │ kurtosis +    │  │
              │  │ compress  │   │ adaptive σ    │  │
              │  └───────────┘   └───────────────┘  │
              └─────────────────────────────────────┘
                        │ PyO3 + numpy FFI │
                        ▼                  ▼
              ┌─────────────────────────────────────┐
              │         NumPy ndarrays (f64)         │
              │     zero-copy read via as_array()    │
              └─────────────────────────────────────┘
```

## How It Works — The Particle Filter Pipeline

Each tick of market data triggers this sequence:

```
 Market Tick
      │
      ▼
 ┌─────────────────────┐
 │ 1. predict_particles │  Propagate N particles forward in time
 │    (parallel/rayon)  │  per-regime dynamics in LOG-PRICE space
 └──────────┬──────────┘
            ▼
 ┌─────────────────────┐
 │ 2. update_weights    │  Score each particle against observation
 │    (likelihood)      │  Gaussian likelihood × velocity alignment
 └──────────┬──────────┘
            ▼
 ┌─────────────────────┐
 │ 3. effective_sample  │  ESS = 1/Σ(wᵢ²)
 │    _size             │  If ESS < threshold → resample
 └──────────┬──────────┘
            ▼
 ┌─────────────────────┐
 │ 4. systematic_       │  O(N) two-pointer resampling
 │    resample          │  Clones high-weight, kills low-weight
 └──────────┬──────────┘
            ▼
 ┌─────────────────────┐
 │ 5. transition_       │  Markov chain: RANGE ⇄ TREND ⇄ PANIC
 │    regimes           │  3×3 row-stochastic transition matrix
 └──────────┬──────────┘
            ▼
 ┌─────────────────────┐
 │ 6. estimate          │  Weighted mean log-price, velocity,
 │                      │  and regime probability vector [3]
 └──────────┬──────────┘
            ▼
      Trade Signal
```

Auxiliary functions provide supplementary signal-processing capabilities that run alongside the core filter loop: `kalman_update` (level/slope smoothing), `calculate_vwap_bands`, `calculate_momentum_score`, `hurst_exponent` (regime validation), `cusum_test` (reversal detection), `volatility_compression` (range squeeze), `rolling_kurtosis` + `adaptive_vwap_sigma` (fat-tail adaptation), `slope_confidence_interval` / `is_slope_significant` / `kalman_slope_acceleration` (trend significance), `ess_and_uncertainty_margin` (dominance test), and `particle_price_variance` (spread estimation).

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/RMANOV/particle-filter-rs.git
cd particle-filter-rs

# 2. Create virtualenv + install deps
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy

# 3. Build (release mode, optimized)
maturin develop --release

# 4. Verify
python tests/test_parity.py
```

Expected output:

```
Particle Filter Rust vs Python Parity Tests
==================================================
  predict_particles: PASS
  update_weights: PASS
  ...
--------------------------------------------------
New functions:
  particle_price_variance: PASS
  ess_and_uncertainty_margin: PASS
  ...
  hurst_exponent: PASS
==================================================
ALL 19 FUNCTIONS PASS PARITY TESTS
```

---

## Integration — One-Line Swap

```python
from particle_filter_rs import (
    # Core particle filter
    predict_particles,
    update_weights,
    transition_regimes,
    systematic_resample,
    effective_sample_size,
    estimate,
    particle_price_variance,
    ess_and_uncertainty_margin,
    # Kalman filter
    kalman_update,
    slope_confidence_interval,
    is_slope_significant,
    kalman_slope_acceleration,
    # Signal processing
    calculate_vwap_bands,
    rolling_kurtosis,
    adaptive_vwap_sigma,
    calculate_momentum_score,
    # Regime analysis
    cusum_test,
    volatility_compression,
    hurst_exponent,
)
```

---

## API Reference

### Particle Filter Core

#### `predict_particles`

Propagate particles forward with regime-specific dynamics. **GIL released** — runs on all CPU cores via rayon.

```python
predict_particles(
    particles: np.ndarray,          # (N, 2) float64 — [log_price, velocity]
    regimes: np.ndarray,            # (N,)   int64   — 0=RANGE, 1=TREND, 2=PANIC
    process_noise_pos: np.ndarray,  # (3,)   float64 — per-regime position noise σ
    process_noise_vel: np.ndarray,  # (3,)   float64 — per-regime velocity noise σ
    imbalance: float,               # order-book imbalance signal
    dt: float,                      # time step (seconds)
    vel_gain: float,                # velocity sensitivity to imbalance
    random_pos: np.ndarray,         # (N,)   float64 — pre-generated N(0,1)
    random_vel: np.ndarray,         # (N,)   float64 — pre-generated N(0,1)
) -> np.ndarray                     # (N, 2) float64 — updated particles
```

**Regime dynamics:**

| Regime | ID | Model | Equation |
|---|---|---|---|
| RANGE | 0 | Mean-reverting | `v' = 0.5·v + ε_v` , `x' = x + v'·dt + ε_x` |
| TREND | 1 | Imbalance-tracking | `v' = v + 0.3·(G·imb - v)·dt + ε_v` , `x' = x + v'·dt + ε_x` |
| PANIC | 2 | Random walk | `v' = v + ε_v` , `x' = x + v'·dt + ε_x` |

Where `ε ~ N(0, σ_regime · √dt)` and `G = vel_gain`.

---

#### `update_weights`

Bayesian weight update via Gaussian likelihood on two observations.

```python
update_weights(
    particles: np.ndarray,          # (N, 2) float64
    regimes: np.ndarray,            # (N,)   int64
    weights: np.ndarray,            # (N,)   float64 — prior weights
    measured_log_price: float,      # observed log-price
    meas_noise_price: np.ndarray,   # (3,)   float64 — per-regime price noise σ
    meas_noise_vel: np.ndarray,     # (3,)   float64 — per-regime velocity noise σ
    imbalance: float,
    vel_gain: float,
) -> np.ndarray                     # (N,) float64 — normalized posterior weights
```

**Likelihood:**
```
L_price = exp(-0.5 · (z - xᵢ)² / σ²_price)
L_vel   = exp(-0.5 · (vᵢ - G·imb)² / σ²_vel)
wᵢ'     = wᵢ · L_price · L_vel
```
Weights normalized with `+1e-300` underflow guard.

---

#### `transition_regimes`

Markov regime switching via cumulative probability search.

```python
transition_regimes(
    regimes: np.ndarray,            # (N,)   int64
    transition_matrix: np.ndarray,  # (3, 3) float64 — row-stochastic
    random_uniform: np.ndarray,     # (N,)   float64 — U(0,1)
) -> np.ndarray                     # (N,)   int64 — new regime assignments
```

**Transition matrix example:**
```
         RANGE  TREND  PANIC
RANGE  [  0.80   0.15   0.05 ]
TREND  [  0.10   0.80   0.10 ]
PANIC  [  0.20   0.30   0.50 ]
```

---

#### `systematic_resample`

O(N) systematic resampling — single-pass, two-pointer.

```python
systematic_resample(
    weights: np.ndarray,    # (N,)   float64
    particles: np.ndarray,  # (N, 2) float64
    regimes: np.ndarray,    # (N,)   int64
    start_offset: float,    # U(0, 1/N) — pre-generated
) -> tuple[
    np.ndarray,  # (N, 2) float64 — resampled particles
    np.ndarray,  # (N,)   int64   — resampled regimes
    np.ndarray,  # (N,)   float64 — uniform weights (1/N)
]
```

---

#### `effective_sample_size`

Particle degeneracy diagnostic.

```python
effective_sample_size(
    weights: np.ndarray,  # (N,) float64
) -> float                # ESS ∈ [1, N]
```

**Formula:** `ESS = 1 / (Σ wᵢ² + 1e-12)`

Rule of thumb: resample when `ESS < N/2`.

---

#### `estimate`

Collapse particle cloud to point estimates + regime beliefs.

```python
estimate(
    particles: np.ndarray,  # (N, 2) float64
    weights: np.ndarray,    # (N,)   float64
    regimes: np.ndarray,    # (N,)   int64
) -> tuple[
    float,       # weighted mean log-price
    float,       # weighted mean velocity
    np.ndarray,  # (3,) float64 — P(RANGE), P(TREND), P(PANIC)
]
```

---

### Supplementary Signals

#### `kalman_update`

2D Kalman filter for smooth level/slope tracking.

```python
kalman_update(
    level: float,                   # current level estimate
    slope: float,                   # current slope estimate
    p: np.ndarray,                  # (2, 2) float64 — covariance matrix P
    measured_log_price: float,
    dt: float,
    q_level: float,                 # process noise for level
    q_slope: float,                 # process noise for slope
    r: float,                       # measurement noise variance
) -> tuple[
    float,       # new level
    float,       # new slope
    np.ndarray,  # (2, 2) float64 — updated covariance P
]
```

**State model:**
```
Predict:  level' = level + slope · dt
          P'     = F·P·Fᵀ + Q       (F = [[1, dt],[0, 1]])
Update:   K      = P'·Hᵀ / (H·P'·Hᵀ + R)
          state  = state + K · (z - H·state)
```
Covariance symmetrized at each step for numerical stability.

---

#### `calculate_vwap_bands`

Volume-weighted average price with standard-deviation bands.

```python
calculate_vwap_bands(
    prices: np.ndarray,   # (M,) float64 — log-price history
    volumes: np.ndarray,  # (M,) float64 — volume history
    window: int,          # lookback window
    band_sigma: float = 1.5,  # band width multiplier (default 1.5σ)
) -> tuple[
    float,  # VWAP
    float,  # upper band (VWAP + σ·band_sigma)
    float,  # lower band (VWAP - σ·band_sigma)
]
# Returns (NaN, NaN, NaN) if len(prices) < window or zero volume
```

**Formula:**
```
VWAP = Σ(pᵢ·vᵢ) / Σ(vᵢ)
σ    = √[ Σ(vᵢ·(pᵢ - VWAP)²) / Σ(vᵢ) ]
```

---

#### `calculate_momentum_score`

Normalized momentum indicator bounded to [-1, +1].

```python
calculate_momentum_score(
    prices: np.ndarray,  # (M,) float64 — log-price history
    window: int,         # lookback window
) -> float               # momentum ∈ [-1, 1]
```

**Algorithm:** Split the last `window` prices at midpoint. Compare `mean(recent_half) - mean(older_half)`. Apply `tanh(Δ × 200)` for bounded output with high sensitivity around zero.

Returns `0.0` if `len(prices) < window` or `len(prices) < 3`.

---

### Extended Core

#### `particle_price_variance`

Weighted variance of particle log-prices — measures spread of the particle cloud.

```python
particle_price_variance(
    particles_pos: np.ndarray,  # (N,) float64 — log-prices
    weights: np.ndarray,        # (N,) float64 — particle weights
    mean_price: float,          # weighted mean log-price
) -> float                      # weighted variance
```

**Formula:** `Var = Σ wᵢ · (xᵢ - μ)²`

---

#### `ess_and_uncertainty_margin`

Combined ESS ratio, statistical uncertainty margin, and trend dominance test.

```python
ess_and_uncertainty_margin(
    weights: np.ndarray,  # (N,) float64
    p_trend: float,       # P(TREND) regime probability
    p_range: float,       # P(RANGE) regime probability
    p_panic: float,       # P(PANIC) regime probability
) -> tuple[
    float,  # ess_ratio ∈ [0, 1]  (ESS / N)
    float,  # uncertainty_margin   (2 × SE for 95% CI)
    bool,   # is_dominant          (trend significantly exceeds others)
]
```

**Dominance test:** `p_trend > p_range + margin AND p_trend > p_panic + margin AND p_trend > 0.45`

---

### Kalman Slope Analysis

#### `slope_confidence_interval`

95% confidence interval for the Kalman slope estimate.

```python
slope_confidence_interval(
    kf_slope: float,   # slope estimate
    kf_p_11: float,    # P[1,1] slope variance
) -> tuple[
    float,  # slope (unchanged)
    float,  # sigma (√variance, clamped ≥ √1e-12)
    float,  # ci_95 = 1.96 × sigma
]
```

---

#### `is_slope_significant`

Test if Kalman slope is statistically significant at the 95% confidence level.

```python
is_slope_significant(
    kf_slope: float,   # slope estimate
    kf_p_11: float,    # P[1,1] slope variance
    direction: int,    # 0=bidirectional, 1=positive only, -1=negative only
) -> bool              # True if |slope| > 1.96σ (in specified direction)
```

---

#### `kalman_slope_acceleration`

Detect acceleration (second derivative) of the Kalman slope for early trend entry.

```python
kalman_slope_acceleration(
    slopes_history: np.ndarray,  # (M,) float64 — historical slope values
    lookback: int = 10,          # window of recent slopes to analyze
) -> tuple[
    float,  # acceleration (recent_avg_change - older_avg_change)
    bool,   # is_accelerating (|acceleration| > 2σ AND > 1e-8)
]
```

Returns `(0.0, False)` if `len(slopes_history) < lookback` or `lookback < 4`.

---

### Statistical Tests

#### `cusum_test`

Page's CUSUM test for structural break (mean-shift) detection.

```python
cusum_test(
    values: np.ndarray,           # (M,) float64 — slopes, returns, etc.
    threshold_h: float = 0.5,     # slack parameter (in σ units)
    min_samples: int = 10,        # minimum samples before testing
) -> tuple[
    bool,   # is_reversal
    int,    # direction: 1=positive, -1=negative, 0=none
    float,  # cusum_value (max absolute CUSUM at alarm or end)
]
```

**Algorithm:** Reference mean from first half. CUSUM+ detects upward shift, CUSUM- detects downward shift. Alarm when |S| > 5σ.

---

#### `rolling_kurtosis`

Excess kurtosis of recent values for fat-tail detection.

```python
rolling_kurtosis(
    values: np.ndarray,     # (M,) float64
    window: int = 50,       # rolling window size
) -> float                  # excess kurtosis, clamped to [-2, 10]
```

**Formula:** `Kurt = E[(x-μ)⁴] / σ⁴ - 3` — zero for normal, positive for fat tails, negative for thin tails.

Returns `0.0` if insufficient data.

---

#### `adaptive_vwap_sigma`

Compute adaptive VWAP band width based on kurtosis.

```python
adaptive_vwap_sigma(
    kurtosis: float,            # from rolling_kurtosis()
    base_sigma: float = 1.5,    # default sigma (normal distribution)
    min_sigma: float = 1.2,     # floor
    max_sigma: float = 2.5,     # ceiling
) -> float                      # adaptive sigma, clamped to [min, max]
```

**Formula:** `σ_adaptive = base + 0.1 × kurtosis` — fat tails widen bands, thin tails tighten them.

---

#### `volatility_compression`

Detect volatility compression (range squeeze) by comparing short-term vs long-term volatility.

```python
volatility_compression(
    prices: np.ndarray,          # (M,) float64 — log-prices
    short_window: int = 10,      # recent volatility window
    long_window: int = 50,       # historical volatility window
) -> tuple[
    float,  # compression_ratio (σ_short / σ_long)
    bool,   # is_compressed (ratio < 0.5)
    bool,   # is_expanding  (ratio > 1.5)
]
```

Returns `(1.0, False, False)` if insufficient data.

---

#### `hurst_exponent`

Hurst exponent via Rescaled Range (R/S) analysis — measures long-term memory in a time series.

```python
hurst_exponent(
    prices: np.ndarray,          # (M,) float64 — log-prices
    min_window: int = 10,        # minimum R/S window
    max_window: int = 50,        # maximum R/S window
) -> tuple[
    float,  # H ∈ [0, 1] — Hurst exponent
    float,  # uncertainty (regression residual std error)
]
```

**Interpretation:**
| H | Regime | Meaning |
|---|---|---|
| H > 0.5 | Trending | Positive autocorrelation — momentum persists |
| H = 0.5 | Random walk | No memory — pure noise |
| H < 0.5 | Mean-reverting | Negative autocorrelation — moves tend to reverse |

Returns `(0.5, 0.5)` if `len(prices) < max_window` or fewer than 3 valid windows.

---

## Project Structure

```
particle-filter-rs/
├── Cargo.toml              # Rust dependencies
├── pyproject.toml           # maturin build configuration
├── README.md
├── src/
│   ├── lib.rs              # PyO3 module — registers all 19 functions
│   ├── predict.rs          # predict_particles     (parallel, rayon)
│   ├── update.rs           # update_weights        (Gaussian likelihood)
│   ├── regime.rs           # transition_regimes    (Markov chain)
│   ├── resample.rs         # systematic_resample + effective_sample_size + ess_and_uncertainty_margin
│   ├── estimate.rs         # estimate + particle_price_variance
│   ├── kalman.rs           # kalman_update + slope_confidence_interval + is_slope_significant + kalman_slope_acceleration
│   ├── vwap.rs             # calculate_vwap_bands + rolling_kurtosis + adaptive_vwap_sigma
│   ├── momentum.rs         # calculate_momentum_score (tanh-normalized)
│   ├── cusum.rs            # cusum_test            (structural break detection)
│   ├── volatility.rs       # volatility_compression (range squeeze)
│   └── hurst.rs            # hurst_exponent        (R/S analysis)
└── tests/
    └── test_parity.py      # Rust vs Python — 34 test assertions, atol=1e-10
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Return new arrays** instead of in-place mutation | Avoids mutable borrow conflicts across PyO3 boundary; callers already reassign (`self.particles = predict_particles(...)`) |
| **Copy to owned `Vec`** before `py.detach()` | Required for `Ungil` safety — NumPy array views carry `'py` lifetimes that cannot cross GIL-release boundaries |
| **rayon only for `predict_particles`** | The predict step is the only O(N) function with sufficient per-element work to amortize thread-pool overhead; other functions are memory-bound |
| **`+1e-300` underflow guard** in weight normalization | Matches Numba original exactly; prevents division-by-zero after many low-likelihood updates |
| **`+1e-12` in ESS denominator** | Prevents infinity when all weight concentrates on one particle |
| **Covariance symmetrize** in Kalman | Accumulated floating-point drift can break positive-definiteness; symmetrize at predict + update steps |
| **Log-price space** everywhere | Scale invariance — a $10 stock and a $10,000 stock produce identical filter dynamics |

---

## Rust Dependency Stack

| Crate | Version | Role |
|---|---|---|
| `pyo3` | 0.27 | Python ↔ Rust bindings, GIL management |
| `numpy` | 0.27 | Zero-copy NumPy ndarray interop |
| `ndarray` | 0.16 | Owned N-dimensional array construction |
| `rayon` | 1.10 | Work-stealing thread pool for `predict_particles` |

Build tool: **maturin** 1.11+ (compiles Rust → `.so` → installs into virtualenv).

---

## Verification

The test suite (`tests/test_parity.py`) runs **34 test assertions** across all 19 functions:

| Test | What it validates |
|---|---|
| `test_predict_particles` | All 3 regime branches (RANGE, TREND, PANIC) |
| `test_update_weights` | Likelihood math + normalization sum-to-one |
| `test_transition_regimes` | Markov transitions match cumulative search |
| `test_systematic_resample` | Resampled indices + uniform weight reset |
| `test_effective_sample_size` | ESS formula accuracy |
| `test_estimate` | Weighted mean + regime probability vector |
| `test_kalman_update` | Predict-update cycle + covariance symmetry |
| `test_vwap_bands` | VWAP + bands at default σ=1.5, σ=2.0, and short data |
| `test_momentum_score` | Tanh-normalized momentum + insufficient data |
| `test_particle_price_variance` | Weighted variance of particle cloud |
| `test_ess_and_uncertainty_margin` | ESS ratio + dominance test (dominant + non-dominant) |
| `test_slope_confidence_interval` | CI calculation + negative variance guard |
| `test_is_slope_significant` | All direction × slope sign combinations |
| `test_kalman_slope_acceleration` | Accelerating, flat, and insufficient data cases |
| `test_cusum_test` | Stable (no alarm), mean-shift (alarm), and short data |
| `test_rolling_kurtosis` | Normal data, insufficient data, default window |
| `test_adaptive_vwap_sigma` | Fat tails, thin tails, clamping, custom params |
| `test_volatility_compression` | Normal, insufficient data, custom windows |
| `test_hurst_exponent` | Random walk, insufficient data, custom windows |

All tests use `seed=42`, `N=500` particles, `atol=1e-10`.

---

## Building a Wheel for Distribution

```bash
# Build a portable wheel (no virtualenv needed for this step)
maturin build --release

# Output: target/wheels/particle_filter_rs-0.1.0-cp3XX-...-linux_x86_64.whl

# Install anywhere:
pip install target/wheels/particle_filter_rs-*.whl
```

---

## Topics

`particle-filter` `sequential-monte-carlo` `bayesian-inference` `regime-switching` `state-estimation` `kalman-filter` `markov-chain` `signal-processing` `time-series` `rust` `pyo3` `numpy` `rayon` `high-performance-computing` `quantitative-finance` `stochastic-filtering`

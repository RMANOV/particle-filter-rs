# particle-filter-rs

**Rust + PyO3 particle filter core** — a compiled, zero-copy, GIL-free sequential Monte Carlo library for regime-switching state estimation.

All 9 numerical core functions ported to safe Rust. Parallel prediction via rayon. Bit-exact parity with the Python originals (verified to `1e-10` tolerance). One-line import swap to activate.

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

Auxiliary functions `kalman_update`, `calculate_vwap_bands`, and `calculate_momentum_score` provide supplementary signal-processing capabilities that can run alongside the core filter loop.

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
  transition_regimes: PASS
  systematic_resample: PASS
  effective_sample_size: PASS
  estimate: PASS
  kalman_update: PASS
  calculate_vwap_bands: PASS
  ...
==================================================
ALL 9 FUNCTIONS PASS PARITY TESTS
```

---

## Integration — One-Line Swap

```python
from particle_filter_rs import (
    predict_particles,
    update_weights,
    transition_regimes,
    systematic_resample,
    effective_sample_size,
    estimate,
    kalman_update,
    calculate_vwap_bands,
    calculate_momentum_score,
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

## Project Structure

```
particle-filter-rs/
├── Cargo.toml              # Rust dependencies
├── pyproject.toml           # maturin build configuration
├── README.md
├── src/
│   ├── lib.rs              # PyO3 module — registers all 9 functions
│   ├── predict.rs          # predict_particles     (parallel, rayon)
│   ├── update.rs           # update_weights        (Gaussian likelihood)
│   ├── regime.rs           # transition_regimes    (Markov chain)
│   ├── resample.rs         # systematic_resample + effective_sample_size
│   ├── estimate.rs         # estimate              (weighted aggregation)
│   ├── kalman.rs           # kalman_update         (2D state-space)
│   ├── vwap.rs             # calculate_vwap_bands  (volume-weighted)
│   └── momentum.rs         # calculate_momentum_score (tanh-normalized)
└── tests/
    └── test_parity.py      # Rust vs Python — 12 test cases, atol=1e-10
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

The test suite (`tests/test_parity.py`) runs **12 test cases** across all 9 functions:

| Test | What it validates |
|---|---|
| `test_predict_particles` | All 3 regime branches (RANGE, TREND, PANIC) |
| `test_update_weights` | Likelihood math + normalization sum-to-one |
| `test_transition_regimes` | Markov transitions match cumulative search |
| `test_systematic_resample` | Resampled indices + uniform weight reset |
| `test_effective_sample_size` | ESS formula accuracy |
| `test_estimate` | Weighted mean + regime probability vector |
| `test_kalman_update` | Predict-update cycle + covariance symmetry |
| `test_vwap_bands` | VWAP + bands at default σ=1.5 |
| `test_vwap_bands` (σ=2.0) | Custom band width multiplier |
| `test_vwap_bands` (short data) | Returns NaN gracefully |
| `test_momentum_score` | Tanh-normalized momentum value |
| `test_momentum_score` (short) | Returns 0.0 for insufficient data |

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

# particle-filter-rs

Rust + PyO3 drop-in replacement for `numba_particle_filter.py` — the JIT-compiled particle filter core used by the trading bot's Entry/Exit engines.

## Functions

| Function | Description |
|---|---|
| `predict_particles` | Parallel particle prediction with regime-specific dynamics (rayon) |
| `update_weights` | Observation likelihood weighting with normalization |
| `transition_regimes` | Markov chain regime transitions |
| `systematic_resample` | O(N) systematic resampling |
| `effective_sample_size` | ESS = 1/sum(w²) |
| `estimate` | Weighted mean + regime probabilities |
| `kalman_update` | 2D Kalman filter [level, slope] |
| `calculate_vwap_bands` | VWAP + σ-bands |
| `calculate_momentum_score` | Normalized momentum [-1, 1] |

## Build

```bash
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy
maturin develop --release
```

## Usage

```python
import particle_filter_rs as pf

# Drop-in replacement: change import in Entry/Exit engines
# from indicators.numba_particle_filter import numba_predict_particles
# to:
# from particle_filter_rs import predict_particles

result = pf.predict_particles(particles, regimes, noise_pos, noise_vel,
                              imbalance, dt, vel_gain, rand_pos, rand_vel)
```

## Test

```bash
python tests/test_parity.py
```

## Dependencies

- Rust: pyo3 0.27, numpy 0.27, rayon 1.10, ndarray 0.16
- Python: numpy

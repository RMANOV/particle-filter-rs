use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Calculate weighted mean estimates and regime probabilities.
///
/// Returns: (mean_log_price, mean_velocity, regime_probs[3])
#[pyfunction]
pub fn estimate<'py>(
    py: Python<'py>,
    particles: PyReadonlyArray2<'py, f64>,
    weights: PyReadonlyArray1<'py, f64>,
    regimes: PyReadonlyArray1<'py, i64>,
) -> PyResult<(f64, f64, Bound<'py, PyArray1<f64>>)> {
    let n = particles.shape()[0];
    let parts = particles.as_array();
    let w = weights.as_array();
    let regs = regimes.as_array();

    let mut mean_price = 0.0f64;
    let mut mean_vel = 0.0f64;

    for i in 0..n {
        let wi = w[i];
        mean_price += parts[[i, 0]] * wi;
        mean_vel += parts[[i, 1]] * wi;
    }

    let mut regime_probs = [0.0f64; 3];
    for i in 0..n {
        let r = regs[i] as usize;
        regime_probs[r] += w[i];
    }

    Ok((
        mean_price,
        mean_vel,
        PyArray1::from_vec(py, regime_probs.to_vec()),
    ))
}

/// Compute weighted variance of particle log-prices.
#[pyfunction]
pub fn particle_price_variance(
    particles_pos: PyReadonlyArray1<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    mean_price: f64,
) -> PyResult<f64> {
    let pos = particles_pos.as_array();
    let w = weights.as_array();
    let n = pos.len();

    let mut variance = 0.0f64;
    for i in 0..n {
        let diff = pos[i] - mean_price;
        variance += w[i] * diff * diff;
    }

    Ok(variance)
}

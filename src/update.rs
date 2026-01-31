use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Update particle weights based on observation likelihood.
///
/// Combines price observation likelihood and velocity-imbalance alignment likelihood.
/// Returns normalized weights.
#[pyfunction]
pub fn update_weights<'py>(
    py: Python<'py>,
    particles: PyReadonlyArray2<'py, f64>,
    regimes: PyReadonlyArray1<'py, i64>,
    weights: PyReadonlyArray1<'py, f64>,
    measured_log_price: f64,
    meas_noise_price: PyReadonlyArray1<'py, f64>,
    meas_noise_vel: PyReadonlyArray1<'py, f64>,
    imbalance: f64,
    vel_gain: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n = particles.shape()[0];
    let parts = particles.as_array();
    let regs = regimes.as_array();
    let w = weights.as_array();
    let mnp = meas_noise_price.as_array();
    let mnv = meas_noise_vel.as_array();

    let target_vel = vel_gain * imbalance;
    let mut new_weights: Vec<f64> = w.to_vec();

    for i in 0..n {
        let r = regs[i] as usize;
        let pos = parts[[i, 0]];
        let vel = parts[[i, 1]];

        // Price likelihood
        let diff_price = measured_log_price - pos;
        let sigma2_price = mnp[r] * mnp[r] + 1e-12;
        let like_price = (-0.5 * diff_price * diff_price / sigma2_price).exp();

        // Velocity-imbalance alignment likelihood
        let diff_vel = vel - target_vel;
        let sigma2_vel = mnv[r] * mnv[r] + 1e-12;
        let like_vel = (-0.5 * diff_vel * diff_vel / sigma2_vel).exp();

        new_weights[i] *= like_price * like_vel;
    }

    // Normalize with underflow protection
    for w in new_weights.iter_mut() {
        *w += 1e-300;
    }
    let total: f64 = new_weights.iter().sum();
    for w in new_weights.iter_mut() {
        *w /= total;
    }

    Ok(PyArray1::from_vec(py, new_weights))
}

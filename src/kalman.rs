use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// 2D Kalman filter update for [level, slope] state.
///
/// State transition: level_new = level + slope * dt
/// Observation: z = level + noise
///
/// Returns: (new_level, new_slope, new_P)
#[pyfunction]
pub fn kalman_update<'py>(
    py: Python<'py>,
    level: f64,
    slope: f64,
    p: PyReadonlyArray2<'py, f64>,
    measured_log_price: f64,
    dt: f64,
    q_level: f64,
    q_slope: f64,
    r: f64,
) -> PyResult<(f64, f64, Bound<'py, PyArray2<f64>>)> {
    let p_arr = p.as_array();

    let p00 = p_arr[[0, 0]];
    let p01 = p_arr[[0, 1]];
    let p10 = p_arr[[1, 0]];
    let p11 = p_arr[[1, 1]];

    // PREDICT
    let level_pred = level + slope * dt;
    let slope_pred = slope;

    let ql = q_level * dt;
    let qs = q_slope * dt;

    // Predicted covariance
    let p00p = p00 + dt * (p10 + p01) + dt * dt * p11 + ql;
    let p01p_raw = p01 + dt * p11;
    let p10p_raw = p10 + dt * p11;
    let p11p = p11 + qs;

    // Symmetrize
    let off = 0.5 * (p01p_raw + p10p_raw);

    // UPDATE
    let s = p00p + r + 1e-12;
    let innovation = measured_log_price - level_pred;

    let k0 = p00p / s;
    let k1 = off / s; // p10p / s, but p10p = off after symmetrize

    let level_new = level_pred + k0 * innovation;
    let slope_new = slope_pred + k1 * innovation;

    // Updated covariance
    let p00n = (1.0 - k0) * p00p;
    let p01n = (1.0 - k0) * off;
    let p10n = -k1 * p00p + off;
    let p11n = -k1 * off + p11p;

    // Symmetrize
    let off2 = 0.5 * (p01n + p10n);

    let mut new_p = Array2::<f64>::zeros((2, 2));
    new_p[[0, 0]] = p00n;
    new_p[[0, 1]] = off2;
    new_p[[1, 0]] = off2;
    new_p[[1, 1]] = p11n;

    Ok((
        level_new,
        slope_new,
        PyArray2::from_owned_array(py, new_p),
    ))
}

use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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

/// Internal: compute slope confidence interval (shared logic).
fn slope_ci_inner(kf_slope: f64, kf_p_11: f64) -> (f64, f64, f64) {
    let var_slope = f64::max(kf_p_11, 1e-12);
    let sigma = var_slope.sqrt();
    let ci_95 = 1.96 * sigma;
    (kf_slope, sigma, ci_95)
}

/// Compute Kalman slope 95% confidence interval.
///
/// Returns: (slope, sigma, ci_95)
#[pyfunction]
pub fn slope_confidence_interval(kf_slope: f64, kf_p_11: f64) -> PyResult<(f64, f64, f64)> {
    Ok(slope_ci_inner(kf_slope, kf_p_11))
}

/// Test if slope is statistically significant at 95% confidence.
///
/// direction: 0=bidirectional, 1=positive only, -1=negative only
#[pyfunction]
pub fn is_slope_significant(kf_slope: f64, kf_p_11: f64, direction: i32) -> PyResult<bool> {
    let (slope, _sigma, ci_95) = slope_ci_inner(kf_slope, kf_p_11);

    if ci_95 <= 0.0 {
        return Ok(false);
    }

    let result = if direction == 0 {
        slope.abs() > ci_95
    } else if direction > 0 {
        slope > ci_95
    } else if direction < 0 {
        slope < -ci_95
    } else {
        false
    };

    Ok(result)
}

/// Detect acceleration of Kalman slope (second derivative).
///
/// Returns: (acceleration, is_accelerating)
#[pyfunction]
#[pyo3(signature = (slopes_history, lookback=10))]
pub fn kalman_slope_acceleration(
    slopes_history: PyReadonlyArray1<'_, f64>,
    lookback: usize,
) -> PyResult<(f64, bool)> {
    let slopes = slopes_history.as_array();
    let n = slopes.len();

    if n < lookback || lookback < 4 {
        return Ok((0.0, false));
    }

    let start = n - lookback;

    // Slope changes (first derivative)
    let changes_len = lookback - 1;
    let mut slope_changes = vec![0.0f64; changes_len];
    for i in 0..changes_len {
        slope_changes[i] = slopes[start + i + 1] - slopes[start + i];
    }

    // Acceleration: compare recent half vs older half
    let half = changes_len / 2;
    if half < 2 {
        return Ok((0.0, false));
    }

    let mut recent_sum = 0.0f64;
    for i in half..changes_len {
        recent_sum += slope_changes[i];
    }
    let recent_avg = recent_sum / (changes_len - half) as f64;

    let mut older_sum = 0.0f64;
    for i in 0..half {
        older_sum += slope_changes[i];
    }
    let older_avg = older_sum / half as f64;

    let acceleration = recent_avg - older_avg;

    // Standard deviation of slope changes
    let mut mean_change = 0.0f64;
    for i in 0..changes_len {
        mean_change += slope_changes[i];
    }
    mean_change /= changes_len as f64;

    let mut var_change = 0.0f64;
    for i in 0..changes_len {
        let diff = slope_changes[i] - mean_change;
        var_change += diff * diff;
    }
    var_change /= (lookback - 2).max(1) as f64;
    let std_change = (var_change + 1e-12).sqrt();

    let threshold = 2.0 * std_change;
    let is_accelerating = acceleration.abs() > threshold && acceleration.abs() > 1e-8;

    Ok((acceleration, is_accelerating))
}

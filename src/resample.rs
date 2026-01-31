use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// O(N) systematic resampling.
///
/// Returns: (new_particles, new_regimes, uniform_weights)
#[pyfunction]
pub fn systematic_resample<'py>(
    py: Python<'py>,
    weights: PyReadonlyArray1<'py, f64>,
    particles: PyReadonlyArray2<'py, f64>,
    regimes: PyReadonlyArray1<'py, i64>,
    start_offset: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let n = weights.shape()[0];
    let w = weights.as_array();
    let parts = particles.as_array();
    let regs = regimes.as_array();

    // Build cumulative sum
    let mut cumsum = vec![0.0f64; n];
    cumsum[0] = w[0];
    for i in 1..n {
        cumsum[i] = cumsum[i - 1] + w[i];
    }
    cumsum[n - 1] = 1.0; // Ensure last element is exactly 1

    let step = 1.0 / n as f64;

    // Two-pointer resampling
    let mut new_particles = Array2::<f64>::zeros((n, 2));
    let mut new_regimes = vec![0i64; n];
    let uniform_weight = 1.0 / n as f64;

    let mut j = 0usize;
    for i in 0..n {
        let pos = start_offset + step * i as f64;
        while j < n - 1 && cumsum[j] < pos {
            j += 1;
        }
        new_particles[[i, 0]] = parts[[j, 0]];
        new_particles[[i, 1]] = parts[[j, 1]];
        new_regimes[i] = regs[j];
    }

    let uniform_weights = vec![uniform_weight; n];

    Ok((
        PyArray2::from_owned_array(py, new_particles),
        PyArray1::from_vec(py, new_regimes),
        PyArray1::from_vec(py, uniform_weights),
    ))
}

/// Calculate ESS = 1 / sum(w^2)
#[pyfunction]
pub fn effective_sample_size(weights: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let w = weights.as_array();
    let sum_sq: f64 = w.iter().map(|x| x * x).sum();
    Ok(1.0 / (sum_sq + 1e-12))
}

/// Compute ESS ratio, uncertainty margin, and trend dominance test.
///
/// Returns: (ess_ratio, uncertainty_margin, is_dominant)
#[pyfunction]
pub fn ess_and_uncertainty_margin(
    weights: PyReadonlyArray1<'_, f64>,
    p_trend: f64,
    p_range: f64,
    p_panic: f64,
) -> PyResult<(f64, f64, bool)> {
    let w = weights.as_array();
    let n = w.len();

    let mut sum_sq = 0.0f64;
    for i in 0..n {
        sum_sq += w[i] * w[i];
    }

    let ess = 1.0 / (sum_sq + 1e-12);
    let ess_ratio = (ess / n as f64).clamp(0.0, 1.0);

    // Statistical uncertainty margin (2*SE for 95% CI)
    let uncertainty_margin = 2.0 * (0.25 / f64::max(ess, 100.0)).sqrt();

    let is_dominant = p_trend > p_range + uncertainty_margin
        && p_trend > p_panic + uncertainty_margin
        && p_trend > 0.45;

    Ok((ess_ratio, uncertainty_margin, is_dominant))
}

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Calculate normalized momentum score [-1, 1].
///
/// Compares recent average to older average in log-space.
/// Uses tanh normalization with 200x scaling for sensitivity.
#[pyfunction]
pub fn calculate_momentum_score(prices: PyReadonlyArray1<'_, f64>, window: usize) -> PyResult<f64> {
    let p = prices.as_array();
    let n = p.len();

    if n < window || n < 3 {
        return Ok(0.0);
    }

    let start = if n > window { n - window } else { 0 };
    let mid = start + (n - start) / 2;

    let mut recent_sum = 0.0f64;
    let mut recent_count = 0usize;
    for i in mid..n {
        recent_sum += p[i];
        recent_count += 1;
    }

    let mut older_sum = 0.0f64;
    let mut older_count = 0usize;
    for i in start..mid {
        older_sum += p[i];
        older_count += 1;
    }

    if recent_count == 0 || older_count == 0 {
        return Ok(0.0);
    }

    let recent_avg = recent_sum / recent_count as f64;
    let older_avg = older_sum / older_count as f64;

    let momentum = recent_avg - older_avg;

    Ok((momentum * 200.0).tanh())
}

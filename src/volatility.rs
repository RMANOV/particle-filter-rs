use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Detect volatility compression (range squeeze).
///
/// Returns: (compression_ratio, is_compressed, is_expanding)
#[pyfunction]
#[pyo3(signature = (prices, short_window=10, long_window=50))]
pub fn volatility_compression(
    prices: PyReadonlyArray1<'_, f64>,
    short_window: usize,
    long_window: usize,
) -> PyResult<(f64, bool, bool)> {
    let p = prices.as_array();
    let n = p.len();

    if n < long_window || n < 3 {
        return Ok((1.0, false, false));
    }

    // Calculate log-returns
    let returns_n = n - 1;
    let mut returns = vec![0.0f64; returns_n];
    for i in 0..returns_n {
        returns[i] = p[i + 1] - p[i];
    }

    if returns_n < long_window {
        return Ok((1.0, false, false));
    }

    // Short-term volatility
    let short_start = returns_n.saturating_sub(short_window);
    let short_count = returns_n - short_start;

    if short_count < 2 {
        return Ok((1.0, false, false));
    }

    let mut short_sum = 0.0f64;
    let mut short_sum_sq = 0.0f64;
    for i in short_start..returns_n {
        short_sum += returns[i];
        short_sum_sq += returns[i] * returns[i];
    }

    let short_mean = short_sum / short_count as f64;
    let short_var = (short_sum_sq / short_count as f64) - (short_mean * short_mean);
    let short_vol = (f64::max(short_var, 0.0) + 1e-12).sqrt();

    // Long-term volatility
    let long_start = returns_n.saturating_sub(long_window);
    let long_count = returns_n - long_start;

    let mut long_sum = 0.0f64;
    let mut long_sum_sq = 0.0f64;
    for i in long_start..returns_n {
        long_sum += returns[i];
        long_sum_sq += returns[i] * returns[i];
    }

    let long_mean = long_sum / long_count as f64;
    let long_var = (long_sum_sq / long_count as f64) - (long_mean * long_mean);
    let long_vol = (f64::max(long_var, 0.0) + 1e-12).sqrt();

    if long_vol < 1e-10 {
        return Ok((1.0, false, false));
    }

    let compression_ratio = short_vol / long_vol;
    let is_compressed = compression_ratio < 0.5;
    let is_expanding = compression_ratio > 1.5;

    Ok((compression_ratio, is_compressed, is_expanding))
}

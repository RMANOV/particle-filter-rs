use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Calculate Hurst exponent via Rescaled Range (R/S) analysis.
///
/// Returns: (H, uncertainty) where H in [0, 1]
/// H > 0.5: trending, H < 0.5: mean-reverting, H = 0.5: random walk
#[pyfunction]
#[pyo3(signature = (prices, min_window=10, max_window=50))]
pub fn hurst_exponent(
    prices: PyReadonlyArray1<'_, f64>,
    min_window: usize,
    max_window: usize,
) -> PyResult<(f64, f64)> {
    let p = prices.as_array();
    let n = p.len();

    if n < max_window {
        return Ok((0.5, 0.5));
    }

    // Calculate returns
    let returns_n = n - 1;
    let mut returns = vec![0.0f64; returns_n];
    for i in 0..returns_n {
        returns[i] = p[i + 1] - p[i];
    }

    // R/S analysis across window sizes
    let n_windows: usize = 5;
    let mut log_rs = [0.0f64; 5];
    let mut log_n = [0.0f64; 5];
    let mut valid_count: usize = 0;

    for w_idx in 0..n_windows {
        let window = min_window + (max_window - min_window) * w_idx / (n_windows - 1);
        if window > returns_n {
            continue;
        }

        let n_chunks = returns_n / window;
        if n_chunks == 0 {
            continue;
        }

        let mut rs_sum = 0.0f64;
        let mut rs_count: usize = 0;

        for chunk in 0..n_chunks {
            let start = chunk * window;
            let end = start + window;

            // Chunk mean
            let mut chunk_mean = 0.0f64;
            for i in start..end {
                chunk_mean += returns[i];
            }
            chunk_mean /= window as f64;

            // Cumulative deviations and range
            let mut cumsum = 0.0f64;
            let mut r_max = 0.0f64;
            let mut r_min = 0.0f64;

            for i in start..end {
                cumsum += returns[i] - chunk_mean;
                if cumsum > r_max {
                    r_max = cumsum;
                }
                if cumsum < r_min {
                    r_min = cumsum;
                }
            }

            let r = r_max - r_min;

            // Standard deviation
            let mut var = 0.0f64;
            for i in start..end {
                let diff = returns[i] - chunk_mean;
                var += diff * diff;
            }
            let s = (var / window as f64).sqrt();

            if s > 1e-12 {
                rs_sum += r / s;
                rs_count += 1;
            }
        }

        if rs_count > 0 {
            let rs_avg = rs_sum / rs_count as f64;
            log_rs[valid_count] = (rs_avg + 1e-12).ln();
            log_n[valid_count] = (window as f64).ln();
            valid_count += 1;
        }
    }

    if valid_count < 3 {
        return Ok((0.5, 0.5));
    }

    // Linear regression: log(R/S) = H * log(n) + c
    let mut mean_x = 0.0f64;
    let mut mean_y = 0.0f64;
    for i in 0..valid_count {
        mean_x += log_n[i];
        mean_y += log_rs[i];
    }
    mean_x /= valid_count as f64;
    mean_y /= valid_count as f64;

    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..valid_count {
        let dx = log_n[i] - mean_x;
        let dy = log_rs[i] - mean_y;
        num += dx * dy;
        den += dx * dx;
    }

    if den < 1e-12 {
        return Ok((0.5, 0.5));
    }

    let mut h = num / den;
    if h < 0.0 {
        h = 0.0;
    } else if h > 1.0 {
        h = 1.0;
    }

    // Uncertainty from residuals
    let mut ss_res = 0.0f64;
    for i in 0..valid_count {
        let pred = mean_y + h * (log_n[i] - mean_x);
        let resid = log_rs[i] - pred;
        ss_res += resid * resid;
    }

    let uncertainty = (ss_res / (valid_count - 2).max(1) as f64).sqrt();

    Ok((h, uncertainty))
}

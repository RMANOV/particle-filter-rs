use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Calculate VWAP and sigma bands.
///
/// Returns: (vwap, upper_band, lower_band) or (nan, nan, nan) if insufficient data
#[pyfunction]
#[pyo3(signature = (prices, volumes, window, band_sigma=1.5))]
pub fn calculate_vwap_bands(
    prices: PyReadonlyArray1<'_, f64>,
    volumes: PyReadonlyArray1<'_, f64>,
    window: usize,
    band_sigma: f64,
) -> PyResult<(f64, f64, f64)> {
    let p = prices.as_array();
    let v = volumes.as_array();
    let n = p.len();

    if n < window {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let start = n - window;

    let mut vol_sum = 0.0f64;
    let mut pv_sum = 0.0f64;

    for i in start..n {
        vol_sum += v[i];
        pv_sum += p[i] * v[i];
    }

    if vol_sum <= 0.0 {
        return Ok((f64::NAN, f64::NAN, f64::NAN));
    }

    let vwap = pv_sum / vol_sum;

    let mut var_sum = 0.0f64;
    for i in start..n {
        let diff = p[i] - vwap;
        var_sum += v[i] * diff * diff;
    }

    let variance = var_sum / vol_sum;
    let std = variance.sqrt();

    let upper = vwap + band_sigma * std;
    let lower = vwap - band_sigma * std;

    Ok((vwap, upper, lower))
}

/// Calculate excess kurtosis of recent values for fat-tail detection.
///
/// Returns excess kurtosis (>0 = fat tails, <0 = thin tails, 0 = normal)
#[pyfunction]
#[pyo3(signature = (values, window=50))]
pub fn rolling_kurtosis(
    values: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> PyResult<f64> {
    let v = values.as_array();
    let n = v.len();

    if n < window || n < 4 {
        return Ok(0.0);
    }

    let start = n - window;

    // Mean
    let mut mean = 0.0f64;
    for i in start..n {
        mean += v[i];
    }
    mean /= window as f64;

    // Second and fourth moments
    let mut m2 = 0.0f64;
    let mut m4 = 0.0f64;
    for i in start..n {
        let diff = v[i] - mean;
        let diff_sq = diff * diff;
        m2 += diff_sq;
        m4 += diff_sq * diff_sq;
    }

    m2 /= window as f64;
    m4 /= window as f64;

    if m2 < 1e-12 {
        return Ok(0.0);
    }

    let kurtosis = (m4 / (m2 * m2)) - 3.0;

    // Clamp to [-2, 10]
    Ok(kurtosis.clamp(-2.0, 10.0))
}

/// Calculate adaptive VWAP band sigma based on kurtosis.
#[pyfunction]
#[pyo3(signature = (kurtosis, base_sigma=1.5, min_sigma=1.2, max_sigma=2.5))]
pub fn adaptive_vwap_sigma(
    kurtosis: f64,
    base_sigma: f64,
    min_sigma: f64,
    max_sigma: f64,
) -> PyResult<f64> {
    let adaptive = base_sigma + 0.1 * kurtosis;
    Ok(adaptive.clamp(min_sigma, max_sigma))
}

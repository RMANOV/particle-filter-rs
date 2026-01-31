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

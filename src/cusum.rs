use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

/// Page's CUSUM test for structural break detection.
///
/// Returns: (is_reversal, direction, cusum_value)
/// direction: 1 = positive reversal, -1 = negative reversal, 0 = no reversal
#[pyfunction]
#[pyo3(signature = (values, threshold_h=0.5, min_samples=10))]
pub fn cusum_test(
    values: PyReadonlyArray1<'_, f64>,
    threshold_h: f64,
    min_samples: usize,
) -> PyResult<(bool, i32, f64)> {
    let vals = values.as_array();
    let n = vals.len();

    if n < min_samples {
        return Ok((false, 0, 0.0));
    }

    let half = n / 2;
    if half < 3 {
        return Ok((false, 0, 0.0));
    }

    // Reference mean from first half
    let mut mean = 0.0f64;
    for i in 0..half {
        mean += vals[i];
    }
    mean /= half as f64;

    // Standard deviation
    let mut var = 0.0f64;
    for i in 0..half {
        let diff = vals[i] - mean;
        var += diff * diff;
    }
    var /= (half - 1).max(1) as f64;
    let std = (var + 1e-12).sqrt();

    let h = threshold_h * std;
    let control_limit = 5.0 * std;

    // CUSUM accumulation
    let mut cusum_pos = 0.0f64;
    let mut cusum_neg = 0.0f64;

    for i in half..n {
        let x = vals[i];

        cusum_pos = f64::max(0.0, cusum_pos + (x - mean - h));
        cusum_neg = f64::min(0.0, cusum_neg + (x - mean + h));

        if cusum_pos > control_limit {
            return Ok((true, 1, cusum_pos));
        }
        if cusum_neg < -control_limit {
            return Ok((true, -1, cusum_neg));
        }
    }

    if cusum_pos.abs() > cusum_neg.abs() {
        Ok((false, 0, cusum_pos))
    } else {
        Ok((false, 0, cusum_neg))
    }
}

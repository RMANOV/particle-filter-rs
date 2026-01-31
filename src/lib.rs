use pyo3::prelude::*;

mod cusum;
mod estimate;
mod hurst;
mod kalman;
mod momentum;
mod predict;
mod regime;
mod resample;
mod update;
mod volatility;
mod vwap;

/// Rust-accelerated particle filter core functions.
/// Regime-switching sequential Monte Carlo with PyO3 bindings.
#[pymodule]
fn particle_filter_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(predict::predict_particles, m)?)?;
    m.add_function(wrap_pyfunction!(update::update_weights, m)?)?;
    m.add_function(wrap_pyfunction!(regime::transition_regimes, m)?)?;
    m.add_function(wrap_pyfunction!(resample::systematic_resample, m)?)?;
    m.add_function(wrap_pyfunction!(resample::effective_sample_size, m)?)?;
    m.add_function(wrap_pyfunction!(estimate::estimate, m)?)?;
    m.add_function(wrap_pyfunction!(kalman::kalman_update, m)?)?;
    m.add_function(wrap_pyfunction!(vwap::calculate_vwap_bands, m)?)?;
    m.add_function(wrap_pyfunction!(momentum::calculate_momentum_score, m)?)?;
    // Extended functions
    m.add_function(wrap_pyfunction!(estimate::particle_price_variance, m)?)?;
    m.add_function(wrap_pyfunction!(resample::ess_and_uncertainty_margin, m)?)?;
    m.add_function(wrap_pyfunction!(kalman::slope_confidence_interval, m)?)?;
    m.add_function(wrap_pyfunction!(kalman::is_slope_significant, m)?)?;
    m.add_function(wrap_pyfunction!(kalman::kalman_slope_acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(cusum::cusum_test, m)?)?;
    m.add_function(wrap_pyfunction!(vwap::rolling_kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(vwap::adaptive_vwap_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(volatility::volatility_compression, m)?)?;
    m.add_function(wrap_pyfunction!(hurst::hurst_exponent, m)?)?;
    Ok(())
}

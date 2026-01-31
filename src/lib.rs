use pyo3::prelude::*;

mod estimate;
mod kalman;
mod momentum;
mod predict;
mod regime;
mod resample;
mod update;
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
    Ok(())
}

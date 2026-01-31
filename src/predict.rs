use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Vectorized parallel particle prediction with regime-specific dynamics.
///
/// RANGE (0): Mean-reverting velocity with alpha=0.5
/// TREND (1): Velocity tracks imbalance with beta=0.3
/// PANIC (2): High-noise random walk
#[pyfunction]
pub fn predict_particles<'py>(
    py: Python<'py>,
    particles: PyReadonlyArray2<'py, f64>,
    regimes: PyReadonlyArray1<'py, i64>,
    process_noise_pos: PyReadonlyArray1<'py, f64>,
    process_noise_vel: PyReadonlyArray1<'py, f64>,
    imbalance: f64,
    dt: f64,
    vel_gain: f64,
    random_pos: PyReadonlyArray1<'py, f64>,
    random_vel: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = particles.shape()[0];
    let parts = particles.as_array();

    // Copy all data to owned buffers for safe parallel mutation
    let mut data: Vec<[f64; 2]> = (0..n).map(|i| [parts[[i, 0]], parts[[i, 1]]]).collect();
    let regs: Vec<i64> = regimes.as_array().to_vec();
    let np_arr = process_noise_pos.as_array();
    let nv_arr = process_noise_vel.as_array();
    let noise_pos: [f64; 3] = [np_arr[0], np_arr[1], np_arr[2]];
    let noise_vel: [f64; 3] = [nv_arr[0], nv_arr[1], nv_arr[2]];
    let rp: Vec<f64> = random_pos.as_array().to_vec();
    let rv: Vec<f64> = random_vel.as_array().to_vec();

    let dt_sqrt = dt.max(1e-8).sqrt();

    py.detach(|| {
        data.par_iter_mut().enumerate().for_each(|(i, particle)| {
            let r = regs[i];
            let pos = particle[0];
            let vel = particle[1];

            let (new_pos, new_vel) = match r {
                0 => {
                    // RANGE - mean reverting
                    let v = 0.5 * vel + rv[i] * noise_vel[0] * dt_sqrt;
                    let p = pos + v * dt + rp[i] * noise_pos[0] * dt_sqrt;
                    (p, v)
                }
                1 => {
                    // TREND - velocity tracks imbalance
                    let target_vel = vel_gain * imbalance;
                    let v =
                        vel + 0.3 * (target_vel - vel) * dt + rv[i] * noise_vel[1] * dt_sqrt;
                    let p = pos + v * dt + rp[i] * noise_pos[1] * dt_sqrt;
                    (p, v)
                }
                _ => {
                    // PANIC - high noise random walk
                    let v = vel + rv[i] * noise_vel[2] * dt_sqrt;
                    let p = pos + v * dt + rp[i] * noise_pos[2] * dt_sqrt;
                    (p, v)
                }
            };

            particle[0] = new_pos;
            particle[1] = new_vel;
        });
    });

    let mut result = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        result[[i, 0]] = data[i][0];
        result[[i, 1]] = data[i][1];
    }

    Ok(PyArray2::from_owned_array(py, result))
}

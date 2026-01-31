use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

/// Markov regime transitions using cumulative probability search.
#[pyfunction]
pub fn transition_regimes<'py>(
    py: Python<'py>,
    regimes: PyReadonlyArray1<'py, i64>,
    transition_matrix: PyReadonlyArray2<'py, f64>,
    random_uniform: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let n = regimes.shape()[0];
    let regs = regimes.as_array();
    let tm = transition_matrix.as_array();
    let ru = random_uniform.as_array();

    let mut new_regimes = vec![0i64; n];

    for i in 0..n {
        let r = regs[i] as usize;
        let u = ru[i];

        let mut cum_prob = 0.0;
        let mut new_regime = 2i64;

        for j in 0..3 {
            cum_prob += tm[[r, j]];
            if u < cum_prob {
                new_regime = j as i64;
                break;
            }
        }

        new_regimes[i] = new_regime;
    }

    Ok(PyArray1::from_vec(py, new_regimes))
}

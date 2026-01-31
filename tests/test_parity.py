"""
Parity tests: Rust (particle_filter_rs) vs pure-Python reference implementations.

Each test feeds identical inputs to both and asserts outputs match within
floating-point tolerance (1e-10).
"""

import numpy as np
import particle_filter_rs as pf

ATOL = 1e-10
N = 500  # Number of particles
np.random.seed(42)


# ─── Reference implementations (pure Python, matching Numba originals) ───


def ref_predict_particles(particles, regimes, noise_pos, noise_vel,
                          imbalance, dt, vel_gain, rand_pos, rand_vel):
    out = particles.copy()
    n = out.shape[0]
    dt_sqrt = max(dt, 1e-8) ** 0.5
    for i in range(n):
        r = regimes[i]
        pos, vel = out[i, 0], out[i, 1]
        if r == 0:
            vel = 0.5 * vel + rand_vel[i] * noise_vel[0] * dt_sqrt
            pos = pos + vel * dt + rand_pos[i] * noise_pos[0] * dt_sqrt
        elif r == 1:
            target = vel_gain * imbalance
            vel = vel + 0.3 * (target - vel) * dt + rand_vel[i] * noise_vel[1] * dt_sqrt
            pos = pos + vel * dt + rand_pos[i] * noise_pos[1] * dt_sqrt
        else:
            vel = vel + rand_vel[i] * noise_vel[2] * dt_sqrt
            pos = pos + vel * dt + rand_pos[i] * noise_pos[2] * dt_sqrt
        out[i, 0] = pos
        out[i, 1] = vel
    return out


def ref_update_weights(particles, regimes, weights, meas, mnp, mnv,
                       imbalance, vel_gain):
    w = weights.copy()
    target = vel_gain * imbalance
    for i in range(len(w)):
        r = regimes[i]
        dp = meas - particles[i, 0]
        s2p = mnp[r] ** 2 + 1e-12
        lp = np.exp(-0.5 * dp * dp / s2p)
        dv = particles[i, 1] - target
        s2v = mnv[r] ** 2 + 1e-12
        lv = np.exp(-0.5 * dv * dv / s2v)
        w[i] *= lp * lv
    w += 1e-300
    w /= w.sum()
    return w


def ref_transition_regimes(regimes, tm, ru):
    out = np.zeros_like(regimes)
    for i in range(len(regimes)):
        r = regimes[i]
        u = ru[i]
        cum = 0.0
        nr = 2
        for j in range(3):
            cum += tm[r, j]
            if u < cum:
                nr = j
                break
        out[i] = nr
    return out


def ref_systematic_resample(weights, particles, regimes, offset):
    n = len(weights)
    cumsum = np.cumsum(weights)
    cumsum[-1] = 1.0
    step = 1.0 / n
    new_p = np.zeros_like(particles)
    new_r = np.zeros_like(regimes)
    j = 0
    for i in range(n):
        pos = offset + step * i
        while j < n - 1 and cumsum[j] < pos:
            j += 1
        new_p[i] = particles[j]
        new_r[i] = regimes[j]
    uw = np.full(n, 1.0 / n)
    return new_p, new_r, uw


def ref_effective_sample_size(weights):
    return 1.0 / (np.sum(weights ** 2) + 1e-12)


def ref_estimate(particles, weights, regimes):
    mp = np.sum(particles[:, 0] * weights)
    mv = np.sum(particles[:, 1] * weights)
    rp = np.zeros(3)
    for i in range(len(weights)):
        rp[regimes[i]] += weights[i]
    return mp, mv, rp


def ref_kalman_update(level, slope, P, meas, dt, ql, qs, r):
    p00, p01, p10, p11 = P[0, 0], P[0, 1], P[1, 0], P[1, 1]
    lp = level + slope * dt
    sp = slope
    Ql = ql * dt
    Qs = qs * dt
    P00p = p00 + dt * (p10 + p01) + dt * dt * p11 + Ql
    P01p_raw = p01 + dt * p11
    P10p_raw = p10 + dt * p11
    P11p = p11 + Qs
    off = 0.5 * (P01p_raw + P10p_raw)
    S = P00p + r + 1e-12
    inn = meas - lp
    k0 = P00p / S
    k1 = off / S
    ln = lp + k0 * inn
    sn = sp + k1 * inn
    P00n = (1.0 - k0) * P00p
    P01n = (1.0 - k0) * off
    P10n = -k1 * P00p + off
    P11n = -k1 * off + P11p
    off2 = 0.5 * (P01n + P10n)
    nP = np.array([[P00n, off2], [off2, P11n]])
    return ln, sn, nP


def ref_vwap_bands(prices, volumes, window, sigma=1.5):
    n = len(prices)
    if n < window:
        return np.nan, np.nan, np.nan
    s = n - window
    vs = np.sum(volumes[s:])
    if vs <= 0:
        return np.nan, np.nan, np.nan
    vwap = np.sum(prices[s:] * volumes[s:]) / vs
    var = np.sum(volumes[s:] * (prices[s:] - vwap) ** 2) / vs
    std = var ** 0.5
    return vwap, vwap + sigma * std, vwap - sigma * std


def ref_momentum_score(prices, window):
    n = len(prices)
    if n < window or n < 3:
        return 0.0
    start = max(0, n - window)
    mid = start + (n - start) // 2
    rc = np.mean(prices[mid:n]) if mid < n else 0.0
    oc = np.mean(prices[start:mid]) if start < mid else 0.0
    if mid >= n or start >= mid:
        return 0.0
    return np.tanh((rc - oc) * 200.0)


def ref_particle_price_variance(particles_pos, weights, mean_price):
    N = len(particles_pos)
    variance = 0.0
    for i in range(N):
        diff = particles_pos[i] - mean_price
        variance += weights[i] * diff * diff
    return variance


def ref_ess_and_uncertainty_margin(weights, p_trend, p_range, p_panic):
    N = len(weights)
    sum_sq = 0.0
    for i in range(N):
        sum_sq += weights[i] * weights[i]
    ess = 1.0 / (sum_sq + 1e-12)
    ess_ratio = min(max(ess / N, 0.0), 1.0)
    uncertainty_margin = 2.0 * np.sqrt(0.25 / max(ess, 100.0))
    is_dominant = (
        p_trend > p_range + uncertainty_margin and
        p_trend > p_panic + uncertainty_margin and
        p_trend > 0.45
    )
    return (ess_ratio, uncertainty_margin, is_dominant)


def ref_slope_confidence_interval(kf_slope, kf_P_11):
    var_slope = max(kf_P_11, 1e-12)
    sigma = np.sqrt(var_slope)
    ci_95 = 1.96 * sigma
    return (kf_slope, sigma, ci_95)


def ref_is_slope_significant(kf_slope, kf_P_11, direction):
    slope, sigma, ci_95 = ref_slope_confidence_interval(kf_slope, kf_P_11)
    if ci_95 <= 0:
        return False
    if direction == 0:
        return abs(slope) > ci_95
    elif direction > 0:
        return slope > ci_95
    elif direction < 0:
        return slope < -ci_95
    return False


def ref_kalman_slope_acceleration(slopes_history, lookback=10):
    n = len(slopes_history)
    if n < lookback or lookback < 4:
        return (0.0, False)
    start = n - lookback
    slope_changes = np.zeros(lookback - 1, dtype=np.float64)
    for i in range(lookback - 1):
        slope_changes[i] = slopes_history[start + i + 1] - slopes_history[start + i]
    half = (lookback - 1) // 2
    if half < 2:
        return (0.0, False)
    recent_sum = 0.0
    for i in range(half, lookback - 1):
        recent_sum += slope_changes[i]
    recent_avg = recent_sum / (lookback - 1 - half)
    older_sum = 0.0
    for i in range(half):
        older_sum += slope_changes[i]
    older_avg = older_sum / half
    acceleration = recent_avg - older_avg
    mean_change = 0.0
    for i in range(lookback - 1):
        mean_change += slope_changes[i]
    mean_change /= (lookback - 1)
    var_change = 0.0
    for i in range(lookback - 1):
        diff = slope_changes[i] - mean_change
        var_change += diff * diff
    var_change /= max(lookback - 2, 1)
    std_change = np.sqrt(var_change + 1e-12)
    threshold = 2.0 * std_change
    is_accelerating = abs(acceleration) > threshold and abs(acceleration) > 1e-8
    return (acceleration, is_accelerating)


def ref_cusum_test(values, threshold_h=0.5, min_samples=10):
    n = len(values)
    if n < min_samples:
        return (False, 0, 0.0)
    half = n // 2
    if half < 3:
        return (False, 0, 0.0)
    mean = 0.0
    for i in range(half):
        mean += values[i]
    mean /= half
    var = 0.0
    for i in range(half):
        diff = values[i] - mean
        var += diff * diff
    var /= max(half - 1, 1)
    std = np.sqrt(var + 1e-12)
    h = threshold_h * std
    control_limit = 5.0 * std
    cusum_pos = 0.0
    cusum_neg = 0.0
    for i in range(half, n):
        x = values[i]
        cusum_pos = max(0.0, cusum_pos + (x - mean - h))
        cusum_neg = min(0.0, cusum_neg + (x - mean + h))
        if cusum_pos > control_limit:
            return (True, 1, cusum_pos)
        if cusum_neg < -control_limit:
            return (True, -1, cusum_neg)
    if abs(cusum_pos) > abs(cusum_neg):
        return (False, 0, cusum_pos)
    else:
        return (False, 0, cusum_neg)


def ref_rolling_kurtosis(values, window=50):
    n = len(values)
    if n < window or n < 4:
        return 0.0
    start = n - window
    mean = 0.0
    for i in range(start, n):
        mean += values[i]
    mean /= window
    m2 = 0.0
    m4 = 0.0
    for i in range(start, n):
        diff = values[i] - mean
        diff_sq = diff * diff
        m2 += diff_sq
        m4 += diff_sq * diff_sq
    m2 /= window
    m4 /= window
    if m2 < 1e-12:
        return 0.0
    kurtosis = (m4 / (m2 * m2)) - 3.0
    return max(-2.0, min(10.0, kurtosis))


def ref_adaptive_vwap_sigma(kurtosis, base_sigma=1.5, min_sigma=1.2, max_sigma=2.5):
    adaptive_sigma = base_sigma + 0.1 * kurtosis
    return max(min_sigma, min(max_sigma, adaptive_sigma))


def ref_volatility_compression(prices, short_window=10, long_window=50):
    n = len(prices)
    if n < long_window or n < 3:
        return (1.0, False, False)
    returns = np.zeros(n - 1, dtype=np.float64)
    for i in range(n - 1):
        returns[i] = prices[i + 1] - prices[i]
    returns_n = n - 1
    if returns_n < long_window:
        return (1.0, False, False)
    short_start = max(0, returns_n - short_window)
    short_sum = 0.0
    short_sum_sq = 0.0
    short_count = returns_n - short_start
    for i in range(short_start, returns_n):
        short_sum += returns[i]
        short_sum_sq += returns[i] * returns[i]
    if short_count < 2:
        return (1.0, False, False)
    short_mean = short_sum / short_count
    short_var = (short_sum_sq / short_count) - (short_mean * short_mean)
    short_vol = np.sqrt(max(short_var, 0.0) + 1e-12)
    long_start = max(0, returns_n - long_window)
    long_sum = 0.0
    long_sum_sq = 0.0
    long_count = returns_n - long_start
    for i in range(long_start, returns_n):
        long_sum += returns[i]
        long_sum_sq += returns[i] * returns[i]
    long_mean = long_sum / long_count
    long_var = (long_sum_sq / long_count) - (long_mean * long_mean)
    long_vol = np.sqrt(max(long_var, 0.0) + 1e-12)
    if long_vol < 1e-10:
        return (1.0, False, False)
    compression_ratio = short_vol / long_vol
    is_compressed = compression_ratio < 0.5
    is_expanding = compression_ratio > 1.5
    return (compression_ratio, is_compressed, is_expanding)


def ref_hurst_exponent(prices, min_window=10, max_window=50):
    n = len(prices)
    if n < max_window:
        return (0.5, 0.5)
    returns = np.zeros(n - 1, dtype=np.float64)
    for i in range(n - 1):
        returns[i] = prices[i + 1] - prices[i]
    n_windows = 5
    log_rs = np.zeros(n_windows, dtype=np.float64)
    log_n = np.zeros(n_windows, dtype=np.float64)
    valid_count = 0
    for w_idx in range(n_windows):
        window = min_window + (max_window - min_window) * w_idx // (n_windows - 1)
        if window > len(returns):
            continue
        n_chunks = len(returns) // window
        if n_chunks == 0:
            continue
        rs_sum = 0.0
        rs_count = 0
        for chunk in range(n_chunks):
            start = chunk * window
            end = start + window
            chunk_mean = 0.0
            for i in range(start, end):
                chunk_mean += returns[i]
            chunk_mean /= window
            cumsum = 0.0
            r_max = 0.0
            r_min = 0.0
            for i in range(start, end):
                cumsum += returns[i] - chunk_mean
                if cumsum > r_max:
                    r_max = cumsum
                if cumsum < r_min:
                    r_min = cumsum
            R = r_max - r_min
            var = 0.0
            for i in range(start, end):
                diff = returns[i] - chunk_mean
                var += diff * diff
            S = np.sqrt(var / window)
            if S > 1e-12:
                rs_sum += R / S
                rs_count += 1
        if rs_count > 0:
            rs_avg = rs_sum / rs_count
            log_rs[valid_count] = np.log(rs_avg + 1e-12)
            log_n[valid_count] = np.log(float(window))
            valid_count += 1
    if valid_count < 3:
        return (0.5, 0.5)
    mean_x = 0.0
    mean_y = 0.0
    for i in range(valid_count):
        mean_x += log_n[i]
        mean_y += log_rs[i]
    mean_x /= valid_count
    mean_y /= valid_count
    num = 0.0
    den = 0.0
    for i in range(valid_count):
        dx = log_n[i] - mean_x
        dy = log_rs[i] - mean_y
        num += dx * dy
        den += dx * dx
    if den < 1e-12:
        return (0.5, 0.5)
    H = num / den
    if H < 0.0:
        H = 0.0
    elif H > 1.0:
        H = 1.0
    ss_res = 0.0
    for i in range(valid_count):
        pred = mean_y + H * (log_n[i] - mean_x)
        ss_res += (log_rs[i] - pred) ** 2
    uncertainty = np.sqrt(ss_res / max(valid_count - 2, 1))
    return (H, uncertainty)


# ─── Tests ───


def test_predict_particles():
    particles = np.random.randn(N, 2).astype(np.float64)
    regimes = np.random.randint(0, 3, N).astype(np.int64)
    noise_pos = np.array([0.01, 0.02, 0.05])
    noise_vel = np.array([0.005, 0.01, 0.03])
    imbalance = 0.15
    dt = 1.0
    vel_gain = 0.5
    rand_pos = np.random.randn(N)
    rand_vel = np.random.randn(N)

    expected = ref_predict_particles(particles, regimes, noise_pos, noise_vel,
                                     imbalance, dt, vel_gain, rand_pos, rand_vel)
    result = pf.predict_particles(particles, regimes, noise_pos, noise_vel,
                                  imbalance, dt, vel_gain, rand_pos, rand_vel)
    assert np.allclose(result, expected, atol=ATOL), \
        f"predict_particles max diff: {np.max(np.abs(result - expected))}"
    print("  predict_particles: PASS")


def test_update_weights():
    particles = np.random.randn(N, 2).astype(np.float64)
    regimes = np.random.randint(0, 3, N).astype(np.int64)
    weights = np.full(N, 1.0 / N)
    meas = 0.5
    mnp = np.array([0.01, 0.02, 0.05])
    mnv = np.array([0.005, 0.01, 0.03])
    imbalance = 0.15
    vel_gain = 0.5

    expected = ref_update_weights(particles, regimes, weights, meas, mnp, mnv,
                                  imbalance, vel_gain)
    result = pf.update_weights(particles, regimes, weights, meas, mnp, mnv,
                               imbalance, vel_gain)
    assert np.allclose(result, expected, atol=ATOL), \
        f"update_weights max diff: {np.max(np.abs(result - expected))}"
    print("  update_weights: PASS")


def test_transition_regimes():
    regimes = np.random.randint(0, 3, N).astype(np.int64)
    tm = np.array([[0.8, 0.15, 0.05],
                   [0.1, 0.8, 0.1],
                   [0.2, 0.3, 0.5]])
    ru = np.random.rand(N)

    expected = ref_transition_regimes(regimes, tm, ru)
    result = pf.transition_regimes(regimes, tm, ru)
    assert np.array_equal(result, expected), "transition_regimes mismatch"
    print("  transition_regimes: PASS")


def test_systematic_resample():
    weights = np.random.dirichlet(np.ones(N))
    particles = np.random.randn(N, 2).astype(np.float64)
    regimes = np.random.randint(0, 3, N).astype(np.int64)
    offset = np.random.rand() / N

    ep, er, ew = ref_systematic_resample(weights, particles, regimes, offset)
    rp, rr, rw = pf.systematic_resample(weights, particles, regimes, offset)

    assert np.allclose(rp, ep, atol=ATOL), \
        f"systematic_resample particles max diff: {np.max(np.abs(rp - ep))}"
    assert np.array_equal(rr, er), "systematic_resample regimes mismatch"
    assert np.allclose(rw, ew, atol=ATOL), "systematic_resample weights mismatch"
    print("  systematic_resample: PASS")


def test_effective_sample_size():
    weights = np.random.dirichlet(np.ones(N))
    expected = ref_effective_sample_size(weights)
    result = pf.effective_sample_size(weights)
    assert abs(result - expected) < ATOL, \
        f"effective_sample_size diff: {abs(result - expected)}"
    print("  effective_sample_size: PASS")


def test_estimate():
    particles = np.random.randn(N, 2).astype(np.float64)
    weights = np.random.dirichlet(np.ones(N))
    regimes = np.random.randint(0, 3, N).astype(np.int64)

    emp, emv, erp = ref_estimate(particles, weights, regimes)
    rmp, rmv, rrp = pf.estimate(particles, weights, regimes)

    assert abs(rmp - emp) < ATOL, f"estimate price diff: {abs(rmp - emp)}"
    assert abs(rmv - emv) < ATOL, f"estimate vel diff: {abs(rmv - emv)}"
    assert np.allclose(rrp, erp, atol=ATOL), "estimate regime_probs mismatch"
    print("  estimate: PASS")


def test_kalman_update():
    level = 4.5
    slope = 0.001
    P = np.array([[0.01, 0.001], [0.001, 0.001]])
    meas = 4.52
    dt = 1.0
    ql = 0.001
    qs = 0.0001
    r = 0.01

    el, es, eP = ref_kalman_update(level, slope, P, meas, dt, ql, qs, r)
    rl, rs, rP = pf.kalman_update(level, slope, P, meas, dt, ql, qs, r)

    assert abs(rl - el) < ATOL, f"kalman level diff: {abs(rl - el)}"
    assert abs(rs - es) < ATOL, f"kalman slope diff: {abs(rs - es)}"
    assert np.allclose(rP, eP, atol=ATOL), \
        f"kalman P max diff: {np.max(np.abs(rP - eP))}"
    print("  kalman_update: PASS")


def test_vwap_bands():
    prices = np.cumsum(np.random.randn(100) * 0.01) + 4.5
    volumes = np.random.rand(100) * 1000 + 100
    window = 20

    ev, eu, el = ref_vwap_bands(prices, volumes, window)
    rv, ru, rl = pf.calculate_vwap_bands(prices, volumes, window)

    assert abs(rv - ev) < ATOL, f"vwap diff: {abs(rv - ev)}"
    assert abs(ru - eu) < ATOL, f"vwap upper diff: {abs(ru - eu)}"
    assert abs(rl - el) < ATOL, f"vwap lower diff: {abs(rl - el)}"
    print("  calculate_vwap_bands: PASS")

    # Test with custom band_sigma
    ev2, eu2, el2 = ref_vwap_bands(prices, volumes, window, sigma=2.0)
    rv2, ru2, rl2 = pf.calculate_vwap_bands(prices, volumes, window, band_sigma=2.0)
    assert abs(rv2 - ev2) < ATOL
    assert abs(ru2 - eu2) < ATOL
    assert abs(rl2 - el2) < ATOL
    print("  calculate_vwap_bands (sigma=2.0): PASS")

    # Test insufficient data
    rv3, ru3, rl3 = pf.calculate_vwap_bands(prices[:5], volumes[:5], 20)
    assert np.isnan(rv3) and np.isnan(ru3) and np.isnan(rl3)
    print("  calculate_vwap_bands (insufficient data): PASS")


def test_momentum_score():
    prices = np.cumsum(np.random.randn(100) * 0.01) + 4.5
    window = 20

    expected = ref_momentum_score(prices, window)
    result = pf.calculate_momentum_score(prices, window)

    assert abs(result - expected) < ATOL, \
        f"momentum_score diff: {abs(result - expected)}"
    print("  calculate_momentum_score: PASS")

    # Test insufficient data
    assert pf.calculate_momentum_score(prices[:2], 20) == 0.0
    print("  calculate_momentum_score (insufficient data): PASS")


def test_particle_price_variance():
    particles_pos = np.random.randn(N)
    weights = np.random.dirichlet(np.ones(N))
    mean_price = np.sum(particles_pos * weights)

    expected = ref_particle_price_variance(particles_pos, weights, mean_price)
    result = pf.particle_price_variance(particles_pos, weights, mean_price)
    assert abs(result - expected) < ATOL, \
        f"particle_price_variance diff: {abs(result - expected)}"
    print("  particle_price_variance: PASS")


def test_ess_and_uncertainty_margin():
    weights = np.random.dirichlet(np.ones(N))
    p_trend, p_range, p_panic = 0.6, 0.25, 0.15

    e_ess, e_unc, e_dom = ref_ess_and_uncertainty_margin(weights, p_trend, p_range, p_panic)
    r_ess, r_unc, r_dom = pf.ess_and_uncertainty_margin(weights, p_trend, p_range, p_panic)

    assert abs(r_ess - e_ess) < ATOL, f"ess_ratio diff: {abs(r_ess - e_ess)}"
    assert abs(r_unc - e_unc) < ATOL, f"uncertainty_margin diff: {abs(r_unc - e_unc)}"
    assert r_dom == e_dom, f"is_dominant mismatch: rust={r_dom} py={e_dom}"
    print("  ess_and_uncertainty_margin: PASS")

    # Test non-dominant case
    e2 = ref_ess_and_uncertainty_margin(weights, 0.3, 0.35, 0.35)
    r2 = pf.ess_and_uncertainty_margin(weights, 0.3, 0.35, 0.35)
    assert r2[2] == e2[2], "is_dominant non-dominant mismatch"
    print("  ess_and_uncertainty_margin (non-dominant): PASS")


def test_slope_confidence_interval():
    kf_slope = 0.005
    kf_P_11 = 0.001

    e_s, e_sig, e_ci = ref_slope_confidence_interval(kf_slope, kf_P_11)
    r_s, r_sig, r_ci = pf.slope_confidence_interval(kf_slope, kf_P_11)

    assert abs(r_s - e_s) < ATOL, f"slope diff: {abs(r_s - e_s)}"
    assert abs(r_sig - e_sig) < ATOL, f"sigma diff: {abs(r_sig - e_sig)}"
    assert abs(r_ci - e_ci) < ATOL, f"ci_95 diff: {abs(r_ci - e_ci)}"
    print("  slope_confidence_interval: PASS")

    # Negative variance guard
    r2 = pf.slope_confidence_interval(0.1, -0.5)
    e2 = ref_slope_confidence_interval(0.1, -0.5)
    assert abs(r2[1] - e2[1]) < ATOL, "negative variance guard failed"
    print("  slope_confidence_interval (negative P_11): PASS")


def test_is_slope_significant():
    # Significant positive slope
    assert pf.is_slope_significant(0.1, 0.001, 0) == \
        ref_is_slope_significant(0.1, 0.001, 0)

    # Not significant (slope too small)
    assert pf.is_slope_significant(0.001, 0.01, 0) == \
        ref_is_slope_significant(0.001, 0.01, 0)

    # Directional tests
    for direction in [-1, 0, 1]:
        for slope in [-0.1, -0.001, 0.001, 0.1]:
            r = pf.is_slope_significant(slope, 0.001, direction)
            e = ref_is_slope_significant(slope, 0.001, direction)
            assert r == e, \
                f"is_slope_significant mismatch: slope={slope}, dir={direction}, rust={r}, py={e}"
    print("  is_slope_significant: PASS")


def test_kalman_slope_acceleration():
    # Accelerating slopes
    slopes = np.array([0.001 * i**2 for i in range(20)], dtype=np.float64)
    e_acc, e_flag = ref_kalman_slope_acceleration(slopes, 10)
    r_acc, r_flag = pf.kalman_slope_acceleration(slopes, 10)
    assert abs(r_acc - e_acc) < ATOL, f"acceleration diff: {abs(r_acc - e_acc)}"
    assert r_flag == e_flag, f"is_accelerating mismatch: rust={r_flag} py={e_flag}"
    print("  kalman_slope_acceleration: PASS")

    # Constant slopes (no acceleration)
    flat = np.full(20, 0.01)
    e2 = ref_kalman_slope_acceleration(flat, 10)
    r2 = pf.kalman_slope_acceleration(flat, 10)
    assert abs(r2[0] - e2[0]) < ATOL
    assert r2[1] == e2[1]
    print("  kalman_slope_acceleration (flat): PASS")

    # Insufficient data
    short = np.array([0.1, 0.2])
    r3 = pf.kalman_slope_acceleration(short, 10)
    assert r3 == (0.0, False)
    print("  kalman_slope_acceleration (insufficient data): PASS")


def test_cusum_test():
    # No reversal: stable data
    stable = np.random.randn(100) * 0.1
    e_rev, e_dir, e_val = ref_cusum_test(stable)
    r_rev, r_dir, r_val = pf.cusum_test(stable)
    assert r_rev == e_rev, f"cusum is_reversal mismatch: rust={r_rev} py={e_rev}"
    assert r_dir == e_dir, f"cusum direction mismatch: rust={r_dir} py={e_dir}"
    assert abs(r_val - e_val) < ATOL, f"cusum value diff: {abs(r_val - e_val)}"
    print("  cusum_test (stable): PASS")

    # Reversal: mean shift
    shift = np.concatenate([np.random.randn(50) * 0.1, np.random.randn(50) * 0.1 + 5.0])
    e2 = ref_cusum_test(shift)
    r2 = pf.cusum_test(shift)
    assert r2[0] == e2[0], f"cusum reversal mismatch: rust={r2[0]} py={e2[0]}"
    assert r2[1] == e2[1], f"cusum direction mismatch: rust={r2[1]} py={e2[1]}"
    assert abs(r2[2] - e2[2]) < ATOL
    print("  cusum_test (shift): PASS")

    # Insufficient data
    r3 = pf.cusum_test(np.array([1.0, 2.0, 3.0]))
    assert r3 == (False, 0, 0.0)
    print("  cusum_test (insufficient data): PASS")


def test_rolling_kurtosis():
    # Normal-ish data
    data = np.random.randn(100)
    expected = ref_rolling_kurtosis(data, 50)
    result = pf.rolling_kurtosis(data, 50)
    assert abs(result - expected) < ATOL, \
        f"rolling_kurtosis diff: {abs(result - expected)}"
    print("  rolling_kurtosis: PASS")

    # Insufficient data
    assert pf.rolling_kurtosis(np.array([1.0, 2.0]), 50) == 0.0
    print("  rolling_kurtosis (insufficient data): PASS")

    # Default window
    e2 = ref_rolling_kurtosis(data)
    r2 = pf.rolling_kurtosis(data)
    assert abs(r2 - e2) < ATOL
    print("  rolling_kurtosis (default window): PASS")


def test_adaptive_vwap_sigma():
    # Normal kurtosis
    assert abs(pf.adaptive_vwap_sigma(0.0) - ref_adaptive_vwap_sigma(0.0)) < ATOL

    # Fat tails
    assert abs(pf.adaptive_vwap_sigma(5.0) - ref_adaptive_vwap_sigma(5.0)) < ATOL

    # Thin tails
    assert abs(pf.adaptive_vwap_sigma(-2.0) - ref_adaptive_vwap_sigma(-2.0)) < ATOL

    # Clamping
    assert abs(pf.adaptive_vwap_sigma(100.0) - ref_adaptive_vwap_sigma(100.0)) < ATOL
    assert abs(pf.adaptive_vwap_sigma(-100.0) - ref_adaptive_vwap_sigma(-100.0)) < ATOL

    # Custom params
    r = pf.adaptive_vwap_sigma(3.0, base_sigma=2.0, min_sigma=1.0, max_sigma=3.0)
    e = ref_adaptive_vwap_sigma(3.0, base_sigma=2.0, min_sigma=1.0, max_sigma=3.0)
    assert abs(r - e) < ATOL
    print("  adaptive_vwap_sigma: PASS")


def test_volatility_compression():
    prices = np.cumsum(np.random.randn(200) * 0.01) + 4.5

    e_ratio, e_comp, e_exp = ref_volatility_compression(prices)
    r_ratio, r_comp, r_exp = pf.volatility_compression(prices)
    assert abs(r_ratio - e_ratio) < ATOL, \
        f"compression_ratio diff: {abs(r_ratio - e_ratio)}"
    assert r_comp == e_comp, f"is_compressed mismatch: rust={r_comp} py={e_comp}"
    assert r_exp == e_exp, f"is_expanding mismatch: rust={r_exp} py={e_exp}"
    print("  volatility_compression: PASS")

    # Insufficient data
    r2 = pf.volatility_compression(np.array([1.0, 2.0]))
    assert r2 == (1.0, False, False)
    print("  volatility_compression (insufficient data): PASS")

    # Custom windows
    e3 = ref_volatility_compression(prices, 5, 30)
    r3 = pf.volatility_compression(prices, 5, 30)
    assert abs(r3[0] - e3[0]) < ATOL
    print("  volatility_compression (custom windows): PASS")


def test_hurst_exponent():
    prices = np.cumsum(np.random.randn(200) * 0.01) + 4.5

    e_h, e_u = ref_hurst_exponent(prices)
    r_h, r_u = pf.hurst_exponent(prices)
    assert abs(r_h - e_h) < ATOL, f"hurst H diff: {abs(r_h - e_h)}"
    assert abs(r_u - e_u) < ATOL, f"hurst uncertainty diff: {abs(r_u - e_u)}"
    print("  hurst_exponent: PASS")

    # Insufficient data
    r2 = pf.hurst_exponent(np.array([1.0, 2.0, 3.0]))
    assert r2 == (0.5, 0.5)
    print("  hurst_exponent (insufficient data): PASS")

    # Custom windows
    e3 = ref_hurst_exponent(prices, 5, 30)
    r3 = pf.hurst_exponent(prices, 5, 30)
    assert abs(r3[0] - e3[0]) < ATOL
    assert abs(r3[1] - e3[1]) < ATOL
    print("  hurst_exponent (custom windows): PASS")


if __name__ == "__main__":
    print("Particle Filter Rust vs Python Parity Tests")
    print("=" * 50)

    test_predict_particles()
    test_update_weights()
    test_transition_regimes()
    test_systematic_resample()
    test_effective_sample_size()
    test_estimate()
    test_kalman_update()
    test_vwap_bands()
    test_momentum_score()

    print("-" * 50)
    print("New functions:")

    test_particle_price_variance()
    test_ess_and_uncertainty_margin()
    test_slope_confidence_interval()
    test_is_slope_significant()
    test_kalman_slope_acceleration()
    test_cusum_test()
    test_rolling_kurtosis()
    test_adaptive_vwap_sigma()
    test_volatility_compression()
    test_hurst_exponent()

    print("=" * 50)
    print("ALL 19 FUNCTIONS PASS PARITY TESTS")

"""
Shared signal processing utilities for higher accuracy across all agents.
These functions implement more robust, literature-aligned preprocessing and metrics
used by the biomechanical, kinematic, statistical, and other agents.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

def safe_array(x, dtype=float) -> np.ndarray:
    """Robust conversion with NaN handling."""
    arr = np.asarray(x, dtype=dtype)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr

def safe_mean(x, default=0.0) -> float:
    """Mean with empty-safe guard. Returns default if array is empty."""
    arr = safe_array(x)
    if len(arr) == 0:
        return default
    return float(np.mean(arr))

def central_difference(y: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """2nd-order central difference for velocity/derivative. More accurate than forward diff."""
    y = safe_array(y)
    n = len(y)
    if n < 3:
        return np.zeros(n)
    dy = np.zeros(n)
    dy[0] = (y[1] - y[0]) / dt
    dy[-1] = (y[-1] - y[-2]) / dt
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * dt)
    return dy

def second_derivative(y: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Central 2nd derivative (acceleration)."""
    y = safe_array(y)
    n = len(y)
    if n < 3:
        return np.zeros(n)
    d2 = np.zeros(n)
    d2[0] = d2[1] = (y[2] - 2*y[1] + y[0]) / (dt**2)
    d2[-1] = d2[-2] = (y[-1] - 2*y[-2] + y[-3]) / (dt**2)
    d2[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / (dt**2)
    return d2

def percent_cycle_normalize(signal: np.ndarray, n_points: int = 101) -> np.ndarray:
    """
    Resample signal to 0-100% of cycle using linear interpolation.
    Critical for accurate Pearson consistency and phase comparisons across variable-length trials.
    """
    signal = safe_array(signal)
    n = len(signal)
    if n < 3:
        return np.full(n_points, np.mean(signal) if n > 0 else 0.0)
    x_old = np.linspace(0, 100, n)
    x_new = np.linspace(0, 100, n_points)
    return np.interp(x_new, x_old, signal)

def multi_channel_percent_normalize(channels: List[List[float]], n_points: int = 101) -> List[np.ndarray]:
    """Normalize every channel independently to % cycle."""
    return [percent_cycle_normalize(np.array(ch)) for ch in channels if len(ch) > 2]

def shoelace_area(x: np.ndarray, y: np.ndarray) -> float:
    """Accurate polygon area for phase portraits (shoelace formula)."""
    x = safe_array(x)
    y = safe_array(y)
    if len(x) < 3:
        return 0.0
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def robust_cv(signal: np.ndarray) -> float:
    """Median-based robust coefficient of variation."""
    signal = safe_array(signal)
    if len(signal) < 2:
        return 0.0
    med = np.median(signal)
    mad = np.median(np.abs(signal - med))
    return mad / (np.abs(med) + 1e-9)

def theil_sen_trend(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Theil-Sen robust slope estimator (median of all pairwise slopes).
    Much more resistant to outliers than OLS LinearRegression.
    Returns (slope, intercept_approx).
    """
    x = safe_array(x)
    y = safe_array(y)
    n = len(x)
    if n < 3:
        return 0.0, float(np.mean(y)) if n else 0.0
    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            dx = x[j] - x[i]
            if abs(dx) > 1e-9:
                slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return 0.0, float(np.mean(y))
    slope = float(np.median(slopes))
    # Approximate intercept using median
    inter = float(np.median(y - slope * x))
    return slope, inter

def approximate_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Vectorized approximate entropy (ApEn) for signal complexity.
    Higher = more irregular (fatigue, poor control).
    """
    signal = safe_array(signal)
    n = len(signal)
    if n < 2 * m + 1:
        return 0.0
    r_val = r * (np.std(signal) + 1e-9)

    def _phi(mm):
        templates = np.array([signal[i:i+mm] for i in range(n - mm + 1)])
        if len(templates) == 0:
            return 1.0
        dists = np.max(np.abs(templates[:, None] - templates[None, :]), axis=2)
        count = np.sum(dists <= r_val)
        return count / (n - mm + 1) if (n - mm + 1) > 0 else 1.0

    return abs(np.log(_phi(m) / (_phi(m+1) + 1e-12) + 1e-12))


def sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    Sample Entropy (SampEn) - vectorized for speed.
    Lower values = more regularity (good motor control).
    Higher = complexity / unpredictability (fatigue, poor coordination).
    """
    signal = safe_array(signal)
    n = len(signal)
    if n < 10:
        return 0.0
    r_val = r * (np.std(signal) + 1e-9)

    def _count_matches_vec(mm):
        templates = np.array([signal[i:i+mm] for i in range(n - mm)])
        if len(templates) == 0:
            return 0
        dists = np.max(np.abs(templates[:, None] - templates[None, :]), axis=2)
        np.fill_diagonal(dists, np.inf)
        return int(np.sum(dists <= r_val))

    a = _count_matches_vec(m) + 1e-12
    b = _count_matches_vec(m + 1) + 1e-12
    return -np.log(b / a)


def jerk_smoothness_index(signal: np.ndarray, dt: float = 0.01) -> float:
    """
    High-precision smoothness via normalized mean squared jerk.
    Lower jerk = smoother movement (better motor control, less compensatory strategy).
    Returns a 0-1 smoothness score (1 = extremely smooth).
    """
    signal = safe_array(signal)
    if len(signal) < 5:
        return 0.5
    jerk = np.diff(signal, n=3) / (dt ** 3)
    msj = np.mean(jerk ** 2)
    # Normalize by amplitude and length
    amp = np.ptp(signal) + 1e-9
    norm_msj = msj / (amp ** 2 + 1e-9)
    smoothness = 1.0 / (1.0 + np.sqrt(norm_msj) * 100)
    return float(np.clip(smoothness, 0.0, 1.0))


def spectral_entropy(signal: np.ndarray, fs: float = 100.0) -> float:
    """
    Spectral entropy of the power spectrum (0-1 normalized).
    High = broadband noise-like (tremor, instability).
    Low = concentrated energy (smooth, periodic control).
    """
    signal = safe_array(signal)
    if len(signal) < 8:
        return 0.5
    # FFT power
    fft_vals = np.fft.rfft(signal)
    ps = np.abs(fft_vals) ** 2
    ps = ps / (np.sum(ps) + 1e-12)
    # Entropy
    se = -np.sum(ps * np.log(ps + 1e-12))
    se_norm = se / np.log(len(ps) + 1e-12)
    return float(np.clip(se_norm, 0.0, 1.0))


def compute_lyapunov_approx(signal: np.ndarray, max_lag: int = 8) -> float:
    """
    Very lightweight largest Lyapunov exponent approximation (Rosenstein style simplified).
    Positive = chaotic/unstable dynamics.
    Near zero or negative = stable attractor (good repeatable motor pattern).
    """
    signal = safe_array(signal)
    n = len(signal)
    if n < 20:
        return 0.0
    # Find nearest neighbors and divergence rate
    divergences = []
    for i in range(5, n - max_lag - 5):
        dists = np.abs(signal[i] - signal)
        nn = np.argmin(dists[:i-3] + 1e9)  # avoid trivial
        for lag in range(1, min(max_lag, n - i - 1)):
            d = abs(signal[i + lag] - signal[nn + lag])
            if d > 1e-8:
                divergences.append(np.log(d))
    if len(divergences) < 3:
        return 0.0
    return float(np.mean(divergences[-3:]) / max_lag)  # rough rate


def compute_multiscale_complexity(signal: np.ndarray, scales: int = 4) -> Dict:
    """
    Multiscale entropy-like measure (coarse graining + entropy at different scales).
    Captures complexity at different time scales (very useful for rehab).
    """
    signal = safe_array(signal)
    if len(signal) < 12:
        return {"ms_entropy": 0.0, "scale_complexity": [0.0]}
    results = []
    for tau in range(1, scales + 1):
        if tau == 1:
            coarse = signal
        else:
            coarse = np.array([np.mean(signal[i:i+tau]) for i in range(0, len(signal)-tau, tau)])
        results.append(approximate_entropy(coarse) + sample_entropy(coarse) * 0.5)
    return {
        "ms_entropy": float(np.mean(results)),
        "scale_complexity": [round(float(r), 4) for r in results],
        "complexity_trend": float(np.polyfit(range(len(results)), results, 1)[0]) if len(results) > 1 else 0.0
    }


# ============================================================
# ROUND 2: EVEN STRONGER PRECISION METRICS
# ============================================================

def detrended_fluctuation_analysis(signal: np.ndarray, max_scale: int = 8) -> float:
    """
    Full Detrended Fluctuation Analysis (DFA) alpha.
    alpha ~ 0.5 = white noise / uncorrelated
    alpha ~ 1.0 = pink noise / long-range correlations (good for skilled movement)
    alpha > 1.0 or < 0.5 often indicates pathology or fatigue.
    """
    signal = safe_array(signal)
    n = len(signal)
    if n < 20:
        return 0.5
    # Cumulative sum
    y = np.cumsum(signal - np.mean(signal))
    scales = np.unique(np.logspace(0.5, np.log10(n//4), num=max_scale).astype(int))
    scales = scales[scales >= 4]
    if len(scales) < 3:
        return 0.5
    fluct = []
    for s in scales:
        n_windows = n // s
        if n_windows < 1:
            continue
        rms = []
        for i in range(n_windows):
            seg = y[i*s:(i+1)*s]
            x = np.arange(s)
            p = np.polyfit(x, seg, 1)
            trend = np.polyval(p, x)
            rms.append(np.sqrt(np.mean((seg - trend)**2)))
        if rms:
            fluct.append(np.mean(rms))
    if len(fluct) < 3:
        return 0.5
    log_scales = np.log(scales[:len(fluct)])
    log_fluct = np.log(fluct)
    alpha = np.polyfit(log_scales, log_fluct, 1)[0]
    return float(np.clip(alpha, 0.1, 1.9))


def recurrence_quantification_lite(signal: np.ndarray, threshold: float = 0.1, max_points: int = 80) -> Dict:
    """
    Lightweight Recurrence Quantification Analysis (RQA).
    recurrence_rate: how often states repeat (high = stereotyped movement)
    determinism: % of recurrent points that form diagonals (high = predictable, deterministic dynamics)
    Very useful for motor control quality.
    """
    signal = safe_array(signal)
    n = min(len(signal), max_points)
    if n < 10:
        return {"recurrence_rate": 0.0, "determinism": 0.5}
    sig = signal[:n]
    # Distance matrix (normalized)
    diff = np.abs(sig[:, None] - sig[None, :])
    max_d = np.max(diff) + 1e-9
    rec_mat = (diff < (threshold * max_d)).astype(float)
    # Recurrence rate
    rec_rate = (np.sum(rec_mat) - n) / (n * n - n + 1e-9)
    # Determinism: count diagonal lines of length >=2
    diag_count = 0
    total_rec = 0
    for i in range(1, n):
        for j in range(1, n):
            if rec_mat[i, j]:
                total_rec += 1
                if rec_mat[i-1, j-1]:
                    diag_count += 1
    determinism = diag_count / (total_rec + 1e-9) if total_rec > 0 else 0.5
    return {
        "recurrence_rate": round(float(rec_rate), 4),
        "determinism": round(float(determinism), 3),
        "recurrence_quality": round(float(0.5 + 0.5 * (determinism - 0.3)), 3)
    }


def multi_lag_autocorrelation(signal: np.ndarray, max_lag: int = 6) -> Dict:
    """
    Summary of autocorrelation at multiple lags.
    High persistence at lag 1-3 = smooth, predictable control.
    Quick drop = noisy or jerky.
    """
    signal = safe_array(signal)
    n = len(signal)
    if n < 10:
        return {"mean_autocorr": 0.0, "lag1": 0.0}
    autoc = []
    mean = np.mean(signal)
    var = np.var(signal) + 1e-9
    for lag in range(1, min(max_lag + 1, n//2)):
        c = np.mean((signal[:-lag] - mean) * (signal[lag:] - mean)) / var
        autoc.append(c)
    return {
        "autocorr_lags": [round(float(x), 3) for x in autoc],
        "mean_autocorr": round(float(np.mean(autoc)), 3),
        "persistence": round(float(np.mean(autoc[:3])) if len(autoc) >= 3 else np.mean(autoc), 3)
    }


def channel_cross_correlation(angles: List[np.ndarray], forces: Optional[np.ndarray] = None, max_lag: int = 5) -> Dict:
    """
    Peak cross-correlation between angle channels and/or force.
    High correlation with small lag = good force-angle coupling.
    """
    if not angles or len(angles) < 2:
        return {"max_cross_corr": 0.0, "best_lag": 0}
    a1 = safe_array(angles[0])
    a2 = safe_array(angles[1]) if len(angles) > 1 else a1
    n = min(len(a1), len(a2))
    if n < 8:
        return {"max_cross_corr": 0.0, "best_lag": 0}
    a1 = a1[:n] - np.mean(a1[:n])
    a2 = a2[:n] - np.mean(a2[:n])
    corrs = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            c = np.corrcoef(a1[-lag:], a2[:lag])[0,1]
        elif lag > 0:
            c = np.corrcoef(a1[:-lag], a2[lag:])[0,1]
        else:
            c = np.corrcoef(a1, a2)[0,1]
        corrs.append((lag, c if not np.isnan(c) else 0))
    best = max(corrs, key=lambda x: abs(x[1]))
    return {
        "max_cross_corr": round(float(best[1]), 3),
        "best_lag": int(best[0]),
        "coupling_quality": round(float(0.5 + 0.5 * abs(best[1])), 3)
    }


def compute_intersession_decline(sessions: list, key: str = 'M') -> Dict:
    """
    Longitudinal fatigue / learning: how peak performance changes across sessions (chronological).
    Negative decline = improving or stable.
    Positive = worsening (fatigue or poor recovery).
    """
    peaks = []
    for s in sessions:
        m = s.get(key)
        if m is not None and len(m) > 3:
            try:
                peaks.append(float(np.max(np.asarray(m))))
            except:
                pass
    if len(peaks) < 2:
        return {"intersession_decline": 0.0, "trend": "insufficient"}
    # Linear trend on peaks (early to late)
    x = np.arange(len(peaks))
    slope, _ = theil_sen_trend(x, np.array(peaks))
    decline_pct = (peaks[-1] - peaks[0]) / (peaks[0] + 1e-9) * 100
    return {
        "intersession_decline_pct": round(decline_pct, 1),
        "peak_trend_slope": round(slope, 4),
        "fatigue_trend": "improving" if slope < -0.3 else "worsening" if slope > 0.8 else "stable"
    }


def icc_2_1(session_signals: list, normalize: bool = True) -> float:
    """
    Полноценный ICC(2,1) — Intraclass Correlation Coefficient для абсолютного согласия (two-way random effects).
    Используется для оценки надёжности/воспроизводимости паттерна движения между сессиями.
    Высокий ICC(2,1) (>0.75-0.9) = отличная воспроизводимость (хорошая моторная стабильность).
    Низкий = высокая вариабельность между попытками.
    session_signals: список 1D массивов (по одной на сессию).
    Нормализует к 0-100% и считает по точкам как subjects, сессиям как raters.
    """
    if len(session_signals) < 2:
        return 0.5
    sigs = []
    for s in session_signals:
        arr = safe_array(s)
        if len(arr) < 5:
            continue
        if normalize:
            arr = percent_cycle_normalize(arr)
        sigs.append(arr)
    if len(sigs) < 2:
        return 0.5
    # Make all same length
    target_len = min(len(s) for s in sigs)
    if target_len < 5:
        return 0.45
    data = np.stack([s[:target_len] for s in sigs]).T  # (n_subjects=points, n_raters=sessions)
    n, k = data.shape
    if n < 3 or k < 2:
        return 0.5
    # Compute means
    grand_mean = np.mean(data)
    subj_means = np.mean(data, axis=1)
    rater_means = np.mean(data, axis=0)
    # Sum of squares
    ssb = k * np.sum((subj_means - grand_mean)**2)   # between subjects
    ssr = n * np.sum((rater_means - grand_mean)**2)   # between raters
    sse = np.sum((data - subj_means[:, None] - rater_means[None, :] + grand_mean)**2)  # error
    # Mean squares
    msb = ssb / (n - 1) if n > 1 else 0
    msr = ssr / (k - 1) if k > 1 else 0
    mse = sse / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0
    # ICC(2,1) formula (Shrout & Fleiss)
    icc = (msb - mse) / (msb + (k - 1) * mse + k * (msr - mse) / n)
    return float(np.clip(icc, 0.0, 1.0))


def bootstrap_ci(data: np.ndarray, func=np.mean, n_boot: int = 200, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Simple bootstrap confidence interval for a statistic."""
    data = safe_array(data)
    if len(data) < 5:
        m = func(data) if len(data) else 0.0
        return m, m, m
    boots = []
    n = len(data)
    rng = np.random.default_rng(42)  # reproducible for determinism
    for _ in range(n_boot):
        samp = rng.choice(data, size=n, replace=True)
        boots.append(func(samp))
    boots = np.array(boots)
    lo = np.percentile(boots, alpha * 100 / 2)
    hi = np.percentile(boots, 100 - alpha * 100 / 2)
    return float(func(data)), float(lo), float(hi)

def _trapz(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    """NumPy 2+ compatible trapezoidal integration (np.trapz removed)."""
    y = np.asarray(y, dtype=float)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x, dtype=float)
    n = min(len(y), len(x))
    if n < 2:
        return 0.0
    return float(np.trapezoid(y[:n], x[:n]))

def extract_best_angle_series(session: dict) -> np.ndarray:
    """Heuristic: pick the channel with largest ROM, or mean of all."""
    angles = session.get('angles', []) or session.get('angles_by_channel', [])
    alen = len(angles) if hasattr(angles, '__len__') else 0
    if alen == 0:
        return np.array([])
    # If it's a flat array or something, treat as single channel
    if not isinstance(angles, (list, tuple)) or (angles and not isinstance(angles[0], (list, tuple, np.ndarray))):
        return safe_array(angles)
    roms = [float(np.ptp(ch)) for ch in angles if hasattr(ch, '__len__') and len(ch) > 1]
    if not roms:
        return np.array([])
    best_idx = int(np.argmax(roms))
    return safe_array(angles[best_idx])

def extract_moment_or_force_proxy(session: dict) -> np.ndarray:
    """Prefer explicit 'M' (leg load moment), fallback to summed forces or computed."""
    m = session.get('M')
    if m is not None:
        try:
            if hasattr(m, '__len__') and len(m) > 0:
                return safe_array(m)
        except Exception:
            pass
    forces = session.get('forces', [])
    flen = len(forces) if hasattr(forces, '__len__') else 0
    if flen > 0:
        total_f = np.array([sum(row) for row in forces])
        angles = session.get('angles', [])
        alen = len(angles) if hasattr(angles, '__len__') else 0
        if alen > 0:
            avg_a = np.array([np.mean(row) for row in angles])
            lever = 0.42 * (1 + 0.25 * np.sin(np.radians(avg_a)))
            return total_f * lever
        return total_f
    return np.array([])

def get_exercise_type(name: str) -> str:
    n = (name or "").upper()
    if "БЕДРА" in n or "HIP" in n:
        return "hip_rotation"
    if "ГОЛЕНИ" in n or "SHANK" in n or "KNEE" in n:
        return "knee_rotation"
    if "СТОПЫ" in n or "ANKLE" in n or "FOOT" in n:
        return "ankle_rotation"
    if "ХОДЬБ" in n or "WALK" in n:
        return "walking"
    return "rotation"

def get_age_group(age: Optional[float]) -> str:
    if age is None:
        return "adult"
    if age < 12:
        return "child"
    if age < 18:
        return "adolescent"
    if age < 65:
        return "adult"
    return "elderly"


# ============================================================
# ADVANCED ACCURACY FUNCTIONS (for even higher precision)
# ============================================================

def compute_analytic_phase(signal: np.ndarray) -> np.ndarray:
    """
    Analytic phase using Hilbert-like transform via FFT (more accurate CRP).
    Returns instantaneous phase in radians.
    """
    s = safe_array(signal)
    n = len(s)
    if n < 4:
        return np.zeros(n)
    # Simple Hilbert approximation via FFT (real part + imag via quadrature)
    fft = np.fft.fft(s)
    h = np.zeros(n)
    h[0] = 1
    h[1:(n+1)//2] = 2
    if n % 2 == 0:
        h[n//2] = 1
    analytic = np.fft.ifft(fft * h)
    phase = np.angle(analytic)
    return phase

def continuous_relative_phase(sig1: np.ndarray, sig2: np.ndarray, normalize: bool = True) -> Dict:
    """
    High-accuracy Continuous Relative Phase (CRP).
    Uses analytic phase (Hilbert proxy) for better timing of coordination.
    Returns mean |CRP|, variability (sd), and a quality score (lower var + moderate mean phase = better).
    """
    p1 = compute_analytic_phase(sig1)
    p2 = compute_analytic_phase(sig2)
    n = min(len(p1), len(p2))
    if n < 5:
        return {"mean_crp_deg": 0.0, "crp_variability": 0.0, "coordination_quality": 0.5}

    crp = np.abs((p1[:n] - p2[:n] + np.pi) % (2 * np.pi) - np.pi)  # wrapped to [-pi, pi]
    mean_crp = float(np.mean(crp))
    var_crp = float(np.std(crp))
    # Quality: penalize very high or very low mean phase + high variability (poor coupling)
    q = 1.0 - min(1.0, (mean_crp / np.pi * 0.6 + var_crp / (np.pi * 0.7)))
    q = max(0.15, min(0.98, q))
    return {
        "mean_crp_deg": round(np.degrees(mean_crp), 2),
        "crp_variability": round(var_crp, 4),
        "coordination_quality": round(q, 3),
        "n_points": n
    }

def vector_coding_coupling(angle1: np.ndarray, angle2: np.ndarray, n_bins: int = 8) -> Dict:
    """
    Vector coding / coupling angle analysis (Chang, Hamill, etc.).
    Computes coupling angles between two joints over the cycle.
    Returns mean coupling angle (deg), variability, and distribution.
    High accuracy for inter-joint coordination patterns (in-phase, anti-phase, etc.).
    """
    a1 = percent_cycle_normalize(safe_array(angle1))
    a2 = percent_cycle_normalize(safe_array(angle2))
    n = min(len(a1), len(a2))
    a1, a2 = a1[:n], a2[:n]

    # Coupling angle = atan2( delta joint2 , delta joint1 )
    da1 = np.diff(np.concatenate([[a1[0]], a1]))
    da2 = np.diff(np.concatenate([[a2[0]], a2]))
    coup = np.arctan2(da2, da1)
    coup_deg = np.degrees(coup) % 360

    mean_coup = float(np.mean(coup_deg))
    var_coup = float(np.std(coup_deg))

    # Simple bin distribution for pattern classification
    bins = np.linspace(0, 360, n_bins + 1)
    hist, _ = np.histogram(coup_deg, bins=bins)
    hist = hist / max(1, hist.sum())

    # Classify dominant pattern
    dominant = "variable"
    if np.max(hist) > 0.35:
        peak_bin = np.argmax(hist)
        center = (bins[peak_bin] + bins[peak_bin+1]) / 2
        if center < 45 or center > 315:
            dominant = "in-phase"
        elif 135 < center < 225:
            dominant = "anti-phase"
        else:
            dominant = "out-of-phase"

    return {
        "mean_coupling_deg": round(mean_coup, 1),
        "coupling_variability": round(var_coup, 1),
        "dominant_pattern": dominant,
        "pattern_confidence": round(float(np.max(hist)), 3)
    }


# ============================================================
# IMPROVED KINEMATICS + KINEMATIC-DYNAMICS (stronger coordination and link to dynamics)
# ============================================================

def enhanced_continuous_relative_phase(sig1: np.ndarray, sig2: np.ndarray) -> Dict:
    """
    Enhanced CRP with additional features for better kinematic analysis:
    - Mean absolute relative phase
    - Circular variance
    - Coordination stability (low var = stable timing)
    """
    base = continuous_relative_phase(sig1, sig2)
    p1 = compute_analytic_phase(safe_array(sig1))
    p2 = compute_analytic_phase(safe_array(sig2))
    n = min(len(p1), len(p2))
    if n < 5:
        return base
    crp = np.abs((p1[:n] - p2[:n] + np.pi) % (2 * np.pi) - np.pi)
    mean_arp = float(np.mean(np.abs(crp)))  # mean absolute relative phase
    # Circular stats for phase
    circ_var = float(1 - np.abs(np.mean(np.exp(1j * crp))))
    stability = 1.0 - min(1.0, base.get("crp_variability", 0) / (np.pi * 0.5))
    base.update({
        "mean_absolute_relative_phase_deg": round(np.degrees(mean_arp), 1),
        "circular_variance": round(circ_var, 3),
        "coordination_stability": round(max(0.1, stability), 3)
    })
    return base


def discrete_relative_phase(angle1: np.ndarray, angle2: np.ndarray, event: str = "max") -> float:
    """
    Discrete Relative Phase (DRP) at key events (max or min of reference joint).
    Common in clinical coordination analysis (e.g., max knee flexion timing relative to hip).
    Returns phase difference in % of cycle.
    """
    a1 = percent_cycle_normalize(safe_array(angle1))
    a2 = percent_cycle_normalize(safe_array(angle2))
    n = min(len(a1), len(a2))
    if n < 10:
        return 0.0
    if event == "max":
        idx = int(np.argmax(a1))
    else:
        idx = int(np.argmin(a1))
    # Phase of second joint at that point
    phase = (a2[idx] - a1[idx]) % 100   # rough % cycle diff proxy
    return float(phase)


def kinematic_dynamics_coupling(angles: np.ndarray, moments: np.ndarray) -> Dict:
    """
    Kinematic-Dynamic coupling: how well angle/velocity relates to moment (inverse dynamics output).
    High correlation at appropriate phase = efficient movement.
    Used to assess if kinematics drive proper loading.
    """
    ang = safe_array(angles)
    mom = safe_array(moments)
    n = min(len(ang), len(mom))
    if n < 8:
        return {"angle_moment_corr": 0.0, "vel_moment_corr": 0.0, "coupling_quality": 0.5}
    ang = ang[:n]
    mom = mom[:n]
    vel = central_difference(ang)
    c1 = float(np.corrcoef(ang, mom)[0,1]) if np.std(mom) > 1e-9 else 0.0
    c2 = float(np.corrcoef(vel, mom)[0,1]) if np.std(mom) > 1e-9 else 0.0
    q = max(0.1, min(0.98, (abs(c1) + abs(c2)) / 2 ))
    return {
        "angle_moment_corr": round(c1, 3),
        "velocity_moment_corr": round(c2, 3),
        "coupling_quality": round(q, 3)
    }


def angular_kinematics_profile(angle: np.ndarray, dt: float = 0.01) -> Dict:
    """
    Rich profile of angular kinematics: peak vel, acc, jerk, timing.
    Helps assess speed of movement and control.
    """
    a = safe_array(angle)
    if len(a) < 5:
        return {"peak_velocity": 0, "peak_acc": 0}
    vel = central_difference(a, dt)
    acc = second_derivative(a, dt)
    jerk = np.diff(acc) / dt if len(acc) > 1 else np.array([0])
    return {
        "peak_angular_velocity": round(float(np.max(np.abs(vel))), 1),
        "peak_angular_acceleration": round(float(np.max(np.abs(acc))), 1),
        "peak_jerk": round(float(np.max(np.abs(jerk))), 1),
        "mean_velocity": round(float(np.mean(np.abs(vel))), 1),
        "velocity_cv": round(float(np.std(vel) / (np.mean(np.abs(vel)) + 1e-9)), 3)
    }


def detrended_fluctuation_analysis(signal: np.ndarray, scales: Optional[List[int]] = None) -> float:
    """
    Lightweight DFA (detrended fluctuation analysis) for long-range correlations.
    Returns scaling exponent alpha ( ~0.5 = uncorrelated noise, >0.7 = persistent, <0.5 anti-persistent).
    Excellent for detecting changes in motor control / fatigue.
    """
    s = safe_array(signal)
    n = len(s)
    if n < 20:
        return 0.5
    if scales is None:
        scales = [max(4, n // 16), max(8, n // 8), max(16, n // 4)]

    def _dfa_for_scale(scale):
        # Integrate
        y = np.cumsum(s - np.mean(s))
        # Divide into boxes
        n_boxes = n // scale
        if n_boxes < 2:
            return 0.0
        flucts = []
        for i in range(n_boxes):
            seg = y[i*scale:(i+1)*scale]
            x = np.arange(len(seg))
            # Linear detrend
            if len(seg) > 1:
                p = np.polyfit(x, seg, 1)
                trend = np.polyval(p, x)
                flucts.append(np.sqrt(np.mean((seg - trend)**2)))
        if not flucts:
            return 0.0
        return np.mean(flucts)

    log_scales = []
    log_flucts = []
    for sc in scales:
        if sc >= n // 2:
            continue
        f = _dfa_for_scale(sc)
        if f > 0:
            log_scales.append(np.log(sc))
            log_flucts.append(np.log(f))

    if len(log_scales) < 2:
        return 0.5
    slope, _ = np.polyfit(log_scales, log_flucts, 1)
    alpha = float(np.clip(slope, 0.1, 1.5))
    return round(alpha, 3)

def cycle_quality_score(cycle_signals: List[np.ndarray]) -> float:
    """
    Data-driven cycle quality / reliability score (0-1).
    Considers smoothness, amplitude consistency, noise level.
    Used to weight sessions in multi-session aggregation.
    """
    if not cycle_signals:
        return 0.3
    scores = []
    for sig in cycle_signals:
        s = safe_array(sig)
        if len(s) < 5:
            continue
        # Smoothness (low jerk proxy)
        jerk = np.mean(np.abs(np.diff(s, 2))) + 1e-9
        smooth = 1.0 / (1.0 + jerk / (np.std(s) + 1e-6))
        # Amplitude presence (not flat)
        amp = (np.ptp(s)) / (np.std(s) + 1e-6) if np.std(s) > 0 else 0
        amp_score = min(1.0, amp / 4.0)
        scores.append(0.6 * smooth + 0.4 * amp_score)
    return float(np.clip(np.mean(scores), 0.1, 0.98)) if scores else 0.4

def monte_carlo_perturb(data: np.ndarray, n: int = 40, noise_level: float = 0.03) -> Dict:
    """
    Monte-Carlo style perturbation for uncertainty (stronger than simple param variation).
    Adds realistic sensor noise and re-computes statistic.
    """
    d = safe_array(data)
    if len(d) < 3:
        m = float(np.mean(d)) if len(d) else 0.0
        return {"mean": m, "std": 0.0, "ci95": (m, m)}
    stats = []
    rng = np.random.default_rng(123)
    base_std = np.std(d) * noise_level
    for _ in range(n):
        noisy = d + rng.normal(0, max(1e-6, base_std), size=len(d))
        stats.append(np.mean(noisy))
    arr = np.array(stats)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci95": (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    }

def weighted_multi_session_aggregate(per_session_metrics: List[Dict], quality_scores: Optional[List[float]] = None) -> Dict:
    """
    Quality-weighted aggregation across sessions (much better than simple mean).
    """
    if not per_session_metrics:
        return {}
    if quality_scores is None or len(quality_scores) != len(per_session_metrics):
        quality_scores = [1.0] * len(per_session_metrics)

    total_w = sum(max(0.05, q) for q in quality_scores) or 1.0
    result = {}
    for key in per_session_metrics[0].keys():
        if isinstance(per_session_metrics[0][key], (int, float)):
            vals = [float(m.get(key, 0)) for m in per_session_metrics]
            wsum = sum(v * max(0.05, q) for v, q in zip(vals, quality_scores))
            result[key] = round(wsum / total_w, 4)
    return result

# ============================================================
# EVEN HIGHER PRECISION HELPERS (for patient data, calculations, analysis)
# ============================================================

def safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    """Numerically stable division."""
    d = float(denom)
    return float(numer) / d if abs(d) > 1e-12 else default

def precise_trapz(y: np.ndarray, x: Optional[np.ndarray] = None, units: str = "") -> float:
    """High-precision trapezoidal integration with optional unit note."""
    val = _trapz(y, x)
    return float(val)

def anthropometric_adjusted_masses(weight_kg: float, height_cm: float) -> Dict[str, float]:
    """
    Более точная антропометрия: используем регрессионные формулы на основе роста + веса
    (на основе данных Dempster, Clauser, Winter, adjusted для индивидуальных пропорций).
    Возвращает массы сегментов в кг + длины + COM + моменты инерции.
    """
    # Базовые пропорции (Winter / Dempster)
    base_mass_frac = {'thigh': 0.100, 'shank': 0.0465, 'foot': 0.0145}
    base_len_frac  = {'thigh': 0.245, 'shank': 0.246, 'foot': 0.152}  # от роста
    base_com_frac  = {'thigh': 0.433, 'shank': 0.433, 'foot': 0.50}

    std_h = 170.0
    std_w = 70.0

    # Масштабирование: линейно по росту для длин, суб-линейно по весу для масс
    h_scale = height_cm / std_h
    w_scale = (weight_kg / std_w) ** 0.9   # чуть ближе к линейному для реализма

    masses = {}
    lengths = {}
    com_fracs = {}
    inertias = {}

    for seg in ['thigh', 'shank', 'foot']:
        # Масса сегмента (кг)
        m = base_mass_frac[seg] * weight_kg * (0.95 + 0.1 * w_scale)  # небольшой весовой корректировщик
        masses[seg] = round(m, 3)

        # Длина сегмента (м) — сильно зависит от роста
        L = base_len_frac[seg] * height_cm / 100.0 * (0.98 + 0.04 * h_scale)
        lengths[seg] = round(L, 3)

        # COM fraction (от проксимального конца)
        com_fracs[seg] = base_com_frac[seg]

        # Приближенный момент инерции (kg·m²) — параллельная ось
        I = m * (L * 0.3) ** 2   # грубая, но улучшенная оценка
        inertias[seg] = round(I, 5)

    return {
        "masses_kg": masses,
        "lengths_m": lengths,
        "com_fractions": com_fracs,
        "inertias_kgm2": inertias,
        "total_leg_length_m": round(lengths['thigh'] + lengths['shank'] + lengths['foot'], 3),
        "normalization_factor": round((weight_kg * (lengths['thigh'] + lengths['shank'] + lengths['foot'])) , 2)
    }

def savitzky_golay_lite(y: np.ndarray, window: int = 5, poly: int = 2) -> np.ndarray:
    """
    Lightweight Savitzky-Golay smoothing (pure numpy, small windows).
    Improves derivative stability for noisy patient sensor data.
    """
    y = safe_array(y)
    n = len(y)
    if n < window or window < 3:
        return y
    half = window // 2
    # Simple convolution with quadratic fit weights (precomputed for common small windows)
    if window == 5 and poly == 2:
        coeffs = np.array([-3, 12, 17, 12, -3]) / 35.0
    elif window == 7 and poly == 2:
        coeffs = np.array([-2, 3, 6, 7, 6, 3, -2]) / 21.0
    else:
        # Fallback to moving average
        coeffs = np.ones(window) / window
    pad = np.pad(y, (half, half), mode='edge')
    out = np.convolve(pad, coeffs, mode='valid')
    return safe_array(out[:n])

def derive_joint_angles(channels: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Compute anatomically meaningful joint angles from raw sensor channels.
    Assumes channels ~ [thigh/proximal, shank, foot/distal].
    knee_angle ≈ shank - thigh (sagittal convention).
    """
    if not channels or len(channels) == 0:
        return {"thigh": np.array([]), "knee": np.array([]), "ankle": np.array([])}
    th = safe_array(channels[0])
    sh = safe_array(channels[1]) if len(channels) > 1 else th * 0.0
    ft = safe_array(channels[2]) if len(channels) > 2 else sh * 0.0

    # Simple but effective kinematic reconstruction
    thigh = th
    knee = sh - th          # relative shank to thigh
    ankle = ft - sh         # relative foot to shank
    return {
        "thigh": thigh,
        "knee": knee,
        "ankle": ankle,
        "knee_rom": float(np.ptp(knee)) if len(knee) > 0 else 0.0
    }

def robust_thirds_fatigue(signal: np.ndarray, n_thirds: int = 3) -> Dict:
    """
    More accurate intra-cycle fatigue: splits, peak decay + trend within thirds + poly fit.
    """
    s = safe_array(signal)
    if len(s) < n_thirds * 4:
        return {"fatigue_percent": 0.0, "per_third_peaks": [], "trend_slope": 0.0, "interpretation": "INSUFFICIENT"}
    thirds = np.array_split(s, n_thirds)
    peaks = [float(np.max(t)) for t in thirds]
    fatigue = safe_div(peaks[0] - peaks[-1], peaks[0] + 1e-9) * 100.0

    # Linear trend across all points (better than just peaks)
    x = np.arange(len(s))
    slope, _ = theil_sen_trend(x, s)
    interp = "LOW_FATIGUE"
    if fatigue > 22 or abs(slope) > 0.008:
        interp = "HIGH_FATIGUE"
    elif fatigue > 12:
        interp = "MODERATE_FATIGUE"
    return {
        "fatigue_percent": round(fatigue, 2),
        "per_third_peaks": [round(p, 2) for p in peaks],
        "trend_slope": round(slope, 6),
        "interpretation": interp
    }

def precise_pearson_normalized(sigs: List[np.ndarray], n_points: int = 101) -> float:
    """Exact Pearson after strict 0-100% normalization (matches DOCX-style analysis)."""
    normed = [percent_cycle_normalize(safe_array(s), n_points) for s in sigs if len(s) > 5]
    normed = [(s - np.mean(s)) / (np.std(s) + 1e-9) for s in normed if np.std(s) > 1e-9]
    if len(normed) < 2:
        return 0.0
    corrs = []
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            c = np.corrcoef(normed[i], normed[j])[0, 1]
            if not np.isnan(c):
                corrs.append(abs(c))
    return round(float(np.mean(corrs)), 4) if corrs else 0.0

def patient_data_quality(patient_info: dict, sessions: list) -> Dict:
    """Overall patient + data quality score for uncertainty weighting."""
    issues = []
    score = 1.0
    w = float(patient_info.get('weight_kg', 0))
    if w < 30 or w > 180:
        issues.append("weight_out_of_range"); score *= 0.85
    links = [float(patient_info.get(k, 0)) for k in ('upper_link_cm', 'middle_link_cm', 'lower_link_cm')]
    if any(l < 15 or l > 55 for l in links):
        issues.append("link_lengths_suspicious"); score *= 0.9
    if not patient_info.get('age_years') and not patient_info.get('birth_date'):
        issues.append("no_age"); score *= 0.95

    sess_qs = []
    for s in sessions:
        m = s.get('M')
        if m is None or (hasattr(m, '__len__') and len(m) == 0):
            m = s.get('forces') or []
        m_len = len(m) if hasattr(m, '__len__') else 0
        if m_len > 0:
            # Support both list-of-rows and ndarray / flat list for M
            if isinstance(m, (list, tuple)) and m and isinstance(m[0], (list, tuple)):
                mm = [sum(r) for r in m]
            else:
                mm = np.asarray(m).ravel().tolist()
            q = cycle_quality_score([np.array(mm)])
            sess_qs.append(q)
    avg_sess_q = float(np.mean(sess_qs)) if sess_qs else 0.6
    final = round(max(0.3, min(0.98, score * (0.4 + 0.6 * avg_sess_q))), 3)
    return {"overall": final, "issues": issues, "per_session_quality": [round(q, 3) for q in sess_qs]}

def compute_session_deltas(session_metrics: List[Dict], key_metrics: List[str] = None) -> Dict:
    """
    Compare consecutive sessions for the same exercise.
    Returns deltas (absolute and %) for key metrics between sessions.
    """
    if key_metrics is None:
        key_metrics = ['rel_force_pct_bw', 'coordination_quality', 'mean_cv', 'mean_fatigue_percent']
    if len(session_metrics) < 2:
        return {"deltas": [], "note": "insufficient sessions for comparison"}

    deltas = []
    for i in range(1, len(session_metrics)):
        prev = session_metrics[i-1]
        curr = session_metrics[i]
        d = {"from_session": i-1, "to_session": i}
        for k in key_metrics:
            p = prev.get(k, prev.get('aggregate', {}).get(k, 0))
            c = curr.get(k, curr.get('aggregate', {}).get(k, 0))
            if p and c:
                abs_d = c - p
                pct = (abs_d / abs(p)) * 100 if p != 0 else 0
                d[k] = {"abs": round(abs_d, 3), "pct": round(pct, 1), "direction": "improved" if abs_d < 0 and 'cv' in k or abs_d > 0 and 'force' in k or abs_d > 0 and 'coord' in k else "worsened" if abs_d != 0 else "stable"}
        deltas.append(d)
    return {"deltas": deltas, "num_comparisons": len(deltas)}

def compute_date_progress(sessions: list, key_metrics: List[str] = None) -> Dict:
    """
    Compare data across dates to track rehabilitation progress.
    Expects sessions to have 'date' (str 'YYYY-MM-DD' or similar) and metrics.
    Returns overall trend, % change from first to last, and per-metric progress.
    """
    if key_metrics is None:
        key_metrics = ['fused_rel_load', 'fused_coordination', 'fused_cv', 'fused_variability']
    dated = [s for s in sessions if s.get('date')]
    if len(dated) < 2:
        return {"progress": "insufficient dated sessions", "num_dates": len(dated)}

    # Sort by date
    try:
        dated.sort(key=lambda s: s['date'])
    except:
        dated.sort(key=lambda s: str(s.get('date', '')))

    first = dated[0]
    last = dated[-1]
    progress = {"first_date": first.get('date'), "last_date": last.get('date'), "num_sessions": len(dated)}
    per_metric = {}
    overall_improvement = 0
    count = 0

    for k in key_metrics:
        f = first.get(k, first.get('aggregate', {}).get(k) or first.get('fused_metrics', {}).get(k))
        l = last.get(k, last.get('aggregate', {}).get(k) or last.get('fused_metrics', {}).get(k))
        if f is not None and l is not None and f != 0:
            change = l - f
            pct = (change / abs(f)) * 100
            direction = "improving" if (k in ['fused_coordination'] and change > 0) or (k in ['fused_rel_load', 'fused_cv', 'fused_variability'] and change < 0) else "worsening" if change != 0 else "stable"
            per_metric[k] = {"first": round(f, 2), "last": round(l, 2), "change_pct": round(pct, 1), "direction": direction}
            overall_improvement += pct if direction == "improving" else -abs(pct) if direction == "worsening" else 0
            count += 1

    if count > 0:
        progress["overall_progress_pct"] = round(overall_improvement / count, 1)
        progress["trend"] = "improving" if overall_improvement > 5 else "worsening" if overall_improvement < -5 else "stable"
        progress["per_metric"] = per_metric
        progress["days_tracked"] = "multiple dates"  # could parse dates for exact days
    else:
        progress["trend"] = "insufficient metrics"

    return progress

def compute_fft_power(signal: np.ndarray, sample_rate: float = 1.0) -> Dict:
    """
    New analysis: Frequency domain analysis for movement periodicity and smoothness.
    Returns dominant frequency, power in low/high bands (for tremor vs smooth control).
    Useful for coordination and fatigue (high freq = tremor/fatigue).
    """
    sig = safe_array(signal)
    n = len(sig)
    if n < 8:
        return {"dominant_freq": 0.0, "low_power_ratio": 0.5, "note": "short signal"}
    # FFT
    fft_vals = np.fft.fft(sig)
    freqs = np.fft.fftfreq(n, d=1.0/sample_rate)
    power = np.abs(fft_vals)**2
    # Positive freqs
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_power = power[pos_mask]
    if len(pos_power) == 0:
        return {"dominant_freq": 0.0, "low_power_ratio": 0.5}
    total_power = np.sum(pos_power)
    low_mask = pos_freqs < 0.1  # low freq < 0.1 Hz (smooth cycles)
    low_power = np.sum(pos_power[low_mask]) if np.any(low_mask) else 0
    low_ratio = low_power / total_power if total_power > 0 else 0.5
    dom_idx = np.argmax(pos_power)
    dom_freq = pos_freqs[dom_idx]
    return {
        "dominant_freq_hz": round(float(dom_freq), 3),
        "low_freq_power_ratio": round(float(low_ratio), 3),
        "high_freq_power_ratio": round(1.0 - float(low_ratio), 3)
    }

def compute_complexity_metrics(signal: np.ndarray) -> Dict:
    """
    STRONGER: Advanced multi-metric complexity analysis.
    Now includes: SampEn (more robust), Multiscale, Jerk-smoothness, Spectral entropy, Lyapunov approx.
    This significantly increases the precision of variability/coordination agents.
    """
    sig = safe_array(signal)
    n = len(sig)
    if n < 10:
        return {"approx_entropy": 0.0, "sample_entropy": 0.0, "ms_entropy": 0.0, "smoothness": 0.5, "note": "short"}

    apen = approximate_entropy(sig)
    sen = sample_entropy(sig)
    ms = compute_multiscale_complexity(sig)
    smooth = jerk_smoothness_index(sig)
    spec_ent = spectral_entropy(sig)
    lyap = compute_lyapunov_approx(sig)

    complexity_score = float(np.clip((apen * 0.3 + sen * 0.35 + (1 - smooth) * 0.2 + spec_ent * 0.15), 0, 1))

    return {
        "approx_entropy": round(float(apen), 3),
        "sample_entropy": round(float(sen), 3),
        "multiscale_entropy": round(ms["ms_entropy"], 3),
        "hurst_exponent_proxy": round(float(detrended_fluctuation_analysis(sig) if n > 15 else 0.5), 3),
        "jerk_smoothness": round(smooth, 3),
        "spectral_entropy": round(spec_ent, 3),
        "lyapunov_approx": round(lyap, 4),
        "overall_complexity_score": round(complexity_score, 3),
        "complexity_level": "high" if complexity_score > 0.72 else "medium" if complexity_score > 0.38 else "low"
    }

def compute_asymmetry_evolution(sessions: list, channel_idx: int = 0) -> Dict:
    """
    STRONGER asymmetry evolution tracking.
    Uses multiple asymmetry metrics (peak SI, area-based, CV difference) + trend + statistical significance proxy.
    Much more sensitive for rehabilitation progress detection.
    """
    if len(sessions) < 2:
        return {"asym_trend": "insufficient data"}
    asyms = []
    area_asyms = []
    cv_diffs = []
    for s in sessions:
        angles = s.get('angles', []) or s.get('angles_by_channel', [])
        m = s.get('M')
        if len(angles) > channel_idx + 1:
            left = safe_array(angles[channel_idx])
            right = safe_array(angles[channel_idx + 1]) if len(angles) > channel_idx + 1 else left
            if len(left) > 3 and len(right) > 3:
                # Classic symmetry index on peaks
                si = 200 * abs(np.max(left) - np.max(right)) / (np.max(left) + np.max(right) + 1e-9)
                asyms.append(si)
                # Area / impulse asymmetry (better for load)
                area_l = _trapz(left)
                area_r = _trapz(right)
                area_si = 200 * abs(area_l - area_r) / (abs(area_l) + abs(area_r) + 1e-9)
                area_asyms.append(area_si)
                # Variability difference
                cv_l = np.std(left) / (np.mean(np.abs(left)) + 1e-9)
                cv_r = np.std(right) / (np.mean(np.abs(right)) + 1e-9)
                cv_diffs.append(abs(cv_l - cv_r))
    if len(asyms) < 2:
        return {"asym_trend": "insufficient"}
    # Multi-metric trend
    slope_peak, _ = theil_sen_trend(np.arange(len(asyms)), np.array(asyms))
    slope_area = theil_sen_trend(np.arange(len(area_asyms)), np.array(area_asyms))[0] if area_asyms else 0
    # Composite asymmetry score
    composite = [0.5*a + 0.3*b + 0.2*c for a,b,c in zip(asyms, area_asyms or asyms, cv_diffs or [0]*len(asyms))]
    slope_comp, _ = theil_sen_trend(np.arange(len(composite)), np.array(composite))
    change = (asyms[-1] - asyms[0]) / (asyms[0] + 1e-9) * 100 if asyms[0] > 0 else 0
    return {
        "asymmetry_values": [round(a, 1) for a in asyms],
        "area_asymmetry": [round(a, 1) for a in area_asyms],
        "asym_trend_slope": round(slope_peak, 4),
        "area_trend_slope": round(slope_area, 4),
        "composite_trend_slope": round(slope_comp, 4),
        "overall_asym_change_pct": round(change, 1),
        "improving": bool(slope_comp < -0.5)  # negative slope = asymmetry decreasing
    }

def multi_signal_cycle_detector(angle_sig: np.ndarray, force_sig: Optional[np.ndarray] = None, min_len: int = 12) -> List[Tuple[int, int]]:
    """
    Higher accuracy cycle detection: prefers force peaks/valleys when available (more reliable for loading cycles),
    falls back to angle extrema + velocity zero-crossings. Returns list of (start, end) indices.
    """
    a = safe_array(angle_sig)
    n = len(a)
    if n < min_len * 2:
        return [(0, n-1)]

    # Velocity for angle
    va = central_difference(a)
    candidates = []
    for i in range(1, n-1):
        # Angle local ext
        if (a[i] > a[i-1] and a[i] > a[i+1]) or (a[i] < a[i-1] and a[i] < a[i+1]):
            candidates.append(i)
        # Velocity sign change (phase reversal)
        if va[i-1] * va[i] < 0:
            candidates.append(i)

    if force_sig is not None and len(force_sig) > 10:
        f = safe_array(force_sig)[:n]
        # Force local max (typically loading peaks)
        for i in range(1, n-1):
            if f[i] > f[i-1] and f[i] > f[i+1] and f[i] > np.mean(f) * 0.6:
                candidates.append(i)

    candidates = sorted(set(candidates))
    cycles = []
    for i in range(len(candidates) - 1):
        st, en = candidates[i], candidates[i+1]
        if en - st >= min_len:
            cycles.append((st, en))

    if not cycles:
        cycles = [(0, n-1)]
    return cycles

def compute_session_reliability(sessions: list, key: str = 'M') -> float:
    """
    Simple but effective inter-session reliability (proxy for ICC(2,1) style).
    Higher = more consistent patient performance across trials (good for normative trust).
    """
    vals = []
    for s in sessions:
        data = s.get(key)
        if data is None or (hasattr(data, '__len__') and len(data) == 0):
            data = []
        dlen = len(data) if hasattr(data, '__len__') else 0
        if dlen > 0:
            if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
                data = [sum(r) for r in data]
            else:
                data = np.asarray(data).ravel().tolist()
            v = float(np.std(data) / (np.mean(np.abs(data)) + 1e-9)) if len(data) > 2 else 0.5
            vals.append(1.0 - min(1.0, v))
    if len(vals) < 2:
        return 0.65
    return round(float(np.mean(vals)), 3)


# Export new stronger functions (Round 2)
__all__ = [
    'safe_array', 'central_difference', 'sample_entropy', 'jerk_smoothness_index',
    'spectral_entropy', 'compute_multiscale_complexity', 'compute_lyapunov_approx',
    'compute_fft_power', 'compute_complexity_metrics', 'compute_asymmetry_evolution',
    'continuous_relative_phase', 'patient_data_quality', 'compute_session_reliability',
    'detrended_fluctuation_analysis', 'recurrence_quantification_lite',
    'multi_lag_autocorrelation', 'channel_cross_correlation', 'compute_intersession_decline'
]

def estimate_overall_uncertainty(agent_outputs: Dict[str, Dict]) -> Dict:
    """
    Aggregates uncertainty signals from agents into overall ensemble uncertainty.
    """
    confs = []
    for name, out in agent_outputs.items():
        c = out.get('enhanced_confidence') or out.get('confidence') or 0.7
        if isinstance(c, (int, float)):
            confs.append(float(c))
    if not confs:
        return {"ensemble_uncertainty": 0.35, "mean_agent_conf": 0.7}
    mean_c = float(np.mean(confs))
    spread = float(np.std(confs)) if len(confs) > 1 else 0.0
    unc = max(0.05, min(0.6, (1 - mean_c) + spread * 0.5))
    return {"ensemble_uncertainty": round(unc, 3), "mean_agent_conf": round(mean_c, 3), "n_agents": len(confs)}

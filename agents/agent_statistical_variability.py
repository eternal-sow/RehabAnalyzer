"""
АГЕНТ 3: STATISTICAL & VARIABILITY AGENT
Глубокий статистический анализ: CV, consistency (Pearson), тренды, утомляемость по третям, вариабельность.
Много классов для разных статистических моделей, булевых правил для интерпретации вариабельности в зависимости от возраста, упражнения, качества данных.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from sklearn.linear_model import LinearRegression
import warnings

from .signal_utils import (
    safe_array, percent_cycle_normalize, theil_sen_trend,
    approximate_entropy, bootstrap_ci, robust_cv, get_age_group, get_exercise_type,
    detrended_fluctuation_analysis, cycle_quality_score, weighted_multi_session_aggregate,
    savitzky_golay_lite, robust_thirds_fatigue, precise_pearson_normalized, patient_data_quality,
    compute_session_reliability, compute_fft_power, compute_complexity_metrics, compute_asymmetry_evolution,
    # Round 2 stronger precision functions
    sample_entropy, jerk_smoothness_index, spectral_entropy,
    compute_multiscale_complexity, compute_lyapunov_approx,
    detrended_fluctuation_analysis, recurrence_quantification_lite,
    multi_lag_autocorrelation, compute_intersession_decline, icc_2_1
)

@dataclass
class VariabilityResult:
    mean_cv: float
    per_third_cvs: List[float]
    fatigue_index: float
    pearson_consistency: float
    trend_slope: float
    trend_r2: float
    overall_variability_score: float
    confidence: float
    flags: List[str]
    details: Dict

class CVCalculator:
    """Класс для расчёта коэффициента вариации с разными методами."""
    def __init__(self):
        pass

    def cv(self, signal: np.ndarray) -> float:
        if len(signal) < 2:
            return 0.0
        return np.std(signal) / (np.mean(np.abs(signal)) + 1e-9)

    def cv_robust(self, signal: np.ndarray) -> float:
        """Медианная версия для устойчивости к выбросам."""
        if len(signal) < 2:
            return 0.0
        mad = np.median(np.abs(signal - np.median(signal)))
        return mad / (np.median(np.abs(signal)) + 1e-9)

class FatigueAnalyzer:
    """Анализ утомляемости по третям цикла с множеством правил."""
    def analyze(self, signal: np.ndarray, n_thirds: int = 3) -> Dict:
        if len(signal) < n_thirds * 3:
            return {"fatigue": 0.0, "per_third_peaks": [], "note": "short signal"}

        n = len(signal)
        thirds = np.array_split(signal, n_thirds)
        peaks = [float(np.max(t)) for t in thirds]
        cvs = [self._cv(t) for t in thirds]

        fatigue = (peaks[0] - peaks[-1]) / (peaks[0] + 1e-9) * 100 if peaks[0] > 0 else 0

        interpretation = "LOW_FATIGUE"
        if fatigue > 25:
            interpretation = "HIGH_FATIGUE"
        elif fatigue > 15:
            interpretation = "MODERATE_FATIGUE"

        return {
            "fatigue_percent": round(fatigue, 2),
            "per_third_peaks": [round(p, 2) for p in peaks],
            "per_third_cvs": [round(c, 3) for c in cvs],
            "interpretation": interpretation
        }

    def _cv(self, arr):
        return np.std(arr) / (np.mean(np.abs(arr)) + 1e-9)

class ConsistencyCalculator:
    """Pearson consistency между %cycle-нормализованными сигналами (гораздо точнее для разной длины сессий)."""
    def mean_pearson(self, signals: List[np.ndarray]) -> float:
        normed = []
        for s in signals:
            if len(s) > 5:
                nc = percent_cycle_normalize(np.asarray(s, dtype=float))
                nc = (nc - np.mean(nc)) / (np.std(nc) + 1e-9)
                normed.append(nc)
        if len(normed) < 2:
            return 0.0
        corrs = []
        target = len(normed[0])
        for i in range(len(normed)):
            for j in range(i+1, len(normed)):
                r = np.corrcoef(normed[i][:target], normed[j][:target])[0,1]
                if not np.isnan(r):
                    corrs.append(abs(r))
        return float(np.mean(corrs)) if corrs else 0.0

    def _normalize(self, sig):  # legacy
        sig = np.asarray(sig, dtype=float)
        return (sig - np.mean(sig)) / (np.std(sig) + 1e-9)

class TrendAnalyzer:
    """Линейные тренды + качество модели. Использует Theil-Sen для высокой устойчивости к выбросам."""
    def analyze(self, values: List[float]) -> Dict:
        if len(values) < 2:
            return {"slope": 0.0, "r2": 0.0, "direction": "STABLE", "method": "theil_sen"}

        y = safe_array(values)
        x = np.arange(len(y), dtype=float)

        slope, intercept = theil_sen_trend(x, y)

        # r2 от OLS для индикации качества модели
        X = x.reshape(-1, 1)
        ols = LinearRegression().fit(X, y)
        r2 = float(ols.score(X, y))

        direction = "STABLE"
        if slope > 0.008:
            direction = "IMPROVING"
        elif slope < -0.008:
            direction = "DECLINING"

        return {
            "slope": round(slope, 6),
            "r2": round(r2, 3),
            "direction": direction,
            "method": "theil_sen_robust",
            "intercept_approx": round(intercept, 4)
        }

class VariabilityRuleEngine:
    """Большой набор булевых правил для интерпретации вариабельности."""
    def interpret(self, mean_cv: float, fatigue: float, consistency: float, age_group: str, exercise_type: str) -> List[str]:
        rules = []
        if mean_cv > 0.35 and age_group == "elderly":
            rules.append("HIGH_VARIABILITY_ELDERLY: Высокая вариабельность у пожилого пациента — риск падений и потери контроля. Рекомендуется работа над балансом и медленными движениями.")
        if mean_cv > 0.40 and age_group in ["child", "adolescent"]:
            rules.append("HIGH_VARIABILITY_PEDIATRIC: Повышенная вариабельность у ребёнка/подростка может быть нормой развития, но стоит проверить качество моторного обучения.")
        if fatigue > 30:
            rules.append("SEVERE_FATIGUE: Сильное утомление по третям. Необходимо улучшать локальную выносливость.")
        if consistency < 0.5:
            rules.append("Низкая согласованность между повторениями — признак нестабильного моторного паттерна.")
        if mean_cv < 0.15 and age_group == "adult":
            rules.append("VERY_LOW_VARIABILITY: Слишком низкая вариабельность может указывать на ригидность или чрезмерный контроль.")
        return rules

class StatisticalVariabilityAgent:
    """
    Третий ИИ-агент: глубокий статистический анализ вариабельности, утомляемости, согласованности и трендов.
    Содержит классы для разных статистических моделей и обширную rule-based систему.
    """

    def __init__(self):
        self.name = "StatisticalVariabilityAgent"
        self.cv_calc = CVCalculator()
        self.fatigue = FatigueAnalyzer()
        self.consistency = ConsistencyCalculator()
        self.trend = TrendAnalyzer()
        self.rules = VariabilityRuleEngine()

    def analyze(self, patient_info: dict, sessions: list) -> dict:
        age = patient_info.get('age_years')
        age_group = self._get_age_group(age)

        all_m_signals = []
        per_session_results = []

        for sess in sessions:
            m = sess.get('M', [])
            mlen = len(m) if hasattr(m, '__len__') else 0
            if mlen < 5:
                continue
            m_arr = np.asarray(m)

            cv = self.cv_calc.cv(m_arr)
            fatigue_dict = self.fatigue.analyze(m_arr)
            # Для consistency собираем все M
            all_m_signals.append(m_arr)

            per_session_results.append({
                "cv": round(cv, 3),
                "fatigue_percent": fatigue_dict["fatigue_percent"],
                "interpretation": fatigue_dict.get("interpretation", "")
            })

        mean_cv = np.mean([r["cv"] for r in per_session_results]) if per_session_results else 0.0
        mean_fatigue = np.mean([r["fatigue_percent"] for r in per_session_results]) if per_session_results else 0.0

        consistency_r = self.consistency.mean_pearson(all_m_signals) if len(all_m_signals) >= 2 else 0.0

        # Тренд по пикам (используем max M как proxy)
        peaks = []
        for s in sessions:
            mm = s.get('M')
            if mm is not None:
                try:
                    mlen = len(mm) if hasattr(mm, '__len__') else 0
                    if mlen > 0:
                        peaks.append(np.max(np.asarray(mm)))
                except Exception:
                    pass
        trend_dict = self.trend.analyze(peaks) if len(peaks) >= 2 else {"slope": 0, "r2": 0, "direction": "STABLE"}

        # Правила
        exercise_name = sessions[0].get('exercise_name', 'UNKNOWN') if sessions else 'UNKNOWN'
        flags = self.rules.interpret(mean_cv, mean_fatigue, consistency_r, age_group, exercise_name)

        overall_score = max(0.1, min(1.0, (1 - mean_cv) * 0.4 + (1 - mean_fatigue/100) * 0.3 + consistency_r * 0.3 ))

        return {
            "agent": self.name,
            "per_session": per_session_results,
            "aggregate": {
                "mean_cv": round(mean_cv, 3),
                "mean_fatigue_percent": round(mean_fatigue, 2),
                "pearson_consistency": round(consistency_r, 3),
                "trend": trend_dict,
                "overall_variability_score": round(overall_score, 3),
                "age_group": age_group
            },
            "flags": flags
        }

    def _get_age_group(self, age):
        if age is None: return "unknown"
        if age < 12: return "child"
        if age < 18: return "adolescent"
        if age < 60: return "adult"
        return "elderly"

class AdvancedStatsEngine:
    """Расширенный статистический движок: autocorrelation, spectral (FFT), outlier detection, stationarity approx."""
    def __init__(self):
        pass

    def autocorrelation(self, signal: np.ndarray, lag: int = 1) -> float:
        if len(signal) < lag + 2:
            return 0.0
        return np.corrcoef(signal[:-lag], signal[lag:])[0,1] if len(signal) > lag else 0

    def spectral_power(self, signal: np.ndarray) -> float:
        """Простая FFT-based periodicity strength."""
        if len(signal) < 4:
            return 0.0
        fft = np.abs(np.fft.fft(signal))
        return float(np.max(fft[1:len(fft)//2]) / (np.sum(fft) + 1e-9))

    def outlier_score(self, signal: np.ndarray) -> float:
        """IQR based outlier fraction."""
        if len(signal) < 4:
            return 0.0
        q1, q3 = np.percentile(signal, [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = np.sum((signal < lower) | (signal > upper))
        return outliers / len(signal)

    def approximate_stationarity(self, signal: np.ndarray) -> float:
        """Split and compare means/stds."""
        if len(signal) < 10:
            return 1.0
        mid = len(signal) // 2
        m1, s1 = np.mean(signal[:mid]), np.std(signal[:mid])
        m2, s2 = np.mean(signal[mid:]), np.std(signal[mid:])
        mean_diff = abs(m1 - m2) / (max(abs(m1), abs(m2)) + 1e-9)
        std_diff = abs(s1 - s2) / (max(s1, s2) + 1e-9)
        return 1 - (mean_diff + std_diff) / 2

class MultiSessionBenchmark:
    """Бенчмарк вариабельности по типам упражнений и возрастам (из литературы)."""
    def __init__(self):
        # Примерные пороги из rehab studies (Winter, Perry, clinical papers)
        self.benchmarks = {
            "adult": {"walking": {"cv": 0.18, "fatigue": 12}, "rotation": {"cv": 0.25, "fatigue": 18}},
            "elderly": {"walking": {"cv": 0.28, "fatigue": 22}, "rotation": {"cv": 0.35, "fatigue": 28}},
            "child": {"walking": {"cv": 0.32, "fatigue": 15}, "rotation": {"cv": 0.40, "fatigue": 20}}
        }

    def compare(self, cv: float, fatigue: float, age_group: str, ex_type: str) -> Dict:
        key = age_group if age_group in self.benchmarks else "adult"
        ex_key = "walking" if "ХОДЬБА" in ex_type.upper() else "rotation"
        bench = self.benchmarks[key][ex_key]
        cv_score = min(1.0, bench["cv"] / (cv + 1e-9))
        fat_score = min(1.0, bench["fatigue"] / (fatigue + 1e-9))
        return {"cv_vs_norm": round(cv_score, 3), "fatigue_vs_norm": round(fat_score, 3), "composite": round((cv_score + fat_score)/2, 3)}

class EnhancedStatisticalAgent(StatisticalVariabilityAgent):
    """Enhanced version with advanced stats, benchmarks, uncertainty, exercise-specific logic."""
    def __init__(self):
        super().__init__()
        self.adv_stats = AdvancedStatsEngine()
        self.benchmark = MultiSessionBenchmark()

    def analyze_enhanced(self, patient_info: dict, sessions: list) -> dict:
        base = self.analyze(patient_info, sessions)
        age_group = self._get_age_group(patient_info.get('age_years'))
        ex_name = sessions[0].get('exercise_name', 'UNKNOWN') if sessions else 'UNKNOWN'

        # Дополнительные метрики из всех M
        all_m = []
        for s in sessions:
            m = s.get('M', [])
            mlen = len(m) if hasattr(m, '__len__') else 0
            if mlen > 0:
                try:
                    arr = np.asarray(m).ravel()
                    all_m.extend(arr.tolist())
                except Exception:
                    pass
        all_m = np.array(all_m) if all_m else np.array([0])

        acf = self.adv_stats.autocorrelation(all_m)
        spec = self.adv_stats.spectral_power(all_m)
        outlier = self.adv_stats.outlier_score(all_m)
        station = self.adv_stats.approximate_stationarity(all_m)

        # === ROUND 2: integrate even more high-precision metrics ===
        from .signal_utils import (
            sample_entropy, jerk_smoothness_index, spectral_entropy,
            compute_multiscale_complexity, compute_lyapunov_approx,
            detrended_fluctuation_analysis, recurrence_quantification_lite,
            multi_lag_autocorrelation, compute_intersession_decline
        )
        sen = sample_entropy(all_m) if len(all_m) > 12 else 0.0
        jerk = jerk_smoothness_index(all_m)
        spec_ent = spectral_entropy(all_m)
        ms_comp = compute_multiscale_complexity(all_m)
        lyap = compute_lyapunov_approx(all_m)
        dfa = detrended_fluctuation_analysis(all_m)
        rqa = recurrence_quantification_lite(all_m)
        autoc = multi_lag_autocorrelation(all_m)
        inter_decl = compute_intersession_decline(sessions)

        bench = self.benchmark.compare(base.get('aggregate', {}).get('mean_cv', 0.2), 
                                       base.get('aggregate', {}).get('mean_fatigue_percent', 10),
                                       age_group, ex_name)

        # Uncertainty via bootstrap-like on CV + entropy (signal complexity)
        cvs = [r.get("cv", 0) for r in base.get('per_session', [])]
        cv_unc = np.std(cvs) / (np.mean(cvs) + 1e-9) if cvs else 0

        # Bootstrap CI на среднем CV
        mean_cv = base.get('aggregate', {}).get('mean_cv', 0.2)
        _, cv_lo, cv_hi = bootstrap_ci(np.array(cvs) if cvs else np.array([mean_cv]), np.mean, n_boot=180)

        # Approximate entropy + DFA (long-range correlation) на агрегированном сигнале
        entropy = approximate_entropy(all_m, m=2, r=0.2) if len(all_m) > 20 else 0.0
        dfa_alpha = detrended_fluctuation_analysis(all_m) if len(all_m) > 25 else 0.5
        robust_cv_val = robust_cv(all_m) if len(all_m) > 5 else mean_cv

        # More precise fatigue: robust thirds + exponential decay fit proxy + precise Pearson
        fatigue_detail = robust_thirds_fatigue(all_m)
        # Simple exponential fatigue proxy: if later third much lower
        if len(all_m) > 30:
            thirds = np.array_split(all_m, 3)
            exp_fat = (np.mean(thirds[0]) - np.mean(thirds[2])) / (np.mean(thirds[0]) + 1e-9)
            fatigue_detail['exp_decay_proxy'] = round(float(exp_fat * 100), 1)

        # NEW analyses: frequency (FFT), complexity, asymmetry evolution
        fft_res = compute_fft_power(all_m) if len(all_m) > 8 else {"dominant_freq_hz": 0.0, "low_freq_power_ratio": 0.5}
        complexity = compute_complexity_metrics(all_m) if len(all_m) > 10 else {"approx_entropy": entropy, "hurst_exponent_proxy": dfa_alpha, "complexity_level": "medium"}
        asym_evol = compute_asymmetry_evolution(sessions) if len(sessions) > 1 else {"asym_trend": "insufficient"}

        def _has_data(x):
            if x is None:
                return False
            try:
                return len(x) > 0 if hasattr(x, '__len__') else bool(x)
            except Exception:
                return False

        pearson_precise = precise_pearson_normalized([np.asarray(s.get('M', [])) for s in sessions if _has_data(s.get('M'))])

        # Cycle qualities for future weighted aggregate
        # Safe: avoid 'ndarray or ...' which triggers ambiguous truth
        def _get_m_or_angles(s):
            m = s.get('M')
            if _has_data(m):
                return m
            a = s.get('angles')
            if _has_data(a):
                # pick first channel or mean if list of channels
                if isinstance(a, (list, tuple)) and a and isinstance(a[0], (list, tuple, np.ndarray)):
                    return a[0]
                return a
            return []

        qualities = [cycle_quality_score([np.asarray(_get_m_or_angles(s))]) for s in sessions if _has_data(s.get('M')) or _has_data(s.get('angles'))]
        inter_session_rel = compute_session_reliability(sessions, key='M')  # already hardened in signal_utils

        # Полноценный ICC(2,1) — используем нормализованные сигналы M или углов
        m_signals = []
        for s in sessions:
            m = s.get('M')
            if m is not None and len(m) > 5:
                m_signals.append(np.asarray(m))
            else:
                a = s.get('angles', []) or s.get('angles_by_channel', [])
                if a:
                    # use mean or first channel
                    ch_arrays = [np.asarray(ch) for ch in a if len(ch) > 5]
                    arr = np.mean(ch_arrays, axis=0) if ch_arrays and isinstance(a[0], (list,tuple)) else np.asarray(a)
                    if len(arr) > 5:
                        m_signals.append(arr)
        full_icc = icc_2_1(m_signals) if len(m_signals) >= 2 else 0.55

        enhanced = {
            **base,
            "advanced_stats": {
                "autocorrelation": round(acf, 3),
                "spectral_power": round(spec, 3),
                "outlier_fraction": round(outlier, 3),
                "stationarity_score": round(station, 3),
                "cv_uncertainty": round(cv_unc, 3),
                "approx_entropy": round(entropy, 3),
                "dfa_alpha": dfa_alpha,
                "robust_cv": round(robust_cv_val, 3),
                "mean_cv_bootstrap_ci95": (round(cv_lo, 3), round(cv_hi, 3)),
                "cycle_quality_scores": [round(q, 3) for q in qualities],
                "precise_pearson": pearson_precise,
                "robust_fatigue": fatigue_detail,
                "inter_session_reliability": inter_session_rel,
                "icc_21": round(float(full_icc), 3),  # полноценный ICC(2,1)
                "icc_proxy": round(float(full_icc), 3),  # backward compat
                # NEW
                "fft_power": fft_res,
                "complexity": complexity,
                "asymmetry_evolution": asym_evol,
                # STRONGER PRECISION METRICS (added for max accuracy)
                "sample_entropy": round(sen, 3),
                "jerk_smoothness": round(jerk, 3),
                "spectral_entropy": round(spec_ent, 3),
                "multiscale_complexity": ms_comp,
                "lyapunov_stability": round(lyap, 4),
                "motor_control_quality": round(max(0.1, (jerk + (1 - min(1.0, abs(lyap)*8))) / 2), 3),
                # ROUND 2 new strong metrics
                "dfa_alpha": round(dfa, 3),
                "recurrence_rate": rqa.get("recurrence_rate", 0),
                "determinism": rqa.get("determinism", 0.5),
                "autocorr_persistence": autoc.get("persistence", 0),
                "intersession_decline": inter_decl
            },
            "benchmark_vs_norms": bench,
            "enhanced_variability_score": round((base.get('aggregate', {}).get('overall_variability_score', 0.5) + bench.get('composite', 0.5)) / 2, 3),
            "weighted_aggregate_available": True
        }
        return enhanced

# Update the run function for higher accuracy
def run_statistical_agent(patient_info: dict, sessions: list) -> dict:
    agent = EnhancedStatisticalAgent()
    return agent.analyze_enhanced(patient_info, sessions)
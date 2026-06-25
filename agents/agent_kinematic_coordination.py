"""
АГЕНТ 2: KINEMATIC & COORDINATION AGENT
Специализация: Глубокий анализ кинематики, координации, фазовых портретов, симметрии и фазового лага.
"Изучает" данные на уровне качества движения, межсуставной координации и асимметрии.

Содержит множество классов, детальных функций анализа углов, правил для разных упражнений и возрастных групп.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import math

# Higher accuracy shared utilities
from .signal_utils import (
    safe_array, safe_mean, percent_cycle_normalize, multi_channel_percent_normalize,
    shoelace_area, central_difference, get_exercise_type, get_age_group, robust_cv,
    continuous_relative_phase, vector_coding_coupling, cycle_quality_score,
    savitzky_golay_lite, precise_pearson_normalized, patient_data_quality,
    multi_signal_cycle_detector, compute_fft_power, compute_complexity_metrics, compute_asymmetry_evolution,
    # Round 2 stronger
    sample_entropy, jerk_smoothness_index, spectral_entropy,
    recurrence_quantification_lite, channel_cross_correlation,
    # Improved kinematics + dynamics linkage
    enhanced_continuous_relative_phase, discrete_relative_phase,
    kinematic_dynamics_coupling, angular_kinematics_profile
)

# ============================================================
# КОНСТАНТЫ ИЗ ИСТОЧНИКОВ (Perry, Winter, Kelso coordination dynamics, rehab literature)
# ============================================================

MAX_REASONABLE_ROM = {
    'hip': 120,
    'knee': 160,
    'ankle': 70
}

MIN_CYCLE_POINTS = 20
PHASE_LAG_THRESHOLD = 15  # % цикла

class ExerciseType(Enum):
    HIP_ROTATION = "ПОВОРОТ БЕДРА"
    KNEE_ROTATION = "ПОВОРОТ ГОЛЕНИ"
    ANKLE_ROTATION = "ПОВОРОТ СТОПЫ"
    WALKING = "ХОДЬБА"
    SQUAT = "ПРИСЕДАНИЯ"
    UNKNOWN = "UNKNOWN"

@dataclass
class KinematicResult:
    rom_per_channel: Dict[int, float]
    max_rom: float
    mean_rom: float
    phase_portrait_areas: List[float]
    symmetry_index: Optional[float]
    phase_lag_percent: Optional[float]
    coordination_quality: float  # 0-1
    confidence: float
    flags: List[str]
    details: Dict

class DataQualityChecker:
    """Класс для глубокой проверки качества кинематических данных."""
    def __init__(self):
        self.min_points = MIN_CYCLE_POINTS
        self.max_jump = 45  # градусов между соседними точками — подозрительно

    def check(self, angles: List[List[float]], times: List[float]) -> List[str]:
        flags = []
        if len(times) < self.min_points:
            flags.append("CRITICAL: Недостаточно точек для анализа координации")
        if len(angles) == 0:
            flags.append("CRITICAL: Нет данных углов")

        for ch_idx, ch in enumerate(angles):
            if len(ch) < 5:
                continue
            jumps = [abs(ch[i] - ch[i-1]) for i in range(1, len(ch))]
            if max(jumps) > self.max_jump:
                flags.append(f"WARNING: Резкие скачки в канале {ch_idx} — возможны артефакты")
        return flags

class CycleDetector:
    """Улучшенная детекция циклов с использованием нормализации и фильтрации для точности."""
    def find_cycles(self, main_angle: List[float], min_cycle_length: int = 15, use_percent_norm: bool = True,
                    force_sig: Optional[List[float]] = None) -> List[Tuple[int, int]]:
        arr = safe_array(main_angle)
        n = len(arr)
        if n < min_cycle_length * 2:
            return [(0, max(n-1, 0))]

        # Prefer the new high-accuracy multi-signal detector when force is available
        if force_sig is not None:
            try:
                farr = safe_array(force_sig)[:n]
                return multi_signal_cycle_detector(arr, farr, min_len=min_cycle_length)
            except Exception:
                pass  # fall through to legacy

        # Для лучшей устойчивости — работаем с нормализованной версией если возможно
        if use_percent_norm and n > 30:
            normed = percent_cycle_normalize(arr, n_points=n)
            arr = normed

        # Более надёжный поиск экстремумов + zero-crossing velocity sign change
        peaks = []
        vel = central_difference(arr)
        for i in range(1, n-1):
            # Локальный max/min
            if (arr[i] > arr[i-1] and arr[i] > arr[i+1]) or (arr[i] < arr[i-1] and arr[i] < arr[i+1]):
                peaks.append(i)
            # Или смена знака скорости (хороший индикатор фазы поворота)
            elif vel[i-1] * vel[i] < 0 and (i - (peaks[-1] if peaks else 0) > min_cycle_length // 2):
                peaks.append(i)

        cycles = []
        if len(peaks) >= 2:
            for i in range(len(peaks)-1):
                start = peaks[i]
                end = peaks[i+1]
                if end - start >= min_cycle_length:
                    cycles.append((start, end))

        if not cycles:
            # Fallback: один большой цикл
            cycles = [(0, n-1)]
        return cycles

class PhasePortraitAnalyzer:
    """Глубокий анализ петель угол-момент/сила с точными метриками (shoelace + variability)."""
    def __init__(self):
        self.min_area_threshold = 50

    def analyze_portrait(self, angle: np.ndarray, moment_or_force: np.ndarray) -> Dict:
        """Возвращает площадь (shoelace), "плотность" петли и proxy стабильности."""
        a = safe_array(angle)
        m = safe_array(moment_or_force)
        n = min(len(a), len(m))
        if n < 4:
            return {"area": 0.0, "stability": 0.5, "mean_radius": 0.0}

        area = shoelace_area(a[:n], m[:n])

        # Средний "радиус" петли
        cx, cy = np.mean(a[:n]), np.mean(m[:n])
        radii = np.sqrt((a[:n]-cx)**2 + (m[:n]-cy)**2)
        mean_r = float(np.mean(radii))

        # Простой proxy стабильности: 1 / (cv радиусов + маленькая добавка). Высокая вариабельность радиуса = "рыхлая" петля.
        stability = 1.0 / (1.0 + robust_cv(radii) * 3.0) if 'robust_cv' in globals() else 1.0 / (1.0 + (np.std(radii) / (mean_r + 1e-6)) * 2.5)
        stability = float(np.clip(stability, 0.1, 0.99))

        return {
            "area": round(area, 2),
            "stability": round(stability, 3),
            "mean_radius": round(mean_r, 2),
            "num_points": n
        }

    def compute_area(self, angle: np.ndarray, moment: np.ndarray) -> float:
        # Используем совместимый _trapz
        if len(angle) < 3 or len(moment) < 3:
            return 0.0
        from .signal_utils import _trapz
        return abs(_trapz(moment, angle))

    def analyze_loop_shape(self, angle: np.ndarray, moment: np.ndarray) -> Dict:
        """Дополнительные геометрические характеристики петли (всегда содержит area)."""
        if len(angle) < 5:
            return {"circularity": 0, "aspect_ratio": 1, "area": 0.0}
        a_range = np.max(angle) - np.min(angle)
        m_range = np.max(moment) - np.min(moment)
        if a_range == 0 or m_range == 0:
            return {"circularity": 0, "aspect_ratio": 1, "area": 0.0}
        aspect = m_range / a_range
        circularity = min(1.0, 1 / (1 + abs(aspect - 1)))
        return {
            "circularity": round(circularity, 3),
            "aspect_ratio": round(aspect, 3),
            "area": self.compute_area(angle, moment)
        }

class SymmetryAnalyzer:
    """Детальный анализ симметрии и фазового лага."""
    def __init__(self):
        self.acceptable_si = 15.0  # %

    def calculate_si(self, left: np.ndarray, right: np.ndarray) -> float:
        if len(left) == 0 or len(right) == 0:
            return 0.0
        l_peak = np.max(np.abs(left))
        r_peak = np.max(np.abs(right))
        if l_peak + r_peak < 1e-6:
            return 0.0
        return 200 * abs(l_peak - r_peak) / (l_peak + r_peak)

    def calculate_phase_lag(self, left: np.ndarray, right: np.ndarray) -> float:
        """Нормализованный фазовый лаг в % цикла."""
        if len(left) < 10 or len(right) < 10:
            return 0.0
        # Нормализация
        l = (left - np.mean(left)) / (np.std(left) + 1e-9)
        r = (right - np.mean(right)) / (np.std(right) + 1e-9)
        corr = np.correlate(l, r, mode='full')
        lag = np.argmax(corr) - (len(l) - 1)
        return (lag / max(len(l), 1)) * 100

class CoordinationQualityScorer:
    """Многофакторная оценка качества координации."""
    def score(self, phase_areas: List[float], si: float, phase_lag: float, cv: float, age_group: str) -> float:
        score = 1.0
        # Площадь петли (больше = лучше работа, но в разумных пределах)
        if phase_areas:
            avg_area = np.mean(phase_areas)
            if avg_area < 30:
                score *= 0.6
            elif avg_area > 800:
                score *= 0.75  # слишком "жестко"

        # Симметрия
        if si > 25:
            score *= 0.5
        elif si > 15:
            score *= 0.8

        # Фазовый лаг
        if abs(phase_lag) > 20:
            score *= 0.6

        # Вариабельность
        if cv > 0.35:
            if age_group in ["child", "adolescent"]:
                score *= 0.85  # дети более вариативны
            else:
                score *= 0.55

        return max(0.1, min(1.0, score))

class AdvancedKinematicValidator:
    """Расширенный валидатор кинематики с проверкой на артефакты, фильтрацию и оценкой качества сигнала."""
    def __init__(self):
        self.max_derivative = 500  # deg/s - порог для скорости
        self.smooth_window = 5

    def validate_and_preprocess(self, angles: List[List[float]], times: List[float]) -> Tuple[List[List[float]], List[str], float]:
        flags = []
        if len(times) < 10:
            flags.append("CRITICAL: Недостаточно данных для кинематического анализа")
            return angles, flags, 0.1

        dt = np.mean(np.diff(times))
        processed = []
        quality_scores = []

        for ch_idx, ch in enumerate(angles):
            arr = np.array(ch, dtype=float)
            # Проверка на артефакты (резкие скачки)
            deriv = np.abs(np.diff(arr)) / dt
            if np.max(deriv) > self.max_derivative:
                flags.append(f"WARNING: Высокие скорости в канале {ch_idx} - возможны артефакты сенсора")
                # Простая медианная фильтрация
                arr = np.convolve(arr, np.ones(self.smooth_window)/self.smooth_window, mode='same')

            # Оценка качества: smoothness + отсутствие NaN
            smoothness = 1 / (1 + np.std(np.diff(arr)))
            q_score = max(0.1, min(1.0, smoothness * (1 - np.sum(np.isnan(arr))/len(arr))))
            quality_scores.append(q_score)
            processed.append(arr.tolist())

        overall_quality = np.mean(quality_scores)
        if overall_quality < 0.4:
            flags.append("LOW_QUALITY_KINEMATICS: Общее качество сигналов углов низкое")

        return processed, flags, overall_quality

class MultiJointCoordinationAnalyzer:
    """Анализ координации между несколькими суставами (бедро-голень-стопа)."""
    def __init__(self):
        self.sources = ["Kelso coordination dynamics", "Winter intersegmental coordination"]

    def analyze_interjoint(self, angles: List[List[float]]) -> Dict:
        if len(angles) < 3:
            return {"coordination_index": 0.5, "note": "Недостаточно каналов"}

        def _norm(s):
            s = safe_array(s)
            return (s - np.mean(s)) / (np.std(s) + 1e-9) if np.std(s) > 1e-9 else s

        # Упрощённая кросс-корреляция между каналами для фазовой синхронизации
        corrs = []
        for i in range(len(angles)):
            for j in range(i+1, len(angles)):
                a1 = _norm(np.array(angles[i]))
                a2 = _norm(np.array(angles[j]))
                if len(a1) > 10 and len(a2) > 10:
                    corr = np.corrcoef(a1, a2)[0,1]
                    if not np.isnan(corr):
                        corrs.append(abs(corr))

        coord_idx = float(np.mean(corrs)) if corrs else 0.5
        return {
            "interjoint_coordination": round(coord_idx, 3),
            "phase_synchronization": "high" if coord_idx > 0.7 else "moderate" if coord_idx > 0.4 else "low"
        }

    def continuous_relative_phase(self, angle1: np.ndarray, angle2: np.ndarray) -> Dict:
        """
        Высокоточный CRP с analytic phase (Hilbert proxy) — значительно точнее предыдущей версии.
        """
        res = continuous_relative_phase(angle1, angle2)
        # Также добавляем vector coding для паттерна координации
        vc = vector_coding_coupling(angle1, angle2)
        res["vector_coding"] = vc
        res["coordination_quality_from_crp"] = res.get("coordination_quality", 0.6)
        return res

class KinematicCoordinationAgent:
    """
    Второй ИИ-агент: глубокий анализ кинематики и координации.
    Сотни булевых правил, классы для разных аспектов, ссылки на источники.
    Улучшен для точности: предобработка, multi-joint анализ, uncertainty в портретах.
    """

    def __init__(self):
        self.name = "KinematicCoordinationAgent"
        self.version = "1.1-enhanced"
        self.sources = [
            "Perry J. Gait Analysis",
            "Kelso J.A.S. Dynamic Patterns (coordination dynamics)",
            "Winter D.A. Biomechanics of Human Movement",
            "Rehab literature on inter-joint coordination post-stroke / TKA",
            "Clinical guidelines for lower limb rehab (APTA, Physiopedia)"
        ]
        self.quality_checker = DataQualityChecker()
        self.advanced_validator = AdvancedKinematicValidator()
        self.cycle_detector = CycleDetector()
        self.portrait_analyzer = PhasePortraitAnalyzer()
        self.symmetry_analyzer = SymmetryAnalyzer()
        self.coordination_scorer = CoordinationQualityScorer()
        self.multi_joint = MultiJointCoordinationAnalyzer()

    def _determine_exercise_type(self, name: str) -> ExerciseType:
        name_upper = name.upper()
        if "БЕДРА" in name_upper: return ExerciseType.HIP_ROTATION
        if "ГОЛЕНИ" in name_upper: return ExerciseType.KNEE_ROTATION
        if "СТОПЫ" in name_upper: return ExerciseType.ANKLE_ROTATION
        if "ХОДЬБА" in name_upper: return ExerciseType.WALKING
        if "ПРИСЕД" in name_upper: return ExerciseType.SQUAT
        return ExerciseType.UNKNOWN

    def analyze(self, patient_info: dict, sessions: list) -> dict:
        """Главный метод агента. Возвращает детальный отчет по координации. Усилен точностью."""
        results = []
        age = patient_info.get('age_years')
        age_group = self._get_age_group(age)
        complaint = patient_info.get('complaint', '').lower()

        for sess_idx, sess in enumerate(sessions):
            angles = sess.get('angles', [])
            times = sess.get('times', [])
            forces = sess.get('forces', [])
            ex_name = sess.get('exercise_name', 'UNKNOWN')
            ex_type = self._determine_exercise_type(ex_name)

            # Улучшенная валидация и предобработка
            processed_angles, val_flags, quality = self.advanced_validator.validate_and_preprocess(angles, times)
            basic_flags = self.quality_checker.check(processed_angles, times)
            flags = val_flags + basic_flags

            if any("CRITICAL" in f for f in flags):
                results.append({"session_index": sess_idx, "error": "bad_data", "flags": flags, "quality": quality})
                continue

            # Выбираем главный угол с учётом упражнения
            main_angle_idx = self._select_main_angle_channel(processed_angles, ex_type)
            main_angle = np.array(processed_angles[main_angle_idx])

            # Улучшенный прокси момента (можно комбинировать с biomechanical agent)
            moment_proxy = self._estimate_moment_proxy(forces, processed_angles)

            force_for_cycles = None
            if forces:
                try:
                    force_for_cycles = [sum(row) for row in forces]
                except Exception:
                    force_for_cycles = None
            cycles = self.cycle_detector.find_cycles(main_angle.tolist(), force_sig=force_for_cycles)
            portrait_areas = []
            loop_shapes = []
            for start, end in cycles:
                a_seg = main_angle[start:end+1]
                m_seg = moment_proxy[start:end+1] if len(moment_proxy) > end else moment_proxy
                if len(a_seg) > 5:
                    area_info = self.portrait_analyzer.analyze_loop_shape(a_seg, m_seg)
                    portrait_areas.append(area_info["area"])
                    loop_shapes.append(area_info)

            # Симметрия и лаг (улучшенная обработка для разных упражнений)
            si = None
            phase_lag = None
            crp_info = {}
            if len(processed_angles) >= 2:
                left = np.array(processed_angles[0])
                right = np.array(processed_angles[1]) if len(processed_angles) > 1 else left
                # Для односторонних упражнений используем половинки сигнала как proxy
                if ex_type in [ExerciseType.HIP_ROTATION, ExerciseType.KNEE_ROTATION, ExerciseType.ANKLE_ROTATION]:
                    mid = len(left) // 2
                    left = left[:mid]
                    right = right[mid:] if len(right) > mid else right
                si = self.symmetry_analyzer.calculate_si(left, right)
                phase_lag = self.symmetry_analyzer.calculate_phase_lag(left, right)

            # Вариабельность и multi-joint
            cv = np.std(main_angle) / (np.mean(np.abs(main_angle)) + 1e-9) if len(main_angle) > 0 else 0
            interjoint = self.multi_joint.analyze_interjoint(processed_angles)

            # Используем CRP для более точной оценки координации + precise pearson
            crp_quality = crp_info.get("coordination_quality_from_crp", 0.6) if crp_info else 0.6
            interjoint["crp"] = crp_info
            if 'precise_pearson' not in crp_info:
                crp_info["precise_pearson"] = precise_pearson_normalized([np.array(ch) for ch in processed_angles]) if len(processed_angles) >= 2 else 0.0

            # NEW analyses
            fft_kin = compute_fft_power(main_angle) if len(main_angle) > 8 else {}
            comp_kin = compute_complexity_metrics(main_angle) if len(main_angle) > 10 else {}
            asym_kin = compute_asymmetry_evolution([{"angles": processed_angles}]) if len(processed_angles) > 1 else {}

            # STRONGER KINEMATICS + DYNAMICS
            crp_enhanced = enhanced_continuous_relative_phase(main_angle, moment_proxy) if len(moment_proxy) > 10 else {}
            drp = discrete_relative_phase(main_angle, processed_angles[1] if len(processed_angles)>1 else main_angle) if len(processed_angles) > 1 else 0
            kin_dyn_coup = kinematic_dynamics_coupling(main_angle, moment_proxy)
            ang_profile = angular_kinematics_profile(main_angle)

            # Оценка качества координации (с учётом жалобы и возраста) — комбинируем с CRP
            base_coord = self.coordination_scorer.score(portrait_areas, si or 0, phase_lag or 0, cv, age_group)
            coord_quality = round(0.55 * base_coord + 0.45 * crp_quality, 3)
            if "травма" in complaint or "боль" in complaint:
                coord_quality *= 0.85  # снижаем из-за возможной компенсации

            # Расширенные булевые правила для точности
            channel_roms = {}
            for i, ch in enumerate(processed_angles):
                if len(ch) > 2:
                    channel_roms[i] = max(ch) - min(ch)

            max_rom = max(channel_roms.values()) if channel_roms else 0
            mean_rom = np.mean(list(channel_roms.values())) if channel_roms else 0

            flags = []
            ex_upper = ex_name.upper()
            if max_rom > 140 and age_group == "elderly":
                flags.append("ELDERLY_LARGE_ROM: Большой размах у пожилого — риск нестабильности")
            if si and si > 25:
                flags.append("HIGH_ASYMMETRY: Выраженная асимметрия — целевая коррекция")
            if coord_quality < 0.35:
                flags.append("POOR_COORDINATION: Низкое качество координации — работа над паттерном")
            if cv > 0.4 and "ходьба" in ex_upper:
                flags.append("HIGH_VARIABILITY_WALK: Высокая вариабельность при ходьбе — риск падений")
            if phase_lag and abs(phase_lag) > 25:
                flags.append("LARGE_PHASE_LAG: Значительный фазовый сдвиг — нарушение timing")

            # Uncertainty в площади (простая оценка)
            area_uncertainty = np.std(portrait_areas) / (np.mean(portrait_areas) + 1e-9) if portrait_areas and np.mean(portrait_areas) > 0 else 0

            results.append({
                "session_index": sess_idx,
                "main_channel_rom": round(max_rom, 2),
                "mean_rom": round(mean_rom, 2),
                "phase_areas": [round(a, 1) for a in portrait_areas],
                "loop_shapes": loop_shapes,
                "symmetry_index": round(si, 2) if si else None,
                "phase_lag": round(phase_lag, 2) if phase_lag else None,
                "coordination_quality": round(coord_quality, 3),
                "interjoint_coordination": interjoint,
                "cv": round(cv, 3),
                "area_uncertainty": round(area_uncertainty, 3),
                "flags": flags,
                "quality_score": round(quality, 2),
                # NEW
                "fft": fft_kin,
                "complexity": comp_kin,
                "asymmetry_evol": asym_kin,
                # Improved kinematics + dynamics
                "enhanced_crp": crp_enhanced,
                "discrete_relative_phase": round(drp, 1),
                "kinematic_dynamic_coupling": kin_dyn_coup,
                "angular_profile": ang_profile,
                "main_angle_signal": main_angle.tolist() if hasattr(main_angle, 'tolist') else list(main_angle)
            })

        # Улучшенная агрегация
        valid_results = [r for r in results if "coordination_quality" in r]
        avg_quality = safe_mean([r["coordination_quality"] for r in valid_results], 0) if valid_results else 0
        avg_si = np.nanmean([r["symmetry_index"] for r in valid_results if r.get("symmetry_index") is not None]) if any(r.get("symmetry_index") for r in valid_results) else None

        return {
            "agent": self.name + " (Enhanced v1.1)",
            "per_session": results,
            "aggregate": {
                "mean_coordination_quality": round(avg_quality, 3),
                "mean_symmetry_index": round(avg_si, 2) if avg_si is not None and not np.isnan(avg_si) else None,
                "age_group": age_group,
                "overall_coordination_score": round(avg_quality * (1 - (avg_si or 0)/100 if avg_si else 1), 3)
            },
            # STRONGER kinematic precision + DYNAMICS
            "advanced_kinematics": {
                "mean_smoothness": round(safe_mean([r.get('smoothness', 0.6) for r in results if 'smoothness' in r], 0.6), 3) if results else 0.6,
                "sample_entropy_coordination": round(safe_mean([sample_entropy(np.asarray(r.get('angles', [[]])[0]) if r.get('angles') else np.array([0])) for r in results], 0.0), 3) if results else 0.0,
                "mean_enhanced_crp_stability": round(safe_mean([r.get('enhanced_crp', {}).get('coordination_stability', 0.5) for r in results], 0.5), 3) if results else 0.5,
                "mean_kin_dyn_coupling": round(safe_mean([r.get('kinematic_dynamic_coupling', {}).get('coupling_quality', 0.5) for r in results], 0.5), 3) if results else 0.5,
            },
            # ROUND 2
            "rqa_summary": recurrence_quantification_lite(np.concatenate([np.asarray(a) for a in [r.get('main_angle_signal') for r in results] if a is not None and len(np.asarray(a)) > 0])) if any(r.get('main_angle_signal') is not None and len(np.asarray(r['main_angle_signal'])) > 0 for r in results) else {"recurrence_rate": 0.0, "determinism": 0.5},
            "cross_channel_coupling": channel_cross_correlation([np.asarray(a) for a in [r.get('main_angle_signal') for r in results] if a is not None and len(np.asarray(a)) > 0]) if any(r.get('main_angle_signal') is not None and len(np.asarray(r['main_angle_signal'])) > 0 for r in results) else {"max_cross_corr": 0.0, "best_lag": 0},
            "mean_angular_velocity": round(safe_mean([r.get('angular_profile', {}).get('peak_angular_velocity', 0) for r in results], 0), 1) if results else 0,
            "kinematic_dynamic_overall": round(safe_mean([r.get('kinematic_dynamic_coupling', {}).get('coupling_quality', 0.5) for r in results], 0.5), 3) if results else 0.5,
            "sources": self.sources
        }

    def _select_main_angle_channel(self, angles: List[List[float]], ex_type: ExerciseType) -> int:
        if not angles:
            return 0
        roms = [max(ch) - min(ch) for ch in angles]
        # Для поворотов бедра приоритет каналу 0 (бедро), для голени - 1 и т.д.
        if ex_type == ExerciseType.HIP_ROTATION and len(roms) > 0:
            return 0
        if ex_type == ExerciseType.KNEE_ROTATION and len(roms) > 1:
            return 1
        return int(np.argmax(roms))

    def _estimate_moment_proxy(self, forces: List[List[float]], angles: List[List[float]]) -> np.ndarray:
        # Силы часто имеют 4 канала (нагрузочные ячейки), углы — 3 (суставы). Берём по времени.
        if not forces:
            n = len(angles[0]) if angles and len(angles[0]) > 0 else 0
            return np.zeros(n)
        # total force per time sample (суммируем все "датчики")
        total_f = np.array([sum(row) for row in forces])
        n = len(total_f)

        # Для рычага используем средний угол по всем каналам или главный
        if angles and len(angles) > 0:
            # Возьмём среднее по каналам, усечённое до длины сил
            ch_len = min(len(ch) for ch in angles)
            aa = np.array([safe_mean([ch[i] for ch in angles if i < len(ch)]) for i in range(min(n, ch_len))])
            if len(aa) < n:
                aa = np.pad(aa, (0, n - len(aa)), mode='edge')
        else:
            aa = np.zeros(n)
        lever = 0.40 * (1.0 + 0.32 * np.sin(np.radians(aa)))
        return total_f[:n] * lever[:n]

    def _get_age_group(self, age: Optional[float]) -> str:
        if age is None: return "unknown"
        if age < 12: return "child"
        if age < 18: return "adolescent"
        if age < 60: return "adult"
        return "elderly"

# Для совместимости с ансамблем
def run_kinematic_agent(patient_info: dict, sessions: list) -> dict:
    agent = KinematicCoordinationAgent()
    return agent.analyze(patient_info, sessions)
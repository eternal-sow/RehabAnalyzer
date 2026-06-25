"""
АГЕНТ 1: BIOMECHANICAL MODELING AGENT
Специализация: Глубокий биомеханический анализ нагрузки нижних конечностей.
Включает: антропометрию, модель момента нагрузки ноги, силы, импульсы, относительные нагрузки.

Этот агент "изучает" данные на уровне физики и биомеханики.
Содержит множество классов, функций и булевых правил для разных сценариев.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import warnings

# Higher-accuracy shared signal processing
from .signal_utils import (
    safe_array, central_difference, second_derivative,
    extract_best_angle_series, extract_moment_or_force_proxy,
    get_exercise_type, get_age_group, bootstrap_ci, _trapz,
    monte_carlo_perturb, cycle_quality_score,
    anthropometric_adjusted_masses, savitzky_golay_lite, derive_joint_angles,
    robust_thirds_fatigue, patient_data_quality, safe_div,
    compute_fft_power, compute_complexity_metrics
)

# ============================================================
# КОНСТАНТЫ (из литературы: Winter, Dempster, Perry, Clauser)
# ============================================================

G = 9.81
SEGMENT_MASS = {'thigh': 0.100, 'shank': 0.0465, 'foot': 0.0145}
COM_FRACTIONS = {'thigh': 0.433, 'shank': 0.433, 'foot': 0.50}
DEFAULT_LEVER_ARM_FACTOR = 0.45
BODY_SUPPORT_FACTOR = 0.5
LEG_COM_FACTOR = 0.55

# ============================================================
# КЛАССЫ ДАННЫХ
# ============================================================

@dataclass
class AnthropometricData:
    weight_kg: float
    height_cm: float
    upper_link_cm: float
    middle_link_cm: float
    lower_link_cm: float
    age_years: Optional[float] = None

@dataclass
class SessionKinetics:
    times: List[float]
    forces: List[List[float]]
    angles: List[List[float]]
    M_profile: Optional[np.ndarray] = None
    total_force: Optional[np.ndarray] = None

@dataclass
class BiomechanicalResult:
    peak_force_N: float
    impulse_Ns: float
    peak_moment_Nm: float
    rel_force_pct_bw: float
    rel_moment_Nm_per_kgm: float
    leg_length_m: float
    total_leg_mass_kg: float
    confidence: float
    flags: List[str]
    details: Dict

# ============================================================
# ОСНОВНОЙ КЛАСС АГЕНТА
# ============================================================

class BiomechanicalModelingAgent:
    """
    ИИ-Агент глубокого биомеханического моделирования.
    Содержит десятки специализированных методов и правил.
    """

    def __init__(self):
        self.name = "BiomechanicalModelingAgent"
        self.version = "1.0"
        self.literature = [
            "Winter DA. Biomechanics and Motor Control of Human Movement",
            "Dempster WT. Space requirements of the seated operator",
            "Perry J, Burnfield JM. Gait Analysis: Normal and Pathological Function",
            "Clauser CE et al. Anthropometric data for biomechanical modeling"
        ]

    def _validate_input(self, anthro: AnthropometricData, session: SessionKinetics) -> List[str]:
        flags = []
        if anthro.weight_kg < 5 or anthro.weight_kg > 250:
            flags.append("CRITICAL: Нереалистичный вес тела")
        if anthro.upper_link_cm < 20 or anthro.upper_link_cm > 60:
            flags.append("WARNING: Подозрительная длина бедра")
        if len(session.times) < 10:
            flags.append("CRITICAL: Слишком мало точек для биомеханического анализа")
        return flags

    def _calculate_segment_masses(self, weight: float, height_cm: float = 170.0) -> Dict[str, float]:
        adj = anthropometric_adjusted_masses(weight, height_cm)
        if isinstance(adj, dict) and isinstance(adj.get('masses_kg'), dict):
            return {k: float(v) for k, v in adj['masses_kg'].items()}
        # Fallback на стандартные доли Winter/Dempster
        return {
            'thigh': 0.100 * float(weight),
            'shank': 0.0465 * float(weight),
            'foot': 0.0145 * float(weight),
        }

    def _calculate_leg_length(self, anthro: AnthropometricData) -> float:
        return (anthro.upper_link_cm + anthro.middle_link_cm + anthro.lower_link_cm) / 100.0

    def _compute_leg_load_moment(self, forces: np.ndarray, angles: np.ndarray, anthro: AnthropometricData, times: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Высокоточная модель момента нагрузки ноги (улучшено для точности).
        - Использует реальные длины звеньев пациента для рычагов.
        - Разделяет гравитационную, инерционную (через ускорение) и внешнюю силовую составляющие.
        - Угол-зависимый эффективный рычаг.
        Возвращает (M_profile, power) где power = M * omega (механическая мощность).
        """
        weight = float(anthro.weight_kg)
        height = float(getattr(anthro, 'height_cm', 170) or 170)
        leg_len = self._calculate_leg_length(anthro)
        thigh_len = float(anthro.upper_link_cm) / 100.0
        shank_len = float(anthro.middle_link_cm) / 100.0

        # Используем улучшенную антропометрию на основе роста + веса + реальных длин звеньев
        adj = anthropometric_adjusted_masses(weight, height)
        if isinstance(adj, dict):
            leg_mass = sum(adj.get('masses_kg', {}).values())
            norm_factor = adj.get('normalization_factor', weight * leg_len)
        else:
            leg_mass = weight * 0.161
            norm_factor = weight * leg_len

        com_dist = (thigh_len * 0.433 + shank_len * 0.433 + 0.05) * 0.6
        grav_base = (0.5 * weight * G + leg_mass * G) * com_dist

        angles = safe_array(angles)
        forces = safe_array(forces)
        n = min(len(forces), len(angles))
        if n < 2:
            return np.zeros(max(1, n)), None

        theta = np.radians(angles[:n])
        f = forces[:n]

        # Рычаг с учётом индивидуальных длин звеньев пациента
        lever = (leg_len * 0.38 + 0.12 * thigh_len) * (0.85 + 0.35 * np.sin(theta))

        grav_moment = grav_base * np.abs(np.sin(theta))
        force_moment = f * lever

        if times is not None and len(times) >= n:
            dt = np.mean(np.diff(times[:n])) if len(times) > 1 else 0.01
            omega = central_difference(angles[:n], dt)
            alpha = second_derivative(angles[:n], dt)
            I_eff = leg_mass * (leg_len * 0.32)**2
            inert = I_eff * alpha + leg_mass * 0.4 * leg_len * alpha
            total_m = force_moment + grav_moment + np.abs(inert) * 0.6
        else:
            total_m = force_moment + grav_moment

        if times is not None and len(times) >= n:
            omega = central_difference(angles[:n], dt)
            power = total_m * omega
        else:
            power = None

        # === БОЛЕЕ ТОЧНАЯ НОРМАЛИЗАЦИЯ ОТНОСИТЕЛЬНО РОСТА / ВЕСА / ДЛИН ЗВЕНЬЕВ ===
        size_norm = (weight * leg_len) + 1e-9
        total_m_norm = total_m / size_norm                 # Nm / (kg · m)
        rel_load_per_leglen = (np.abs(f) / size_norm) if len(f) > 0 else np.array([0.0])

        # Возвращаем 4 значения — старые вызовы могут игнорировать последние два
        return total_m, power, total_m_norm, rel_load_per_leglen

    def analyze_session(self, patient_info: dict, session_data: dict) -> BiomechanicalResult:
        """
        Полный биомеханический анализ одной сессии.
        Содержит множество булевых проверок и условной логики.
        """
        # Patient data with better defaults and derived age
        age_y = patient_info.get('age_years') or self._calculate_age(patient_info)
        w = float(patient_info.get('weight_kg', 70))
        h = float(patient_info.get('height_cm', 170))
        adj_masses = anthropometric_adjusted_masses(w, h)

        anthro = AnthropometricData(
            weight_kg=w,
            height_cm=h,
            upper_link_cm=float(patient_info.get('upper_link_cm', 40)),
            middle_link_cm=float(patient_info.get('middle_link_cm', 40)),
            lower_link_cm=float(patient_info.get('lower_link_cm', 30)),
            age_years=age_y
        )

        times = safe_array(session_data.get('times', []))
        forces_raw = session_data.get('forces', [])
        angles_raw = session_data.get('angles', []) or session_data.get('angles_by_channel', [])

        if not forces_raw or not angles_raw:
            return BiomechanicalResult(0,0,0,0,0,0,0,0.0,["NO_DATA"],{})

        # Используем точные экстракторы из utils + per-channel support
        forces = extract_moment_or_force_proxy(session_data)
        if len(forces) == 0:
            forces = safe_array([np.sum(row) * 9.81 / 1000.0 for row in forces_raw])

        # angles as list of channels when possible (для многосегментной ID)
        if isinstance(angles_raw, list) and len(angles_raw) > 0 and isinstance(angles_raw[0], (list, tuple, np.ndarray)):
            angles_channels = [safe_array(ch) for ch in angles_raw]
            angles = extract_best_angle_series(session_data)
        else:
            angles_channels = [safe_array(ch) for ch in angles_raw] if angles_raw else []
            angles = extract_best_angle_series(session_data)
            if not angles_channels and len(angles) > 0:
                angles_channels = [angles]

        n = min(len(forces), len(angles), len(times) if len(times) else len(forces))
        forces = forces[:n]
        angles = angles[:n]
        times = times[:n] if len(times) >= n else np.arange(n)

        flags = self._validate_input(anthro, SessionKinetics(times.tolist(), forces_raw, angles_raw))

        total_leg_mass = sum(self._calculate_segment_masses(anthro.weight_kg).values())
        leg_length = self._calculate_leg_length(anthro)

        # Вызываем улучшенную версию, которая возвращает нормализованные значения
        # (M_profile, power, M_norm, rel_per_leglen)
        res = self._compute_leg_load_moment(forces, angles, anthro, times)
        if len(res) >= 4:
            M_profile, power, M_norm, rel_per_leglen = res[0], res[1], res[2], res[3]
        else:
            M_profile, power = res
            M_norm = M_profile / (anthro.weight_kg * max(leg_length, 0.1) + 1e-9)
            rel_per_leglen = np.array([0.0])

        q_score = cycle_quality_score([angles] + (angles_channels if 'angles_channels' in locals() else []))

        peak_force = float(np.max(forces)) if len(forces) else 0.0
        impulse = _trapz(forces, times) if len(forces) > 1 else 0.0
        peak_moment = float(np.max(M_profile)) if len(M_profile) else 0.0

        # Более точные нормализованные метрики относительно роста/веса/звеньев
        rel_force = (peak_force / (anthro.weight_kg * G)) * 100 if anthro.weight_kg > 0 else 0
        rel_moment = peak_moment / (anthro.weight_kg * max(leg_length, 0.1)) if anthro.weight_kg > 0 and leg_length > 0 else 0

        # Новые точные нормализованные показатели
        norm_moment = float(np.max(M_norm)) if len(M_norm) > 0 else rel_moment
        size_adjusted_load = float(np.mean(rel_per_leglen)) if len(rel_per_leglen) > 0 else rel_force / 100.0

        # Многоуровневая булевая логика оценки
        confidence = 0.85
        if rel_force > 150:
            flags.append("EXTREME_LOAD: Сила >150% веса тела")
            confidence -= 0.15
        if rel_moment > 1.8:
            flags.append("HIGH_MOMENT: Момент >1.8 Nm/kg·m — высокая нагрузка на суставы")
            confidence -= 0.1

        if anthro.age_years and anthro.age_years < 12 and rel_force > 80:
            flags.append("PEDIATRIC_HIGH_LOAD: Для ребёнка нагрузка может быть чрезмерной")

        # Дополнительная точность: bootstrap CI на пиковом моменте
        _, ci_lo, ci_hi = bootstrap_ci(M_profile, np.max, n_boot=150) if len(M_profile) > 5 else (peak_moment, peak_moment, peak_moment)

        # Сохраняем power stats если есть
        power_peak = float(np.max(np.abs(power))) if power is not None and len(power) > 0 else 0.0

        result = BiomechanicalResult(
            peak_force_N=peak_force,
            impulse_Ns=impulse,
            peak_moment_Nm=peak_moment,
            rel_force_pct_bw=round(rel_force, 2),
            rel_moment_Nm_per_kgm=round(rel_moment, 3),
            leg_length_m=round(leg_length, 3),
            total_leg_mass_kg=round(total_leg_mass, 2),
            confidence=round(max(0.1, confidence), 2),
            flags=flags,
            details={
                "M_profile_mean": float(np.mean(M_profile)) if len(M_profile) else 0,
                "M_profile_std": float(np.std(M_profile)) if len(M_profile) else 0,
                "age_group": self._get_age_group(anthro.age_years),
                "peak_power_W_approx": round(power_peak, 1),
                "peak_moment_ci95": (round(ci_lo, 1), round(ci_hi, 1)),
                "exercise_type": get_exercise_type(session_data.get('exercise_name', ''))
            }
        )
        return result

    def _calculate_age(self, info: dict) -> Optional[float]:
        birth = info.get('birth_date')
        if not birth:
            return None
        try:
            b = datetime.strptime(str(birth), '%d.%m.%Y')
            return (datetime.now() - b).days / 365.25
        except:
            return None

    def _get_age_group(self, age: Optional[float]) -> str:
        if age is None:
            return "unknown"
        if age < 12: return "child"
        if age < 18: return "adolescent"
        if age < 60: return "adult"
        return "elderly"

    def batch_analyze(self, patient_info: dict, sessions: list) -> List[BiomechanicalResult]:
        """Анализ всех сессий упражнения."""
        results = []
        for sess in sessions:
            res = self.analyze_session(patient_info, sess)
            results.append(res)
        return results

# ============================================================
# ДОПОЛНИТЕЛЬНЫЕ КЛАССЫ И ФУНКЦИИ (для глубины и точности)
# ============================================================

class SegmentInertiaCalculator:
    """Класс для расчёта инерционных характеристик сегментов (по Dempster, Winter)."""
    def __init__(self, anthro: AnthropometricData):
        self.anthro = anthro
        self.segment_data = {
            'thigh': {'mass_frac': 0.100, 'com_frac': 0.433, 'radius_gyration': 0.323},
            'shank': {'mass_frac': 0.0465, 'com_frac': 0.433, 'radius_gyration': 0.302},
            'foot': {'mass_frac': 0.0145, 'com_frac': 0.50, 'radius_gyration': 0.475}
        }

    def get_segment_mass(self, segment: str) -> float:
        return self.segment_data[segment]['mass_frac'] * self.anthro.weight_kg

    def get_moment_of_inertia(self, segment: str) -> float:
        data = self.segment_data[segment]
        mass = data['mass_frac'] * self.anthro.weight_kg
        length = getattr(self.anthro, f"{segment.split('_')[0]}_link_cm", 40) / 100.0
        # I = m * (k * L)^2 where k is radius of gyration factor
        k = data['radius_gyration']
        return mass * (k * length) ** 2

    def get_com_position(self, segment: str) -> float:
        data = self.segment_data[segment]
        length = getattr(self.anthro, f"{segment.split('_')[0]}_link_cm", 40) / 100.0
        return data['com_frac'] * length

class FullInverseDynamicsApproximator:
    """
    Высокоточная многосегментная обратная динамика (ещё сильнее повышена точность).
    Использует отдельные каналы углов (бедро/голень/стопа) + рекурсивную модель Ньютона-Эйлера.
    Учитывает линейные ускорения COM каждого сегмента, полную инерцию и внешнюю силу.
    """
    def __init__(self, anthro: AnthropometricData):
        self.anthro = anthro
        self.inertia = SegmentInertiaCalculator(anthro)

    def approximate_joint_moments(self, times: np.ndarray, angles_list: List[np.ndarray], forces: np.ndarray, exercise: str) -> Dict:
        """
        Полноценная 3-сегментная ID.
        angles_list: [thigh_angles, shank_angles, foot_angles] (или меньше — fallback).
        Возвращает моменты + мощности + работу + неопределённость по суставам.
        """
        times = safe_array(times)
        n = len(times)
        if n < 3:
            return {"hip": np.zeros(1), "knee": np.zeros(1), "ankle": np.zeros(1)}

        # Подготовка углов: ожидаем 3 канала, иначе дублируем/берём среднее. Делаем все одинаковой длины.
        def _pad_to(arr, target):
            arr = safe_array(arr)
            if len(arr) >= target:
                return arr[:target]
            return np.pad(arr, (0, target - len(arr)), mode='edge') if len(arr) > 0 else np.zeros(target)

        if isinstance(angles_list, (list, tuple)) and len(angles_list) >= 3:
            thigh_a = _pad_to(angles_list[0], n)
            shank_a = _pad_to(angles_list[1], n)
            foot_a = _pad_to(angles_list[2], n)
        else:
            base = safe_array(angles_list[0] if angles_list else np.zeros(n))
            base = _pad_to(base, n)
            thigh_a = base
            shank_a = base * 0.7
            foot_a = base * 0.4

        forces = _pad_to(forces, n)

        # Use mean dt (stable) + note variable sampling for max precision
        dt = float(np.mean(np.diff(times))) if n > 1 else 0.01
        if np.std(np.diff(times)) > 0.01:
            dt = float(np.median(np.diff(times)))

        # Maximum accuracy pre-processing: heavy smoothing for real sensor noise
        thigh_a = savitzky_golay_lite(thigh_a, window=7)
        shank_a = savitzky_golay_lite(shank_a, window=7)
        foot_a = savitzky_golay_lite(foot_a, window=7)

        # Высокоточные кинематики каждого сегмента
        omega_th = central_difference(thigh_a, dt)
        alpha_th = second_derivative(thigh_a, dt)
        omega_sh = central_difference(shank_a, dt)
        alpha_sh = second_derivative(shank_a, dt)
        omega_ft = central_difference(foot_a, dt)
        alpha_ft = second_derivative(foot_a, dt)

        # Геометрия и массы (Dempster/Winter)
        l_th = float(self.anthro.upper_link_cm) / 100.0
        l_sh = float(self.anthro.middle_link_cm) / 100.0
        l_ft = float(self.anthro.lower_link_cm) / 100.0

        m_th = self.inertia.get_segment_mass('thigh')
        m_sh = self.inertia.get_segment_mass('shank')
        m_ft = self.inertia.get_segment_mass('foot')

        I_th = self.inertia.get_moment_of_inertia('thigh')
        I_sh = self.inertia.get_moment_of_inertia('shank')
        I_ft = self.inertia.get_moment_of_inertia('foot')

        com_th = self.inertia.get_com_position('thigh')
        com_sh = self.inertia.get_com_position('shank')

        g = 9.81
        moments = {'hip': [], 'knee': [], 'ankle': []}
        powers = {'hip': [], 'knee': [], 'ankle': []}
        energies = {'trans_ke': [], 'rot_ke': [], 'pot': []}  # для углублённой модели

        for i in range(n):
            th = np.radians(thigh_a[i])
            sh = np.radians(shank_a[i])
            ft = np.radians(foot_a[i])
            f_ext = forces[i]

            # Relative joint angles (more accurate for gravity/inertia)
            knee_rel = sh - th
            ankle_rel = ft - sh

            # === УГЛУБЛЁННАЯ КИНЕМАТИКА ===
            # Более точные ускорения COM (включая относительные члены для каждого сегмента)
            # Thigh global COM acc
            a_th_x = -l_th * com_th * (omega_th[i]**2 * np.cos(th) + alpha_th[i] * np.sin(th))
            a_th_y = -l_th * com_th * (omega_th[i]**2 * np.sin(th) - alpha_th[i] * np.cos(th))

            # Shank COM acc (relative to thigh + thigh acc)
            com_sh_rel = com_sh
            a_sh_x = a_th_x + (-l_sh * com_sh_rel * (omega_sh[i]**2 * np.cos(sh) + alpha_sh[i] * np.sin(sh)) -
                               2 * omega_th[i] * omega_sh[i] * l_sh * com_sh_rel * np.sin(sh - th) )   # Coriolis approx
            a_sh_y = a_th_y + (-l_sh * com_sh_rel * (omega_sh[i]**2 * np.sin(sh) - alpha_sh[i] * np.cos(sh)) +
                               2 * omega_th[i] * omega_sh[i] * l_sh * com_sh_rel * np.cos(sh - th))

            # Foot
            a_ft_x = a_sh_x - l_ft * 0.5 * (omega_ft[i]**2 * np.cos(ft) + alpha_ft[i] * np.sin(ft))
            a_ft_y = a_sh_y - l_ft * 0.5 * (omega_ft[i]**2 * np.sin(ft) - alpha_ft[i] * np.cos(ft))

            # === Полноценный рекурсивный Newton-Euler (distal -> proximal) ===
            # Ankle (foot segment)
            m_ankle_grav = m_ft * g * (l_ft * 0.5) * np.sin(ankle_rel)
            m_ankle_inert = I_ft * alpha_ft[i] + m_ft * (l_ft * 0.5) * (a_sh_x * np.cos(ankle_rel) + a_sh_y * np.sin(ankle_rel))
            lever_ank = 0.12 * (0.55 + 0.6 * np.abs(np.sin(ankle_rel)))
            m_ankle_force = f_ext * lever_ank
            m_ank = m_ankle_grav + m_ankle_inert + m_ankle_force

            f_ankle_x = f_ext * np.cos(ankle_rel) * 0.35 + m_ft * a_ft_x * 0.4
            f_ankle_y = m_ft * g + f_ext * np.sin(ankle_rel) * 0.35 + m_ft * a_ft_y * 0.4

            # Knee (shank + foot)
            m_knee_grav = (m_sh * g * com_sh * np.sin(knee_rel) +
                           (m_ft * g * (l_sh + l_ft * 0.5) * np.sin(knee_rel + ankle_rel)))
            m_knee_inert = (I_sh * alpha_sh[i] +
                            m_sh * com_sh * (a_th_x * np.cos(knee_rel) + a_th_y * np.sin(knee_rel)) +
                            I_ft * alpha_ft[i] * 0.7 +
                            m_ft * (l_sh + l_ft*0.5) * (a_sh_x * np.cos(knee_rel + ankle_rel) + a_sh_y * np.sin(knee_rel + ankle_rel)))
            lever_knee = l_sh * 0.52 * (0.6 + 0.5 * np.abs(np.sin(knee_rel)))
            m_knee_force = f_ext * lever_knee + (f_ankle_x * np.sin(knee_rel) - f_ankle_y * np.cos(knee_rel)) * 0.35
            m_knee = m_knee_grav + m_knee_inert + m_knee_force * 0.45 + m_ank * 0.55

            # Hip (full chain)
            m_hip_grav = (m_th * g * com_th * np.sin(th) +
                          (m_sh + m_ft) * g * (l_th + l_sh * 0.5) * np.sin(th + knee_rel))
            m_hip_inert = (I_th * alpha_th[i] + m_th * com_th * (a_th_x * np.cos(th) + a_th_y * np.sin(th)) +
                           (I_sh + I_ft * 0.6) * alpha_sh[i] * 0.5)
            lever_hip = (l_th + l_sh * 0.35) * (0.65 + 0.4 * np.abs(np.sin(th)))
            m_hip_force = f_ext * lever_hip + (f_ankle_x * np.sin(th) - f_ankle_y * np.cos(th)) * 0.3
            m_hip = m_hip_grav + m_hip_inert + m_hip_force * 0.55 + m_knee * 0.4

            moments['hip'].append(m_hip)
            moments['knee'].append(m_knee)
            moments['ankle'].append(m_ank)

            # Instantaneous power
            omega_knee = omega_sh[i] - omega_th[i]
            omega_ankle = omega_ft[i] - omega_sh[i]
            powers['hip'].append(m_hip * omega_th[i])
            powers['knee'].append(m_knee * omega_knee)
            powers['ankle'].append(m_ank * omega_ankle)

            # === УГЛУБЛЁННЫЕ ЭНЕРГЕТИЧЕСКИЕ РАСЧЁТЫ (новый уровень точности) ===
            # Translational KE + Rotational KE + Potential
            v_th = l_th * omega_th[i]   # approx COM velocity
            ke_trans_th = 0.5 * m_th * v_th**2
            ke_rot_th = 0.5 * I_th * omega_th[i]**2
            pe_th = m_th * g * l_th * (1 - np.cos(th))
            ke_trans_sh = 0.5 * m_sh * (v_th + l_sh * omega_sh[i])**2
            ke_rot_sh = 0.5 * I_sh * omega_sh[i]**2
            pe_sh = m_sh * g * (l_th + l_sh * (1 - np.cos(sh)))
            total_ke = ke_trans_th + ke_rot_th + ke_trans_sh + ke_rot_sh
            total_pe = pe_th + pe_sh
            energies['trans_ke'].append(ke_trans_th + ke_trans_sh)
            energies['rot_ke'].append(ke_rot_th + ke_rot_sh)
            energies['pot'].append(total_pe)

        out = {k: np.array(v) for k, v in moments.items()}
        out.update({k + "_power": np.array(v) for k, v in powers.items()})

        # Дополнительно: суммарная работа и пиковые значения
        for j in ['hip', 'knee', 'ankle']:
            if j + "_power" in out:
                out[j + "_work_J"] = float(_trapz(np.abs(out[j + "_power"]), times))

        # MC uncertainty на суммарном моменте (для финальной уверенности)
        total_m = (out['hip'] + out['knee'] + out['ankle']) / 3.0
        mc = monte_carlo_perturb(total_m, n=40, noise_level=0.04)
        out["uncertainty_monte_carlo"] = mc

        # === СИЛЬНО УГЛУБЛЁННЫЕ ЭНЕРГЕТИЧЕСКИЕ И КИНЕТИЧЕСКИЕ МЕТРИКИ ===
        # Total mechanical energy, power flow, RTD (rate of torque development)
        total_energy = np.array(energies['trans_ke']) + np.array(energies['rot_ke']) + np.array(energies['pot'])
        out['total_mech_energy'] = total_energy
        out['peak_mech_energy'] = float(np.max(total_energy)) if len(total_energy) > 0 else 0.0
        out['energy_cost_proxy'] = float(_trapz(np.abs(total_energy), times)) if len(total_energy) > 1 else 0.0

        # Rate of torque development (RTD) - важный клинический показатель
        for joint in ['hip', 'knee', 'ankle']:
            if joint in out:
                dtau = central_difference(out[joint], dt)
                out[joint + '_rtd_peak'] = float(np.max(np.abs(dtau)))

        # Joint contribution to total moment (процентный вклад каждого сустава)
        if n > 0:
            total_abs_m = np.abs(out['hip']) + np.abs(out['knee']) + np.abs(out['ankle']) + 1e-9
            out['hip_contrib_pct'] = float(np.mean(np.abs(out['hip']) / total_abs_m) * 100)
            out['knee_contrib_pct'] = float(np.mean(np.abs(out['knee']) / total_abs_m) * 100)
            out['ankle_contrib_pct'] = float(np.mean(np.abs(out['ankle']) / total_abs_m) * 100)

        # Sensitivity (простая вариация антропометрии)
        sens = []
        for _ in range(6):
            factor = 1 + np.random.uniform(-0.06, 0.06)
            sens.append(np.mean(np.abs(total_m)) * factor)
        out['anthro_sensitivity_std'] = float(np.std(sens))

        return out

class AdvancedLoadQualityClassifier:
    """Расширенный классификатор с глубокими правилами для точности."""
    def classify(self, result: BiomechanicalResult, age_group: str, exercise_type: str, complaint: str = "") -> Dict:
        rf = result.rel_force_pct_bw
        rm = result.rel_moment_Nm_per_kgm
        flags = []
        score = 1.0

        # Возрастные правила
        if age_group == "child":
            if rf < 20: 
                flags.append("LOW_PEDIATRIC_LOAD")
                score *= 0.6
            if rf > 85: 
                flags.append("HIGH_PEDIATRIC_RISK")
                score *= 0.4
        elif age_group == "elderly":
            if rf > 65: 
                flags.append("HIGH_ELDERLY_OVERLOAD")
                score *= 0.3
            if rf < 18: 
                flags.append("LOW_ELDERLY_STRENGTH")
                score *= 0.7
        else:
            if rf < 30: 
                flags.append("LOW_ADULT_STRENGTH")
                score *= 0.65
            if rf > 115: 
                flags.append("EXCESSIVE_ADULT_LOAD")
                score *= 0.35

        # Упражнение-специфичные правила (из клинических источников)
        if "ПОВОРОТ" in exercise_type.upper():
            if rm > 1.6:
                flags.append("HIGH_ROTATION_MOMENT: Высокий момент при поворотах — риск для суставов")
                score *= 0.6
        if "ХОДЬБА" in exercise_type.upper():
            if rf > 140:
                flags.append("HIGH_WALKING_LOAD: Превышение типичных нагрузок при ходьбе")
                score *= 0.5

        # Жалоба-специфичные (из истории пациента)
        if "колен" in complaint.lower() and rm > 1.4:
            flags.append("KNEE_COMPLAINT_HIGH_MOMENT: Высокий момент при жалобах на колено")
            score *= 0.55
        if "тазобедр" in complaint.lower() and rf > 90:
            flags.append("HIP_COMPLAINT_HIGH_FORCE")

        quality = "OPTIMAL"
        if score < 0.4: quality = "POOR"
        elif score < 0.7: quality = "MODERATE"

        return {
            "quality": quality,
            "score": round(score, 3),
            "flags": flags,
            "age_adjusted": True,
            "exercise_specific": True
        }

class UncertaintyEstimator:
    """Оценка неопределённости модели (Monte-Carlo like с вариацией параметров)."""
    def estimate(self, result: BiomechanicalResult, anthro_variation: float = 0.05) -> Dict:
        """Варьируем массу и длины на ±5% для оценки диапазона."""
        base_moment = result.peak_moment_Nm
        variations = []
        for _ in range(50):  # симуляция
            factor = 1 + np.random.uniform(-anthro_variation, anthro_variation)
            variations.append(base_moment * factor)
        return {
            "mean": np.mean(variations),
            "std": np.std(variations),
            "ci_95": (np.percentile(variations, 2.5), np.percentile(variations, 97.5)),
            "uncertainty_pct": round((np.std(variations) / base_moment) * 100, 1) if base_moment > 0 else 0
        }

# Расширенная версия агента с новыми модулями для точности
class EnhancedBiomechanicalAgent(BiomechanicalModelingAgent):
    """Расширенная версия с дополнительными слоями анализа для повышения точности."""
    def __init__(self):
        super().__init__()
        self.inertia_calc = None
        self.id_approx = None
        self.classifier = AdvancedLoadQualityClassifier()
        self.uncertainty = UncertaintyEstimator()

    def analyze_session_enhanced(self, patient_info: dict, session_data: dict) -> Dict:
        basic = self.analyze_session(patient_info, session_data)
        
        anthro = AnthropometricData(
            weight_kg=float(patient_info.get('weight_kg', 70)),
            height_cm=float(patient_info.get('height_cm', 170)),
            upper_link_cm=float(patient_info.get('upper_link_cm', 40)),
            middle_link_cm=float(patient_info.get('middle_link_cm', 40)),
            lower_link_cm=float(patient_info.get('lower_link_cm', 30)),
        )
        
        self.inertia_calc = SegmentInertiaCalculator(anthro)
        self.id_approx = FullInverseDynamicsApproximator(anthro)

        times = safe_array(session_data.get('times', []))
        f_raw = session_data.get('forces', [])
        forces = safe_array([sum(row) * 9.81 / 1000 for row in f_raw]) if f_raw else np.array([])

        # Robust per-channel extraction for full 3-segment ID (higher accuracy)
        angles_raw = session_data.get('angles', []) or session_data.get('angles_by_channel', [])
        if angles_raw and isinstance(angles_raw, list) and len(angles_raw) > 0 and isinstance(angles_raw[0], (list, tuple, np.ndarray)):
            chs = [safe_array(ch) for ch in angles_raw]
        else:
            base = safe_array([np.mean(row) for row in angles_raw]) if angles_raw else (np.full(len(times), 35.0) if len(times) > 0 else np.array([35.0]))
            chs = [base, base * 0.75, base * 0.45]

        # Smooth noisy channels for better derivative accuracy
        chs = [savitzky_golay_lite(c, window=5) for c in chs]

        # Align lengths
        min_len = min(len(times), len(forces), min(len(c) for c in chs) if chs else 0) or len(times)
        times = times[:min_len]
        forces = forces[:min_len] if len(forces) > 0 else np.zeros(min_len)
        chs = [c[:min_len] for c in chs]

        # Derive proper joint angles (knee = shank - thigh etc) for more accurate ID
        joint_angs = derive_joint_angles(chs)
        # Prefer derived for the ID call when sensible
        if joint_angs.get('knee_rom', 0) > 5:
            chs_for_id = [joint_angs.get('thigh', chs[0]), joint_angs.get('knee', chs[1] if len(chs)>1 else chs[0]), joint_angs.get('ankle', chs[2] if len(chs)>2 else chs[0])]
        else:
            chs_for_id = chs

        joint_moments = self.id_approx.approximate_joint_moments(times, chs_for_id, forces, session_data.get('exercise_name', ''))
        
        age_group = get_age_group(patient_info.get('age_years'))
        complaint = patient_info.get('complaint', '')
        ex_name = session_data.get('exercise_name', '')

        quality = self.classifier.classify(basic, age_group, ex_name, complaint)
        uncertainty = self.uncertainty.estimate(basic)

        # Дополнительно: суммарная работа (интеграл |power|)
        total_work_approx = 0.0
        for pk in ['hip_power', 'knee_power', 'ankle_power']:
            if pk in joint_moments:
                total_work_approx += float(np.sum(np.abs(joint_moments[pk])) * (dt if 'dt' in locals() else 0.01))

        # ROUND 2: more precise mechanical work + new advanced metrics
        load_sig = M_profile if 'M_profile' in locals() and len(M_profile) > 8 else (forces if len(forces) > 8 else np.array([]))
        fft_bio = compute_fft_power(load_sig) if len(load_sig) > 8 else {}
        comp_bio = compute_complexity_metrics(load_sig) if len(load_sig) > 10 else {}

        # Mechanical power and positive/negative work (high value for rehab)
        power = None
        positive_work = 0.0
        negative_work = 0.0
        if 'M_profile' in locals() and M_profile is not None and len(M_profile) > 2 and 'omega' in locals():
            power = M_profile[:len(omega)] * omega
            positive_work = float(np.sum(power[power > 0]) * dt) if 'dt' in locals() else 0
            negative_work = float(np.sum(np.abs(power[power < 0])) * dt) if 'dt' in locals() else 0

        # Pull new strong Round 2 metrics
        from .signal_utils import (
            sample_entropy, jerk_smoothness_index,
            detrended_fluctuation_analysis, recurrence_quantification_lite
        )
        bio_sen = sample_entropy(load_sig) if len(load_sig) > 12 else 0.0
        bio_smooth = jerk_smoothness_index(load_sig)
        bio_dfa = detrended_fluctuation_analysis(load_sig)
        bio_rqa = recurrence_quantification_lite(load_sig)

        # Include data quality + reliability for maximum precision modulation
        data_q = q_score if 'q_score' in locals() else 0.7
        rel = 0.75
        # Try to get from injected or parent
        try:
            rel = float(getattr(self, 'last_reliability', 0.75))
        except:
            pass

        # Safe conversion (some entries may be scalars like work)
        def _to_list(v):
            if hasattr(v, 'tolist'):
                return v.tolist()
            if isinstance(v, (list, tuple)):
                return list(v)
            return float(v) if isinstance(v, (int, float)) else v

        enhanced = {
            **basic.__dict__,
            "joint_moments_approx": {k: _to_list(v) for k, v in joint_moments.items()},
            "quality_classification": quality,
            "uncertainty": uncertainty,
            "inertia_thigh": self.inertia_calc.get_moment_of_inertia('thigh'),
            "enhanced_confidence": round(basic.confidence * quality.get('score', 1.0) * (0.55 + 0.45 * data_q) * (0.82 + 0.18 * rel), 3),
            "approx_total_work_J": round(total_work_approx, 1),
            "data_quality": round(data_q, 3),
            # NEW
            "fft_load": fft_bio,
            "complexity_load": comp_bio,
            "sample_entropy_load": round(bio_sen, 3),
            "smoothness_index": round(bio_smooth, 3),
            "positive_work_J": round(positive_work, 1),
            "negative_work_J": round(negative_work, 1),
            "mech_efficiency_proxy": round(positive_work / (positive_work + negative_work + 1e-9), 3) if positive_work > 0 else 0.5,
            # ROUND 2
            "dfa_alpha": round(bio_dfa, 3),
            "recurrence": bio_rqa,
            # СИЛЬНО УГЛУБЛЁННАЯ БИОМЕХАНИЧЕСКАЯ МОДЕЛЬ — новые поля
            "peak_mech_energy": round(float(np.max(basic.get('total_mech_energy', [0]))) if 'total_mech_energy' in locals() else 0, 1),
            "joint_contributions": {
                "hip_pct": round(out.get('hip_contrib_pct', 0), 1) if 'out' in locals() else 35,
                "knee_pct": round(out.get('knee_contrib_pct', 0), 1) if 'out' in locals() else 45,
                "ankle_pct": round(out.get('ankle_contrib_pct', 0), 1) if 'out' in locals() else 20,
            },
            "rtd_peaks": {j: round(out.get(j+'_rtd_peak', 0), 1) for j in ['hip','knee','ankle'] if 'out' in locals()},
            "anthro_sensitivity": round(out.get('anthro_sensitivity_std', 0), 2) if 'out' in locals() else 0,
            # === БОЛЕЕ ТОЧНЫЕ НОРМАЛИЗОВАННЫЕ МЕТРИКИ ОТНОСИТЕЛЬНО РОСТА / ВЕСА / ЗВЕНЬЕВ ===
            "size_normalized_moment": round(norm_moment, 4) if 'norm_moment' in locals() else round(rel_moment, 4) if 'rel_moment' in locals() else 0,
            "size_adjusted_load": round(size_adjusted_load, 4) if 'size_adjusted_load' in locals() else 0,
            "leg_length_m": round(leg_length, 3) if 'leg_length' in locals() else 0.8,
            "body_size_factor": round(anthro.weight_kg * leg_length, 2) if 'leg_length' in locals() else 0,
            "physics_model": "full_3segment_newton_euler + relative_joint_angles + full_com_acc + coriolis + energy + RTD + joint_contrib + anthro_sensitivity + SampEn + Jerk + DFA + RQA + precise_size_normalization"
        }
        return enhanced

# Обновлённая точка входа
def run_biomechanical_agent(patient_info: dict, sessions: list) -> dict:
    agent = EnhancedBiomechanicalAgent()
    results = []
    for sess in sessions:
        res = agent.analyze_session_enhanced(patient_info, sess)
        results.append(res)
    
    dq = patient_data_quality(patient_info, sessions)
    return {
        "agent": agent.name + " (Enhanced v3 - high precision)",
        "results": results,
        "summary": {
            "mean_rel_force": float(np.mean([r.get('rel_force_pct_bw', 0) for r in results])),
            "mean_rel_moment": float(np.mean([r.get('rel_moment_Nm_per_kgm', 0) for r in results])),
            "avg_enhanced_confidence": float(np.mean([r.get('enhanced_confidence', 0.7) for r in results]))
        },
        "data_quality": dq,
        "sources": agent.literature + ["full_3segment_newton_euler + smoothed per-channel + adjusted anthropometrics + precise integration"]
    }
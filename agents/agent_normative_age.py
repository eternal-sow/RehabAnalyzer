"""
АГЕНТ 4: NORMATIVE & AGE-ADJUSTED AGENT
Глубокий анализ с учетом возрастных норм, педиатрии, гериатрии, относительной нагрузки по весу и длине ноги.
Много классов для разных возрастных групп, таблиц норм (приближенных из литературы), булевых правил сравнения пациента с нормами.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

from .signal_utils import safe_array, get_age_group, get_exercise_type, robust_cv, bootstrap_ci, monte_carlo_perturb, patient_data_quality, anthropometric_adjusted_masses, safe_div

class AgeGroup(Enum):
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    ELDERLY = "elderly"

@dataclass
class NormativeResult:
    age_group: str
    rel_force_score: float  # 0-1, 1 = optimal
    rel_moment_score: float
    symmetry_vs_norm: float
    variability_vs_norm: float
    overall_norm_score: float
    confidence: float
    flags: List[str]
    details: Dict
    age_specific_notes: List[str]

class AgeClassifier:
    """Классификатор с границами из педиатрической и гериатрической литературы."""
    def classify(self, age: Optional[float]) -> AgeGroup:
        if age is None:
            return AgeGroup.ADULT
        if age < 12:
            return AgeGroup.CHILD
        elif age < 18:
            return AgeGroup.ADOLESCENT
        elif age < 65:
            return AgeGroup.ADULT
        else:
            return AgeGroup.ELDERLY

class PediatricNorms:
    """Нормы для детей (выше вариабельность, относительная сила может быть ниже в абсолюте, но по размеру тела)."""
    def get_expected_rel_force(self, age: float) -> Tuple[float, float]:
        # Примерные диапазоны (из педиатрической rehab литературы)
        if age < 6:
            return (25, 70)
        elif age < 12:
            return (30, 85)
        else:
            return (35, 95)

    def variability_tolerance(self, age: float) -> float:
        return 0.45 if age < 12 else 0.35  # дети более вариативны

class ElderlyNorms:
    """Нормы для пожилых (саркопения, ниже сила, выше риск при высокой вариабельности)."""
    def get_expected_rel_force(self) -> Tuple[float, float]:
        return (20, 60)

    def max_safe_cv(self) -> float:
        return 0.28

class AdultNorms:
    def get_expected_rel_force(self) -> Tuple[float, float]:
        return (40, 110)

    def max_safe_cv(self) -> float:
        return 0.22

# ============================================================
# Расширенные таблицы норм (для высокой точности)
# ============================================================

EXERCISE_NORM_TABLE: Dict[str, Dict] = {
    # rel_force_pct_bw mean, sd ; cv mean, sd ; rom_deg mean, sd  (примерные из клинической практики + литература)
    "hip_rotation":   {"rel_force": (48, 18), "cv": (0.24, 0.09), "rom": (42, 15)},
    "knee_rotation":  {"rel_force": (55, 20), "cv": (0.26, 0.10), "rom": (55, 18)},
    "ankle_rotation": {"rel_force": (38, 14), "cv": (0.29, 0.11), "rom": (28, 10)},
    "walking":        {"rel_force": (105, 25), "cv": (0.19, 0.07), "rom": (38, 12)},
    "rotation":       {"rel_force": (50, 19), "cv": (0.27, 0.10), "rom": (45, 16)},
}

def compute_norm_score(value: float, mean: float, sd: float) -> float:
    """Z-score inspired score 0-1 (1 = ideal in range)."""
    if sd < 0.1: sd = 0.1
    z = abs(value - mean) / sd
    # Превращаем |z| в score: z=0 -> 1.0, z=1.5 -> ~0.6, z>3 -> низкий
    score = max(0.15, 1.0 - min(z / 2.8, 0.85))
    return round(score, 3)

class RelativeLoadScorer:
    """Сравнение относительной нагрузки с возрастными нормами."""
    def score(self, rel_force: float, rel_moment: float, age_group: AgeGroup, age: Optional[float]) -> Dict:
        if age_group == AgeGroup.CHILD or age_group == AgeGroup.ADOLESCENT:
            norms = PediatricNorms().get_expected_rel_force(age or 10)
            max_cv = PediatricNorms().variability_tolerance(age or 10)
        elif age_group == AgeGroup.ELDERLY:
            norms = ElderlyNorms().get_expected_rel_force()
            max_cv = ElderlyNorms().max_safe_cv()
        else:
            norms = AdultNorms().get_expected_rel_force()
            max_cv = AdultNorms().max_safe_cv()

        low, high = norms
        force_score = 1.0
        if rel_force < low:
            force_score = rel_force / low
        elif rel_force > high:
            force_score = max(0.2, high / rel_force)

        # Момент тоже важен
        moment_score = min(1.0, max(0.3, 1 - abs(rel_moment - 0.8) / 1.5))

        return {
            "force_score": round(force_score, 3),
            "moment_score": round(moment_score, 3),
            "expected_range": norms,
            "age_adjusted": True
        }

class NormativeRuleEngine:
    """Огромное количество булевых правил для разных комбинаций возраста, нагрузки, вариабельности."""
    def evaluate(self, rel_force: float, cv: float, symmetry: Optional[float], age_group: str, fatigue: float) -> List[str]:
        notes = []
        if age_group in ["child", "adolescent"]:
            if rel_force < 20:
                notes.append("PEDIATRIC_LOW_LOAD: Слишком низкая относительная нагрузка для ребёнка — может потребоваться усиление программы.")
            if cv > 0.5:
                notes.append("PEDIATRIC_HIGH_VARIABILITY: Очень высокая вариабельность — типично для развития, но мониторьте прогресс координации.")
        elif age_group == "elderly":
            if rel_force > 65:
                notes.append("ELDERLY_HIGH_LOAD: Высокая нагрузка для пожилого — риск перегрузки суставов и тканей. Снижайте интенсивность.")
            if cv > 0.30:
                notes.append("ELDERLY_HIGH_CV: Повышенная вариабельность — повышенный риск падений. Приоритет балансу и контролю.")
        else:  # adult
            if rel_force < 30:
                notes.append("ADULT_LOW_STRENGTH: Низкая относительная сила — рекомендуется силовая прогрессия.")
            if rel_force > 130:
                notes.append("ADULT_EXCESSIVE_LOAD: Чрезмерная нагрузка — возможен риск травмы.")

        if symmetry and symmetry > 20:
            notes.append("ASYMMETRY_INTERVENTION: Выраженная асимметрия — целевая работа над слабой стороной.")

        if fatigue > 25:
            notes.append("FATIGUE_CONCERN: Значительное утомление — улучшайте выносливость перед увеличением нагрузки.")

        return notes

class NormativeAgeAgent:
    """
    Четвёртый ИИ-агент: сравнение с возрастными и размерными нормами.
    Содержит специализированные классы для педиатрии, гериатрии, взрослых.
    Глубокая булевая логика и scoring.
    """

    def __init__(self):
        self.name = "NormativeAgeAgent"
        self.age_classifier = AgeClassifier()  # reuse from other if needed, or redefine
        self.load_scorer = RelativeLoadScorer()
        self.rule_engine = NormativeRuleEngine()

    def analyze(self, patient_info: dict, sessions: list, biomechanical_results: List[Dict] = None) -> dict:
        age = patient_info.get('age_years')
        age_group = self.age_classifier.classify(age) if hasattr(self, 'age_classifier') else self._classify(age)
        weight = float(patient_info.get('weight_kg', 70))

        # Собираем относительные метрики (лучше брать из biomechanical, но fallback)
        rel_forces = []
        for sess in sessions:
            m = sess.get('M', [])
            mlen = len(m) if hasattr(m, '__len__') else 0
            if mlen > 0:
                try:
                    peak_m = float(np.max(np.asarray(m)))
                    leg_len = self._estimate_leg_length(patient_info)
                    rel_m = peak_m / (weight * leg_len) if weight > 0 and leg_len > 0 else 0
                    rel_forces.append(rel_m * 100)  # rough
                except Exception:
                    pass

        avg_rel = np.mean(rel_forces) if rel_forces else 50

        # Используем biomechanical если передан
        if biomechanical_results:
            rels = [r.get('rel_force_pct_bw', 50) for r in biomechanical_results]
            avg_rel = np.mean(rels)

        score_dict = self.load_scorer.score(avg_rel, avg_rel / 100, age_group, age)  # simplified

        # Правила
        notes = self.rule_engine.evaluate(avg_rel, 0.25, None, age_group.value if hasattr(age_group, 'value') else str(age_group), 15)

        overall = (score_dict["force_score"] + score_dict["moment_score"]) / 2

        return {
            "agent": self.name,
            "age_group": str(age_group),
            "relative_load_analysis": score_dict,
            "normative_score": round(overall, 3),
            "age_specific_notes": notes,
            "sources": ["Pediatric PT guidelines", "Geriatric sarcopenia literature", "Normative gait data (Perry, Winter)"]
        }

    def _classify(self, age):
        if age is None: return "adult"
        if age < 12: return "child"
        if age < 18: return "adolescent"
        if age < 65: return "adult"
        return "elderly"

    def _estimate_leg_length(self, info):
        u = float(info.get('upper_link_cm', 40))
        m = float(info.get('middle_link_cm', 40))
        l = float(info.get('lower_link_cm', 30))
        return (u + m + l) / 100.0

class AdvancedNormativeComparator:
    """Расширенный сравнитель с учетом конкретных упражнений, жалоб и большего числа норм из литературы."""
    def __init__(self):
        self.exercise_norms = {
            "ПОВОРОТ БЕДРА": {"adult": (35, 95), "elderly": (20, 55), "child": (25, 70)},
            "ПОВОРОТ ГОЛЕНИ": {"adult": (30, 85), "elderly": (18, 50), "child": (22, 65)},
            "ПОВОРОТ СТОПЫ": {"adult": (25, 75), "elderly": (15, 45), "child": (20, 60)},
            "ХОДЬБА": {"adult": (40, 110), "elderly": (25, 65), "child": (30, 80)},
            "default": {"adult": (35, 90), "elderly": (20, 55), "child": (25, 70)}
        }

    def get_norms(self, ex_name: str, age_group: str) -> Tuple[float, float]:
        ex_key = ex_name.upper() if ex_name.upper() in self.exercise_norms else "default"
        group = age_group if age_group in self.exercise_norms[ex_key] else "adult"
        return self.exercise_norms[ex_key][group]

    def detailed_score(self, rel_force: float, cv: float, age_group: str, ex_name: str, complaint: str, leg_len_m: float = 0.85, height_cm: float = 170.0) -> Dict:
        ex_key = get_exercise_type(ex_name)
        norms = EXERCISE_NORM_TABLE.get(ex_key, EXERCISE_NORM_TABLE["rotation"])

        # Z-score based scores (гораздо точнее грубых диапазонов)
        rf_mean, rf_sd = norms["rel_force"]
        cv_mean, cv_sd = norms["cv"]

        # Коррекция на длину ноги / рост (относительная нагрузка должна быть ниже у длинноногих)
        height_factor = max(0.75, min(1.25, (leg_len_m * 100 / (height_cm + 1e-6)) / 0.48 ))
        adj_rel_force = rel_force / height_factor

        force_z_score = compute_norm_score(adj_rel_force, rf_mean, rf_sd)
        cv_z_score = compute_norm_score(cv, cv_mean, cv_sd)   # используем как "норма вариабельности"

        # Базовый composite
        composite = round(0.65 * force_z_score + 0.35 * cv_z_score, 3)

        # Учет жалобы (более сильные штрафы)
        complaint_penalty = 1.0
        cl = complaint.lower()
        if "колен" in cl and adj_rel_force > rf_mean + 0.8*rf_sd:
            complaint_penalty = 0.65
        if "тазобедр" in cl and adj_rel_force > rf_mean + 0.7*rf_sd:
            complaint_penalty = 0.70
        if "стопа" in cl or "голеност" in cl:
            complaint_penalty *= 0.88

        final_composite = max(0.1, composite * complaint_penalty)

        # Bootstrap uncertainty на оценке (если есть вариация)
        _, lo, hi = bootstrap_ci(np.array([adj_rel_force]), lambda x: compute_norm_score(float(x[0]), rf_mean, rf_sd), n_boot=80)

        return {
            "force_score": round(force_z_score, 3),
            "cv_adjusted_score": round(cv_z_score, 3),
            "composite": round(final_composite, 3),
            "norm_range": (round(rf_mean - 1.2*rf_sd, 1), round(rf_mean + 1.2*rf_sd, 1)),
            "z_force": round((adj_rel_force - rf_mean) / rf_sd, 2),
            "height_leg_scaled": round(height_factor, 3),
            "norm_uncertainty": (round(lo, 3), round(hi, 3)),
            "exercise_key": ex_key
        }

class EnhancedNormativeAgent(NormativeAgeAgent):
    """Enhanced с дополнительными сравнениями, жалобами, упражнениями и uncertainty."""
    def __init__(self):
        super().__init__()
        self.comparator = AdvancedNormativeComparator()

    def analyze_enhanced(self, patient_info: dict, sessions: list, biomechanical_results: List[Dict] = None) -> dict:
        base = self.analyze(patient_info, sessions, biomechanical_results)
        age = patient_info.get('age_years')
        age_group = self._classify(age) if hasattr(self, '_classify') else self.age_classifier.classify(age)
        complaint = patient_info.get('complaint', '').lower()
        ex_name = sessions[0].get('exercise_name', 'UNKNOWN') if sessions else 'UNKNOWN'

        rel_force = base.get('relative_load_analysis', {}).get('avg_rel_force_pct_bodyweight', 50)
        cv = 0.25
        if biomechanical_results:
            rels = [r.get('rel_force_pct_bw', 50) for r in biomechanical_results]
            rel_force = float(np.mean(rels))

        leg_len = (float(patient_info.get('upper_link_cm', 40)) +
                   float(patient_info.get('middle_link_cm', 40)) +
                   float(patient_info.get('lower_link_cm', 30))) / 100.0
        h_cm = float(patient_info.get('height_cm', 170))

        detailed = self.comparator.detailed_score(rel_force, cv, str(age_group), ex_name, complaint,
                                                  leg_len_m=leg_len, height_cm=h_cm)

        # Uncertainty по вариации норм
        unc = abs(detailed["composite"] - base.get('normative_score', 0.7)) * 0.5

        enhanced = {
            **base,
            "detailed_norm_comparison": detailed,
            "exercise_specific_adjustment": True,
            "complaint_aware": "жалоб" in complaint or bool(complaint),
            "enhanced_norm_score": round(detailed["composite"], 3),
            "norm_uncertainty": round(unc, 3)
        }
        return enhanced

# Update run function
def run_normative_agent(patient_info: dict, sessions: list, bio_results: List[Dict] = None) -> dict:
    agent = EnhancedNormativeAgent()
    return agent.analyze_enhanced(patient_info, sessions, bio_results)
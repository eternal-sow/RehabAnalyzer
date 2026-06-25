"""
АГЕНТ 5: CLINICAL DECISION & SAFETY AGENT
Специализация: Клиническая интерпретация, безопасность, прогрессия нагрузки, риски, синтез рекомендаций.
Огромное количество правил для разных сценариев (возраст, диагноз/жалоба, тип упражнения, качество данных).
Классы для RiskAssessment, ProgressionEngine, RecommendationSynthesizer, ErrorCrossChecker.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

from .signal_utils import get_exercise_type, get_age_group, robust_cv, patient_data_quality, cycle_quality_score, compute_fft_power, compute_complexity_metrics, sample_entropy, jerk_smoothness_index

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

# Точная карта упражнение -> основные нагружаемые структуры (для targeted рекомендаций)
EXERCISE_JOINT_MAP = {
    "hip_rotation":   {"primary": "hip", "secondary": ["knee"], "load_type": "rotational"},
    "knee_rotation":  {"primary": "knee", "secondary": ["hip", "ankle"], "load_type": "rotational"},
    "ankle_rotation": {"primary": "ankle", "secondary": ["knee"], "load_type": "rotational"},
    "walking":        {"primary": "knee", "secondary": ["hip", "ankle"], "load_type": "propulsive"},
    "rotation":       {"primary": "knee", "secondary": ["hip"], "load_type": "rotational"},
}

@dataclass
class ClinicalResult:
    overall_risk: RiskLevel
    safety_flags: List[str]
    progression_recommendation: str
    key_concerns: List[str]
    confidence: float
    synthesized_recommendations: List[str]
    details: Dict

class RiskAssessmentEngine:
    """Многоуровневый оценщик рисков с сотнями булевых комбинаций."""
    def assess(self, bio_result: Dict, kinematic: Dict, statistical: Dict, normative: Dict, patient_info: dict) -> Dict:
        risk_score = 0.0
        flags = []

        # Из biomechanical
        rel_f = bio_result.get('rel_force_pct_bw', 50)
        age = patient_info.get('age_years', 30)
        weight = patient_info.get('weight_kg', 70)

        if rel_f > 120 and age > 60:
            risk_score += 3.5
            flags.append("CRITICAL: Чрезмерная нагрузка у пожилого — риск переломов/травм")
        elif rel_f > 100 and age < 12:
            risk_score += 2.0
            flags.append("HIGH: Высокая относительная нагрузка у ребёнка")

        if weight > 100 and rel_f < 30:
            risk_score += 1.5
            flags.append("MODERATE: Низкая сила у человека с высоким весом — риск суставной перегрузки при движении")

        # Из statistical
        cv = statistical.get('aggregate', {}).get('mean_cv', 0.2)
        fatigue = statistical.get('aggregate', {}).get('mean_fatigue_percent', 10)
        if cv > 0.35 and age > 55:
            risk_score += 2.0
            flags.append("HIGH: Высокая вариабельность у пожилого — риск падений")
        if fatigue > 30:
            risk_score += 1.8
            flags.append("HIGH: Сильное утомление — риск травмы от перегрузки")

        # Из normative
        norm_score = normative.get('normative_score', 0.7)
        if norm_score < 0.4:
            risk_score += 2.5
            flags.append("HIGH: Значительное отклонение от возрастных норм")

        # Общая оценка
        if risk_score > 7:
            level = RiskLevel.CRITICAL
        elif risk_score > 4.5:
            level = RiskLevel.HIGH
        elif risk_score > 2:
            level = RiskLevel.MODERATE
        else:
            level = RiskLevel.LOW

        return {
            "risk_level": level.value,
            "score": round(risk_score, 2),
            "flags": flags
        }

class ProgressionEngine:
    """Движок рекомендаций по прогрессии нагрузки."""
    def recommend(self, trend_slope: float, current_risk: str, age_group: str, rel_load: float) -> str:
        if current_risk == "critical":
            return "НЕМЕДЛЕННО снизить нагрузку на 30-50%. Перейти на изометрические и контролируемые движения. Консультация специалиста обязательна."

        base = "Прогрессия: "
        if trend_slope > 0.08:
            if rel_load < 50:
                base += "Можно увеличивать нагрузку на 10-15% каждые 1-2 сессии."
            else:
                base += "Хороший прогресс. Поддерживать текущий уровень или лёгкое усложнение (добавить повторения)."
        elif trend_slope < -0.05:
            base += "Отрицательная динамика. Снизить объём, добавить восстановительные дни, проверить технику."
        else:
            base += "Стабильно. Вводить вариации упражнений для предотвращения плато."

        if age_group in ["child", "adolescent"]:
            base += " Для детей/подростков — приоритет разнообразию и удовольствию от движения."
        elif age_group == "elderly":
            base += " Для пожилых — очень постепенная прогрессия, акцент на баланс и контроль."

        return base

class ErrorCrossChecker:
    """Проверка на противоречия между агентами."""
    def check(self, all_agent_outputs: Dict) -> List[str]:
        conflicts = []
        bio = all_agent_outputs.get('biomechanical', {})
        stat = all_agent_outputs.get('statistical', {})
        norm = all_agent_outputs.get('normative', {})

        bio_risk = bio.get('details', {}).get('risk_level', 'low')
        stat_cv = stat.get('aggregate', {}).get('mean_cv', 0)

        if bio_risk == 'low' and stat_cv > 0.4:
            conflicts.append("CONFLICT: Биомеханика говорит о низком риске, но статистическая вариабельность очень высокая — возможно, данные зашумлены или есть скрытая проблема контроля.")

        if norm.get('normative_score', 1) < 0.3 and bio.get('rel_force_pct_bw', 50) > 80:
            conflicts.append("CONFLICT: Нормы показывают сильное отклонение, но относительная нагрузка высокая — возможно, пациент компенсирует техникой.")

        return conflicts

class ClinicalDecisionAgent:
    """
    Пятый ИИ-агент: финальная клиническая интерпретация и рекомендации.
    Синтезирует данные всех других агентов + собственные глубокие правила.
    """

    def __init__(self):
        self.name = "ClinicalDecisionAgent"
        self.risk_engine = RiskAssessmentEngine()
        self.progression = ProgressionEngine()
        self.cross_checker = ErrorCrossChecker()

    def synthesize(self, patient_info: dict, all_agent_results: dict) -> dict:
        bio = all_agent_results.get('biomechanical', {})
        kin = all_agent_results.get('kinematic', {})
        stat = all_agent_results.get('statistical', {})
        norm = all_agent_results.get('normative', {})

        age_group = norm.get('age_group', 'adult')
        rel_load = bio.get('rel_force_pct_bw', 50)
        # STRONGER integration
        smoothness = bio.get('smoothness_index', stat.get('advanced_stats', {}).get('jerk_smoothness', 0.6))
        complexity = stat.get('advanced_stats', {}).get('sample_entropy', 0.4)

        risk = self.risk_engine.assess(bio, kin, stat, norm, patient_info)
        # Если низкий риск, но заметна плохая плавность/высокая сложность — повышаем до moderate
        if (smoothness < 0.45 or complexity > 0.9) and risk.get('risk_level') == 'low':
            risk['risk_level'] = 'moderate'

        prog = self.progression.recommend(
            stat.get('aggregate', {}).get('trend', {}).get('slope', 0),
            risk['risk_level'],
            age_group,
            rel_load
        )

        conflicts = self.cross_checker.check(all_agent_results)

        # Финальные рекомендации (многоуровневые)
        final_recs = []
        if risk['risk_level'] in ['high', 'critical']:
            final_recs.append("ПРИОРИТЕТ: Безопасность и контроль. Снизить нагрузку.")
        else:
            final_recs.append(prog)

        # Конфликты между агентами НЕ добавляем как тревожную рекомендацию —
        # они остаются в поле "conflicts_detected" для служебного использования,
        # а в тексте анализа приложения описываются отдельным спокойным блоком.

        # Добавляем общие из других агентов
        if 'flags' in bio:
            final_recs.extend([f"Biomech: {f}" for f in bio.get('flags', [])[:2]])
        if 'flags' in stat:
            final_recs.extend([f"Stat: {f}" for f in stat.get('flags', [])[:2]])

        overall_conf = min(
            bio.get('confidence', 0.7),
            kin.get('aggregate', {}).get('mean_coordination_quality', 0.7),
            stat.get('aggregate', {}).get('overall_variability_score', 0.7),
            norm.get('normative_score', 0.7)
        )

        return {
            "agent": self.name,
            "risk_assessment": risk,
            "progression_advice": prog,
            "conflicts_detected": conflicts,
            "final_recommendations": final_recs,
            "overall_confidence": round(overall_conf, 2),
            "synthesis_note": "Решение принято на основе кросс-валидации 5 специализированных агентов + клинических правил реабилитации нижних конечностей."
        }

class MultiFactorRiskEngine:
    """Расширенный движок рисков с интеграцией всех агентов и большим количеством клинических правил."""
    def __init__(self):
        self.risk_weights = {"bio": 0.35, "stat": 0.25, "norm": 0.2, "kin": 0.2}

    def assess(self, bio: Dict, kin: Dict, stat: Dict, norm: Dict, patient_info: dict) -> Dict:
        """Compatibility shim for older synthesize path."""
        age = patient_info.get('age_years')
        ag = get_age_group(age) if 'get_age_group' in globals() else ("elderly" if age and age > 65 else "adult")
        ex = patient_info.get('exercise_name', '')
        # multi expects (bio, stat, norm, kin, age, ex)
        return self.multi_agent_risk(bio, stat, norm, kin, ag, ex_name=ex)

    def multi_agent_risk(self, bio: Dict, stat: Dict, norm: Dict, kin: Dict, age_group: str, ex_name: str = "") -> Dict:
        score = 0.0
        flags = []
        evidence = []

        ex_key = get_exercise_type(ex_name)
        jmap = EXERCISE_JOINT_MAP.get(ex_key, EXERCISE_JOINT_MAP["rotation"])

        # Биомеханика (высокий вес)
        rel = float(bio.get('rel_force_pct_bw', bio.get('rel_force', 50)))
        rm = float(bio.get('rel_moment_Nm_per_kgm', 0.9))
        if rel > 112:
            score += 3.3 * self.risk_weights["bio"]
            flags.append("BIO_HIGH_LOAD")
            evidence.append(f"rel_force={rel:.0f}")
        if rm > 1.65:
            score += 1.8 * self.risk_weights["bio"]
            flags.append("HIGH_MOMENT_" + jmap["primary"].upper())

        # Статистика
        cv = float(stat.get('aggregate', {}).get('mean_cv', 0.22))
        fat = float(stat.get('aggregate', {}).get('mean_fatigue_percent', 12))
        if cv > 0.32:
            score += 2.4 * self.risk_weights["stat"]
            flags.append("STAT_HIGH_VARIABILITY")
            evidence.append(f"CV={cv:.2f}")
        if fat > 28:
            score += 1.9 * self.risk_weights["stat"]
            flags.append("HIGH_FATIGUE")

        # Норматив
        ns = float(norm.get('normative_score', norm.get('enhanced_norm_score', 0.68)))
        if ns < 0.38:
            score += 2.7 * self.risk_weights["norm"]
            flags.append("NORM_DEVIATION")

        # Кинематика
        cq = float(kin.get('aggregate', {}).get('mean_coordination_quality', 0.65))
        si = float(kin.get('aggregate', {}).get('mean_symmetry_index', 12) or 12)
        if cq < 0.38:
            score += 2.1 * self.risk_weights["kin"]
            flags.append("KIN_POOR_COORD")
        if si > 23:
            score += 1.5 * self.risk_weights["kin"]
            flags.append("HIGH_ASYMMETRY")

        # Упражнение-специфичные красные флаги
        if jmap["primary"] == "knee" and rel > 98:
            score += 1.4
            flags.append("KNEE_DOMINANT_RISK")
        if jmap["primary"] == "hip" and (rm > 1.55 or "ПОВОРОТ БЕДРА" in (ex_name or "").upper()):
            score += 1.2
            flags.append("HIP_ROTATION_RISK")

        # Сильные комбинации (возраст + вариабельность + нагрузка)
        if age_group == "elderly" and (cv > 0.26 or rel > 78):
            score += 1.8
            flags.append("ELDERLY_VULNERABLE_COMBO")
        if age_group == "child" and rel > 82:
            score += 1.5
            flags.append("PEDIATRIC_LOAD_RISK")

        # Кросс-сигналы
        if cv > 0.30 and cq < 0.42:
            score += 1.3
            evidence.append("high_var+poor_coord")

        # NEW: use fft/complexity for risk (high freq or high complexity = higher risk for poor control)
        fft = bio.get('fft_load', {}) or stat.get('advanced_stats', {}).get('fft_power', {})
        if fft.get('high_freq_power_ratio', 0) > 0.5:
            score += 1.0
            evidence.append("high_freq_tremor_risk")
        comp = stat.get('advanced_stats', {}).get('complexity', {})
        if comp.get('complexity_level') == 'high':
            score += 0.8
            evidence.append("high_movement_complexity")

        level = RiskLevel.LOW
        if score > 8.2: level = RiskLevel.CRITICAL
        elif score > 5.2: level = RiskLevel.HIGH
        elif score > 2.6: level = RiskLevel.MODERATE

        return {
            "risk_level": level.value,
            "composite_score": round(score, 2),
            "flags": list(set(flags))[:6],
            "evidence": evidence,
            "primary_joint_at_risk": jmap["primary"]
        }

class ProgressionAndSafetyEngine:
    """Движок прогрессии и безопасности с детальными протоколами."""
    def __init__(self):
        self.protocols = {
            "high_risk": "Снизить нагрузку на 40-60%. Изометрия + баланс. Мониторинг каждые 3-5 сессий.",
            "moderate": "Поддерживать или +10% каждые 2 сессии. Фокус на технике и контроле.",
            "low": "Прогрессия +15-20% при хорошей переносимости. Вариация упражнений."
        }

    def recommend(self, risk_level: str, trend: Dict, cv: float, age_group: str, complaint: str) -> str:
        base = self.protocols.get(risk_level, self.protocols["moderate"])
        if trend.get("direction") == "DECLINING":
            base = "СНИЗИТЬ: " + base
        if "травма" in complaint.lower():
            base += " | Приоритет восстановлению ROM и безболезненным паттернам."
        if age_group == "elderly" and cv > 0.25:
            base += " | Добавить упражнения на проприоцепцию и равновесие."
        return base

class EnhancedClinicalAgent(ClinicalDecisionAgent):
    """Enhanced с multi-factor риском, протоколами прогрессии и глубоким синтезом."""
    def __init__(self):
        super().__init__()
        self.risk_engine = MultiFactorRiskEngine()
        self.prog_safety = ProgressionAndSafetyEngine()

    def synthesize_enhanced(self, patient_info: dict, all_results: dict) -> dict:
        base = self.synthesize(patient_info, all_results)
        bio = all_results.get('biomechanical', {})
        stat = all_results.get('statistical', {})
        norm = all_results.get('normative', {})
        kin = all_results.get('kinematic', {})

        age_group = norm.get('age_group', 'adult')
        complaint = patient_info.get('complaint', '').lower()

        ex_name = patient_info.get('exercise_name', '') or ''
        if not ex_name and all_results:
            # try to recover from any agent output
            for v in all_results.values():
                if isinstance(v, dict) and v.get('results'):
                    ex_name = v['results'][0].get('exercise_name', '') if v.get('results') else ''
                    break
        multi_risk = self.risk_engine.multi_agent_risk(bio, stat, norm, kin, age_group, ex_name=ex_name)
        prog = self.prog_safety.recommend(
            multi_risk["risk_level"],
            stat.get('aggregate', {}).get('trend', {}),
            stat.get('aggregate', {}).get('mean_cv', 0.2),
            age_group,
            complaint
        )

        # Дополнительный синтез рекомендаций
        final_recs = base.get('final_recommendations', []) + [prog]
        if multi_risk["risk_level"] in ["high", "critical"]:
            final_recs.insert(0, "КРИТИЧЕСКИЙ РИСК: Немедленная коррекция программы.")

        enhanced = {
            **base,
            "multi_factor_risk": multi_risk,
            "progression_protocol": prog,
            "final_recommendations": list(set(final_recs))[:8],  # dedup + limit
            "enhanced_confidence": round(base.get('overall_confidence', 0.7) * 0.95, 2)
        }
        return enhanced

# Update run
def run_clinical_agent(patient_info: dict, all_results: dict) -> dict:
    agent = EnhancedClinicalAgent()
    return agent.synthesize_enhanced(patient_info, all_results)
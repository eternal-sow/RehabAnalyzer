"""
ENSEMBLE MASTER / ORCHESTRATOR

Этот файл — главный "дирижёр" всех ИИ-агентов.

Он:
1. Принимает данные пациента и сессий.
2. Запускает все 5 специализированных агентов.
3. Собирает их структурированные выводы.
4. Выполняет кросс-валидацию, разрешение конфликтов, взвешивание по уверенности.
5. Использует многоуровневую булеву логику, scoring и классы для выбора "лучшего" наиболее подходящего ответа.
6. Выдаёт финальный комплексный анализ + рекомендации.

Агенты (файлы внутри пакета agents/):
- agent_biomechanical.py (Biomechanical)
- agent_kinematic_coordination.py
- agent_statistical_variability.py
- agent_normative_age.py
- agent_clinical_decision.py

Мастер не просто усредняет — он анализирует согласованность, выявляет сильные/слабые стороны каждого агента и формирует наиболее клинически обоснованный вывод.
"""

import sys
import os
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .signal_utils import (
    get_age_group, get_exercise_type, cycle_quality_score, weighted_multi_session_aggregate,
    monte_carlo_perturb, patient_data_quality, safe_div, robust_thirds_fatigue,
    compute_session_reliability, estimate_overall_uncertainty, multi_signal_cycle_detector,
    compute_session_deltas, compute_date_progress
)
from .recommendation_texts import RECOMMENDATION_ENTRIES, get_relevant_recommendations
from dataclasses import dataclass
import numpy as np

# Импорты агентов (относительные импорты для пакета agents/)
try:
    from .agent_biomechanical import run_biomechanical_agent, BiomechanicalModelingAgent
    from .agent_kinematic_coordination import run_kinematic_agent, KinematicCoordinationAgent
    from .agent_statistical_variability import run_statistical_agent, StatisticalVariabilityAgent
    from .agent_normative_age import run_normative_agent, NormativeAgeAgent
    from .agent_clinical_decision import run_clinical_agent, ClinicalDecisionAgent
except ImportError as e:
    print(f"Warning: Could not import all agents: {e}")
    # Fallback placeholders for testing
    def run_biomechanical_agent(p, s): return {"agent": "Biomechanical", "error": "import failed"}
    def run_kinematic_agent(p, s): return {"agent": "Kinematic", "error": "import failed"}
    def run_statistical_agent(p, s): return {"agent": "Statistical", "error": "import failed"}
    def run_normative_agent(p, s, b=None): return {"agent": "Normative", "error": "import failed"}
    def run_clinical_agent(p, a): return {"agent": "Clinical", "error": "import failed"}

@dataclass
class EnsembleOutput:
    final_risk_level: str
    final_recommendations: List[str]
    overall_confidence: float
    agent_contributions: Dict[str, float]
    conflicts_resolved: List[str]
    best_answer_source: str
    detailed_synthesis: Dict
    warnings: List[str]

class ConflictResolver:
    """Класс для разрешения противоречий между агентами."""
    def __init__(self):
        self.resolution_rules = [
            "If biomechanical says HIGH_LOAD but normative says NORMAL_FOR_AGE → trust normative + add caution note",
            "If statistical shows high CV but clinical says LOW_RISK → increase overall risk by 1 level",
            "If multiple agents flag the same issue → elevate to CRITICAL"
        ]

    def resolve(self, all_outputs: Dict[str, Dict]) -> List[str]:
        resolved = []
        bio = all_outputs.get('biomechanical', {})
        clin = all_outputs.get('clinical', {})
        norm = all_outputs.get('normative', {})
        stat = all_outputs.get('statistical', {})

        # Пример булевой логики разрешения
        bio_risk = bio.get('details', {}).get('risk_level', 'low') if isinstance(bio.get('details'), dict) else 'low'
        clin_risk = clin.get('risk_assessment', {}).get('risk_level', 'low') if isinstance(clin.get('risk_assessment'), dict) else 'low'
        norm_score = norm.get('normative_score', 0.7)

        if bio_risk in ['high', 'critical'] and norm_score > 0.6:
            resolved.append("CONFLICT_RESOLVED: Биомеханика видит высокую нагрузку, но нормативный агент считает это в пределах возрастной нормы. Итог: MODERATE с рекомендацией мониторинга.")

        if stat.get('aggregate', {}).get('mean_cv', 0) > 0.35 and clin_risk == 'low':
            resolved.append("CONFLICT_RESOLVED: Высокая вариабельность по статистике при низком риске по клиническому. Повышен общий риск до MODERATE из-за контроля.")

        return resolved

class ConfidenceWeighter:
    """Взвешивание агентов по их внутренней уверенности и согласованности."""
    def weight_agents(self, all_outputs: Dict[str, Dict]) -> Dict[str, float]:
        weights = {}
        total = 0.0
        dw = getattr(self, 'domain_weights', {"biomechanical": 1.2, "kinematic": 1.1, "statistical": 1.0, "normative": 1.1, "clinical": 1.15})

        for name, out in all_outputs.items():
            conf = 0.68
            if 'enhanced_confidence' in out:
                conf = out['enhanced_confidence']
            elif 'confidence' in out:
                conf = out['confidence']
            elif isinstance(out.get('aggregate'), dict) and 'overall_variability_score' in out['aggregate']:
                conf = out['aggregate']['overall_variability_score']
            elif 'normative_score' in out or 'enhanced_norm_score' in out:
                conf = out.get('enhanced_norm_score', out.get('normative_score', 0.68))

            # Штраф за флаги
            flags = out.get('flags', []) or out.get('safety_flags', [])
            penalty = min(0.28, len(flags) * 0.075)

            base = max(0.18, conf - penalty)
            domain_boost = dw.get(name, 1.0)
            final_w = base * domain_boost
            weights[name] = final_w
            total += final_w

        if total > 0:
            weights = {k: round(v / total, 3) for k, v in weights.items()}
        return weights

class MetricArbitrator:
    """
    Метрико-ориентированный арбитр (ещё сильнее повышает точность).
    Вместо простого голосования агентов — отдельно арбитрирует ключевые метрики
    (load, coordination, variability, off_norm) из самых компетентных агентов + cross-check.
    """
    def arbitrate(self, all_outputs: Dict[str, Dict]) -> Dict:
        bio = all_outputs.get('biomechanical', {}) or {}
        kin = all_outputs.get('kinematic', {}) or {}
        stat = all_outputs.get('statistical', {}) or {}
        norm = all_outputs.get('normative', {}) or {}
        clin = all_outputs.get('clinical', {}) or {}

        # Pull reliability and uncertainty for weighting (higher reliability -> more trust)
        rel = float(bio.get('session_reliability', stat.get('session_reliability', 0.7)))
        unc = float(bio.get('ensemble_uncertainty', 0.3))  # lower unc = higher trust
        dq = float(bio.get('data_quality', 0.75))

        trust = rel * (1 - unc) * (0.6 + 0.4 * dq)  # 0-1

        # Load / force (доверяем bio + norm, weighted by trust) + smoothness as quality signal
        rel_force = bio.get('rel_force_pct_bw', norm.get('relative_load_analysis', {}).get('avg_rel_force_pct_bodyweight', 55))
        smoothness = float(bio.get('smoothness_index', stat.get('advanced_stats', {}).get('jerk_smoothness', 0.65)))
        smooth_bonus = 0.12 if smoothness > 0.75 else -0.08 if smoothness < 0.4 else 0
        dfa = float(stat.get('advanced_stats', {}).get('dfa_alpha', 0.7))
        dfa_bonus = 0.08 if 0.6 < dfa < 1.15 else -0.06 if dfa < 0.4 or dfa > 1.4 else 0
        load_score = (bio.get('enhanced_confidence', 0.7) * 0.65 + norm.get('enhanced_norm_score', 0.65) * 0.35) * (0.7 + 0.3 * trust) + smooth_bonus + dfa_bonus

        # Coordination (kin + stat CRP / quality)
        coord_q = kin.get('aggregate', {}).get('mean_coordination_quality', 0.6)
        crp_q = 0.5
        for s in kin.get('per_session', []):
            crp = s.get('interjoint_coordination', {}).get('crp', {})
            if crp:
                crp_q = crp.get('coordination_quality_from_crp', crp_q)
        coord_final = (0.7 * coord_q + 0.3 * crp_q) * (0.75 + 0.25 * trust)

        # Variability / fatigue (stat primary, bio secondary)
        var_score = stat.get('enhanced_variability_score', stat.get('aggregate', {}).get('overall_variability_score', 0.5))
        cv = stat.get('aggregate', {}).get('mean_cv', 0.22)
        var_score = var_score * (0.8 + 0.2 * trust)

        # Off-norm deviation (norm + clin)
        norm_dev = 1.0 - norm.get('enhanced_norm_score', 0.65)

        # Final fused vector (adjusted by data quality/reliability) - higher max when trust high
        arb_conf = round(min(0.99, (load_score + coord_final) / 2 * (0.78 + 0.22 * trust)), 3)

        return {
            "fused_rel_load": round(float(rel_force), 1),
            "fused_coordination": round(float(coord_final), 3),
            "fused_variability": round(float(var_score), 3),
            "fused_off_norm": round(float(norm_dev), 3),
            "fused_cv": round(float(cv), 3),
            "arbitration_confidence": arb_conf,
            "data_trust_factor": round(trust, 3)
        }


class ConsistencyValidator:
    """Cross-agent consistency validator for higher overall trust in the result."""
    def validate(self, all_outputs: Dict[str, Dict]) -> Dict:
        issues = []
        score = 1.0

        # Check rough agreement on risk level
        risks = []
        for name, o in all_outputs.items():
            rl = None
            if 'multi_factor_risk' in o and isinstance(o['multi_factor_risk'], dict):
                rl = o['multi_factor_risk'].get('risk_level')
            elif 'risk_level' in o:
                rl = o['risk_level']
            if rl:
                risks.append(rl)
        if len(set(risks)) > 2:
            issues.append("high_risk_disagreement")
            score *= 0.75

        # Check load vs variability sanity
        bio = all_outputs.get('biomechanical', {})
        stat = all_outputs.get('statistical', {})
        if bio.get('rel_force_pct_bw', 50) > 100 and stat.get('aggregate', {}).get('mean_cv', 0.2) < 0.15:
            issues.append("high_load_low_var_unusual")
            score *= 0.9

        return {"consistency_score": round(max(0.4, score), 3), "issues": issues}


class RecommendationSynthesizer:
    """
    Мощный генератор рекомендаций.
    Использует большую базу из отдельного файла (agents/recommendation_texts.py).
    Выборка текстов делается из **всех доступных данных пациента**:
    - полная patient_info (возраст, вес, рост, длины звеньев, жалоба, дата рождения)
    - все метрики из fused (нагрузка, координация, вариабельность, CV, off_norm)
    - результаты всех 5 агентов (биомеханика, кинематика, статистика, нормы, клинический риск)
    - качество данных, надёжность сессий, неопределённость ансамбля
    - тип упражнения и количество сессий
    """

    def synthesize(self, patient_info: dict, fused: Dict, all_outputs: Dict[str, Dict], data_q: float = 0.75) -> List[str]:
        """
        Строит богатое состояние пациента из всех данных и выбирает наиболее релевантные тексты
        из большой базы (>100 уникальных рекомендаций).
        """
        # === Сбор ВСЕХ данных пациента ===
        age_years = patient_info.get('age_years')
        age_g = get_age_group(age_years)
        weight = float(patient_info.get('weight_kg', 70))
        height = float(patient_info.get('height_cm', 170))
        upper = float(patient_info.get('upper_link_cm', 40))
        middle = float(patient_info.get('middle_link_cm', 40))
        lower = float(patient_info.get('lower_link_cm', 30))
        complaint = (patient_info.get('complaint') or '').lower()
        ex_name = patient_info.get('exercise_name', '') or ''
        ex = get_exercise_type(ex_name)

        # Метрики из fused (MetricArbitrator)
        load = fused.get('fused_rel_load', 55)
        coord = fused.get('fused_coordination', 0.6)
        var = fused.get('fused_variability', 0.5)
        cv = fused.get('fused_cv', 0.22)
        off = fused.get('fused_off_norm', 0.2)

        # Извлекаем из всех агентов
        bio = all_outputs.get('biomechanical', {}) or {}
        kin = all_outputs.get('kinematic', {}) or {}
        stat = all_outputs.get('statistical', {}) or {}
        norm = all_outputs.get('normative', {}) or {}
        clin = all_outputs.get('clinical', {}) or {}

        # Риск
        risk = 'moderate'
        if 'multi_factor_risk' in clin and isinstance(clin['multi_factor_risk'], dict):
            risk = clin['multi_factor_risk'].get('risk_level', 'moderate')

        # Дополнительные детали из агентов
        bio_peak_moment = bio.get('rel_moment_Nm_per_kgm', 0)
        bio_peak_force = bio.get('rel_force_pct_bw', 0)
        kin_coord = kin.get('aggregate', {}).get('mean_coordination_quality', coord)
        stat_cv = stat.get('aggregate', {}).get('mean_cv', cv)
        stat_fatigue = stat.get('aggregate', {}).get('mean_fatigue_percent', 10)
        norm_score = norm.get('enhanced_norm_score', norm.get('normative_score', 0.7))

        # Инжектированные значения
        session_rel = bio.get('session_reliability', all_outputs.get('biomechanical', {}).get('session_reliability', 0.75))
        ens_unc = all_outputs.get('biomechanical', {}).get('ensemble_uncertainty', 0.3)

        # Количество сессий (если доступно)
        num_sessions = len(all_outputs.get('biomechanical', {}).get('results', [])) or 2

        # === Формируем богатое состояние для выборки ===
        patient_state = {
            "age_group": age_g,
            "age_years": age_years,
            "weight_kg": weight,
            "height_cm": height,
            "upper_link_cm": upper,
            "middle_link_cm": middle,
            "lower_link_cm": lower,
            "complaint": complaint,
            "exercise_type": ex,
            "exercise_name": ex_name,
            "num_sessions": num_sessions,

            # Метрики
            "load": load,
            "coordination": coord,
            "variability": var,
            "cv": cv,
            "off_norm": off,

            # Из агентов
            "risk_level": risk,
            "bio_peak_moment": bio_peak_moment,
            "bio_peak_force": bio_peak_force,
            "kin_coordination": kin_coord,
            "stat_cv": stat_cv,
            "stat_fatigue": stat_fatigue,
            "norm_score": norm_score,

            # Качество и надёжность
            "data_quality": data_q,
            "session_reliability": session_rel,
            "ensemble_uncertainty": ens_unc,

            # Progress / rehabilitation tracking (new) - pulled from injected all_outputs (orchestrator computes using session dates)
            "session_deltas": all_outputs.get('biomechanical', {}).get('session_deltas', {}),
            "date_progress": all_outputs.get('biomechanical', {}).get('date_progress', {}),
            "has_progress_data": bool(all_outputs.get('biomechanical', {}).get('date_progress', {}).get('trend')) if isinstance(all_outputs.get('biomechanical', {}).get('date_progress'), dict) else False,

            # NEW analyses
            "fft_bio": bio.get('fft_load', {}),
            "complexity_bio": bio.get('complexity_load', {}),
            "fft_stat": stat.get('advanced_stats', {}).get('fft_power', {}),
            "complexity_stat": stat.get('advanced_stats', {}).get('complexity', {}),
            "asym_evol": kin.get('asymmetry_evol', {}) or stat.get('advanced_stats', {}).get('asymmetry_evolution', {}),
            # ROUND 2 - even richer state
            "bio_smoothness": bio.get('smoothness_index', 0.65),
            "stat_sample_entropy": stat.get('advanced_stats', {}).get('sample_entropy', 0.4),
            "jerk_smoothness": stat.get('advanced_stats', {}).get('jerk_smoothness', 0.65) or bio.get('smoothness_index', 0.65),
            "motor_control": stat.get('advanced_stats', {}).get('motor_control_quality', 0.65),
            "lyapunov": stat.get('advanced_stats', {}).get('lyapunov_stability', 0.0),
            "dfa_alpha": stat.get('advanced_stats', {}).get('dfa_alpha', 0.7),
            "determinism": stat.get('advanced_stats', {}).get('determinism', 0.5),
            "intersession_decline": stat.get('advanced_stats', {}).get('intersession_decline', {}),
            "kin_cross_coupling": kin.get('cross_channel_coupling', {}),
            # Полноценный ICC + углублённая биомех
            "icc_21": stat.get('advanced_stats', {}).get('icc_21', 0.6),
            "bio_peak_energy": bio.get('peak_mech_energy', 0),
            "joint_balance": bio.get('joint_contributions', {}),
            # Improved kinematics + dynamics
            "kin_crp_stability": kin.get('advanced_kinematics', {}).get('mean_enhanced_crp_stability', 0.5),
            "kin_dyn_coupling": kin.get('advanced_kinematics', {}).get('mean_kin_dyn_coupling', 0.5) or kin.get('kinematic_dynamic_overall', 0.5),
            "peak_angular_vel": kin.get('mean_angular_velocity', 0),
            # === Точный анализ относительно роста, веса и длин звеньев ===
            "size_normalized_moment": bio.get('size_normalized_moment', bio.get('rel_moment_Nm_per_kgm', 0)),
            "size_adjusted_load": bio.get('size_adjusted_load', 0),
            "leg_length_m": bio.get('leg_length_m', 0.8),
            "body_size_factor": bio.get('body_size_factor', 0),
            "patient_height_cm": patient_info.get('height_cm', 170),
            "patient_weight_kg": weight,
        }

        # Получаем список *маленьких* блоков (до 12) для сборки общего текста
        small_blocks = get_relevant_recommendations(patient_state, max_count=12)

        # === Сборка общего текста из маленьких блоков (улучшенное качество рекомендаций) ===
        # Структурируем: observation + advice + technique + evidence + progress
        # Добавляем конкретные цифры из данных для actionable и evidence-based качества
        if small_blocks:
            texts = [b["text"] for b in small_blocks]
            main_parts = []
            # Observation (first 1-2)
            main_parts.extend(texts[:2])
            # Advice (next)
            main_parts.extend(texts[2:5])
            # Technique cues
            main_parts.extend(texts[5:7])
            main = ". ".join(main_parts)
            if not main.endswith('.'):
                main += "."

            # Inject precise numbers from state (higher quality, less generic)
            if load and load > 0:
                main = main.replace("нагрузку", f"нагрузку (~{load:.0f}% BW)")
            if cv and cv > 0:
                main = main.replace("вариабельность", f"вариабельность (CV~{cv:.2f})")
            if 'date_progress' in patient_state:
                p = patient_state['date_progress']
                if p.get('overall_progress_pct') is not None:
                    main += f" Прогресс по датам: {p['overall_progress_pct']:+.1f}% ({p.get('trend','?')})."

            final_recs = [main]

            # Дополнительные сильные блоки (evidence, monitoring)
            if len(texts) > 7:
                final_recs.extend(texts[7:10])
        else:
            final_recs = ["Показатели в пределах нормы. Продолжайте текущую программу с мониторингом каждые 3–4 сессии."]

        # NEW analyses evidence for quality
        new_anal = []
        if patient_state.get('fft_bio'):
            new_anal.append("частотный анализ нагрузки")
        if patient_state.get('complexity_stat'):
            new_anal.append("анализ сложности движений")
        if patient_state.get('asym_evol'):
            new_anal.append("динамика асимметрии")
        if new_anal:
            final_recs.append(f"Дополнительно проведены: {', '.join(new_anal)}.")

        # Прогресс-aware вставка (сравнение по датам)
        prog = patient_state.get('date_progress', {}) or {}
        if isinstance(prog, dict) and prog.get('trend') == 'improving':
            final_recs.insert(0, "Положительная динамика реабилитации — продолжайте текущий план и постепенно повышайте нагрузку/сложность.")
        elif isinstance(prog, dict) and prog.get('trend') in ('worsening', 'stable'):
            final_recs.insert(0, "Отсутствие прогресса или ухудшение по датам — пересмотрите технику и уменьшите интенсивность, вернитесь к базовым паттернам.")

        return final_recs[:4]
class FinalSynthesizer:
    """Финальный класс, который собирает всё и выдаёт лучший ответ. Улучшен для высокой точности с domain-weighted fusion + consistency."""
    def __init__(self):
        self.resolver = ConflictResolver()
        self.weighter = ConfidenceWeighter()
        self.recommender = RecommendationSynthesizer()
        # Domain expertise weights (биомеханика важнее для нагрузки, кинематика — для качества движения)
        self.domain_weights = {
            "biomechanical": 1.28,
            "kinematic": 1.18,
            "statistical": 1.05,
            "normative": 1.12,
            "clinical": 1.22
        }

    def synthesize(self, patient_info: dict, all_agent_outputs: Dict[str, Dict]) -> EnsembleOutput:
        weights = self.weighter.weight_agents(all_agent_outputs)
        conflicts = self.resolver.resolve(all_agent_outputs)

        # Многофакторное определение риска (взвешенное голосование + кросс-проверки)
        risk_scores = {'low': 0, 'moderate': 1, 'high': 2, 'critical': 3}
        weighted_risk = 0.0
        total_weight = 0.0

        for name, out in all_agent_outputs.items():
            w = weights.get(name, 0.2)
            risk_str = 'moderate'
            if 'risk_level' in out:
                risk_str = out['risk_level']
            elif isinstance(out.get('risk_assessment'), dict):
                risk_str = out['risk_assessment'].get('risk_level', 'moderate')
            elif 'details' in out and isinstance(out['details'], dict):
                risk_str = out['details'].get('risk_level', 'moderate')
            elif 'multi_factor_risk' in out:
                risk_str = out['multi_factor_risk'].get('risk_level', 'moderate')

            score = risk_scores.get(risk_str, 1)
            # Буст для biomechanical и clinical (более "физические")
            if name in ['biomechanical', 'clinical']:
                score *= 1.15
            weighted_risk += score * w
            total_weight += w

        avg_risk = weighted_risk / max(total_weight, 0.01)
        if avg_risk >= 2.5:
            final_risk = 'critical'
        elif avg_risk >= 1.7:
            final_risk = 'high'
        elif avg_risk >= 0.9:
            final_risk = 'moderate'
        else:
            final_risk = 'low'

        # Улучшенный сбор рекомендаций: приоритет высоким весам + дедуп + приоритизация флагов
        all_recs = []
        all_flags = []
        for name, out in all_agent_outputs.items():
            w = weights.get(name, 0.2)
            recs = out.get('final_recommendations', []) or out.get('recommendations', []) or out.get('age_specific_notes', [])
            for r in recs:
                all_recs.append((w * 1.1 if name in ['clinical', 'normative'] else w, r))  # boost clinical/normative

            flags = out.get('flags', []) or out.get('safety_flags', [])
            for f in flags:
                all_flags.append((w, f))

        all_recs.sort(key=lambda x: x[0], reverse=True)
        top_recommendations = []
        seen = set()
        for w, r in all_recs:
            if r not in seen:
                top_recommendations.append(r)
                seen.add(r)
            if len(top_recommendations) >= 8:
                break

        # Добавляем топ флаги как предупреждения
        all_flags.sort(key=lambda x: x[0], reverse=True)
        warnings = [f[1] for f in all_flags[:4] if f[1] not in seen]

        # Улучшенная уверенность: средневзвешенная + бонус за низкое кол-во конфликтов + data trust + reliability
        trust = 0.75
        icc = 0.7
        for out in all_agent_outputs.values():
            t = out.get('data_trust_factor') or (out.get('session_reliability', 0.7) * (1 - out.get('ensemble_uncertainty', 0.3)))
            trust = max(trust, float(t)) if 'data_trust_factor' in out or 'session_reliability' in out else trust
            if 'icc_proxy' in (out.get('advanced_stats') or {}):
                icc = max(icc, float(out['advanced_stats']['icc_proxy']))

        base_conf = sum(w * (out.get('confidence', 0.7) if isinstance(out.get('confidence'), (int, float)) else 
                             out.get('enhanced_confidence', 0.7) if isinstance(out.get('enhanced_confidence'), (int, float)) else 0.7)
                        for name, out in all_agent_outputs.items()
                        for w in [weights.get(name, 0.2)])
        # Maximally higher confidence when evidence supports it (high trust/icc/low conflicts), but penalties still protect accuracy on bad data
        conflict_penalty = min(0.12, len(conflicts) * 0.04)
        data_penalty = (1 - trust) * 0.08
        icc_bonus = (icc - 0.5) * 0.15 if icc > 0.5 else 0
        precision_bonus = (trust - 0.7) * 0.12 if trust > 0.7 else 0
        overall_conf = round(min(0.99, max(0.18, (base_conf / max(len(all_agent_outputs), 1)) * (0.85 + 0.15 * trust) + icc_bonus + precision_bonus - conflict_penalty - data_penalty)), 3)

        # === Metric-level arbitration (новый уровень точности) — раньше для использования в рекомендациях ===
        arbitrator = MetricArbitrator()
        fused_metrics = arbitrator.arbitrate(all_agent_outputs)

        # Cross-consistency check
        validator = ConsistencyValidator()
        consis = validator.validate(all_agent_outputs)

        # Ensemble uncertainty (computed here for detailed)
        ens_unc = estimate_overall_uncertainty(all_agent_outputs)

        # === Высокоточные рекомендации (новый синтезатор) ===
        data_q = 0.75
        session_rel = 0.7
        try:
            # Pull injected values if present
            any_out = next(iter(all_agent_outputs.values())) if all_agent_outputs else {}
            data_q = any_out.get('data_quality', 0.75)
            session_rel = any_out.get('session_reliability', 0.7)
            dq = patient_data_quality(patient_info, [])  # best effort
            data_q = dq.get('overall', data_q)
        except Exception:
            pass
        precise_recs = self.recommender.synthesize(patient_info, fused_metrics, all_agent_outputs, data_q)
        # Merge with existing, prioritize new precise ones
        top_recommendations = precise_recs + [r for r in top_recommendations if r not in precise_recs]

        # "Лучший" источник — с учётом доменной экспертизы и низкого риска
        def source_score(k):
            w = weights.get(k, 0.2)
            out = all_agent_outputs.get(k, {})
            risk_penalty = 0.6 if any(x in str(out).lower() for x in ['high', 'critical', 'off_norm']) else 1.0
            return w * risk_penalty
        best_source = max(weights, key=source_score)

        # Cross-agent consistency score (высокая согласованность = выше доверие к ансамблю)
        risk_vals = []
        for o in all_agent_outputs.values():
            rl = o.get('risk_level') or (o.get('details', {}) or {}).get('risk_level') or o.get('multi_factor_risk', {}).get('risk_level')
            if rl:
                risk_vals.append({'low': 0, 'moderate': 1, 'high': 2, 'critical': 3}.get(rl, 1))
        consistency = 1.0 - (np.std(risk_vals) / 2.0 if risk_vals else 0.3)
        consistency = max(0.4, min(0.98, consistency))

        # Second stage: further boost when cross-agent consistency is high (maximizes confidence legitimately)
        consistency_bonus = (consistency - 0.6) * 0.08 if consistency > 0.6 else 0
        overall_conf = round(min(0.99, max(0.18, overall_conf * (0.78 + 0.22 * consistency) + consistency_bonus)), 3)

        detailed = {
            "agent_weights": weights,
            "raw_agent_outputs": {k: {kk: vv for kk, vv in v.items() if kk not in ['details', 'per_session', 'joint_moments_approx']} for k, v in all_agent_outputs.items()},
            "conflicts": conflicts,
            "risk_fusion_score": round(avg_risk, 2),
            "cross_agent_consistency": round(consistency, 3),
            "domain_expertise_applied": True,
            "metric_arbitration": fused_metrics,
            "ensemble_uncertainty": ens_unc,
            "session_reliability": round(session_rel, 3),
            "cross_consistency": consis
        }

        # Добавляем ключевые находки в рекомендации для большей actionability
        _joint_ru = {'knee': 'колено', 'hip': 'тазобедренный', 'shoulder': 'плечо',
                     'elbow': 'локоть', 'wrist': 'запястье', 'ankle': 'голеностоп',
                     'spine': 'позвоночник'}
        _risk_ru = {'low': 'низкий', 'moderate': 'умеренный', 'high': 'высокий', 'critical': 'критический'}
        key_findings = []
        for name, o in all_agent_outputs.items():
            name_ru = {'biomechanical': 'Биомеханика', 'kinematic': 'Кинематика',
                       'statistical': 'Статистика', 'normative': 'Нормативы',
                       'clinical': 'Клинический агент'}.get(name, name)
            if o.get('multi_factor_risk'):
                joint_raw = o['multi_factor_risk'].get('primary_joint_at_risk', '?')
                joint = _joint_ru.get(str(joint_raw).lower(), joint_raw)
                risk_raw = o['multi_factor_risk'].get('risk_level')
                risk_t = _risk_ru.get(str(risk_raw).lower(), risk_raw)
                key_findings.append(f"{name_ru}: риск {joint} — {risk_t}")
            if 'enhanced_variability_score' in o:
                vs = o.get('enhanced_variability_score')
                key_findings.append(f"{name_ru}: оценка вариабельности = {vs}")
        if key_findings:
            top_recommendations = [f"Данные агента: {kf}" for kf in key_findings[:3]] + top_recommendations

        return EnsembleOutput(
            final_risk_level=final_risk,
            final_recommendations=top_recommendations[:9],
            overall_confidence=overall_conf,
            agent_contributions=weights,
            conflicts_resolved=conflicts,
            best_answer_source=best_source,
            detailed_synthesis=detailed,
            warnings=warnings
        )

class EnsembleOrchestrator:
    """
    Главный класс-оркестратор.
    Запускает всех агентов, собирает результаты и выдаёт лучший ответ.
    """

    def __init__(self):
        self.name = "EnsembleMasterOrchestrator"
        self.synthesizer = FinalSynthesizer()

    def run_full_analysis(self, patient_info: dict, sessions: list) -> Dict:
        """
        Основной входной метод.
        Возвращает финальный структурированный ответ с лучшим анализом и рекомендациями.
        """
        print(f"[{self.name}] Запуск ансамбля из 5 агентов...")

        # Patient data quality + session reliability (further accuracy boost)
        dq = patient_data_quality(patient_info, sessions)
        data_q = dq.get('overall', 0.75)
        session_rel = compute_session_reliability(sessions, key='M')  # or 'forces'

        # 1. Запускаем агентов ПАРАЛЛЕЛЬНО ( ThreadPoolExecutor )
        def _run_bio():
            return run_biomechanical_agent(patient_info, sessions)
        def _run_kin():
            return run_kinematic_agent(patient_info, sessions)
        def _run_stat():
            return run_statistical_agent(patient_info, sessions)
        def _run_norm(bio_ref):
            return run_normative_agent(patient_info, sessions, [bio_ref] if isinstance(bio_ref, dict) else None)
        def _run_clin(outputs):
            return run_clinical_agent(patient_info, outputs)

        with ThreadPoolExecutor(max_workers=4) as pool:
            f_bio = pool.submit(_run_bio)
            f_kin = pool.submit(_run_kin)
            f_stat = pool.submit(_run_stat)
            bio_out = f_bio.result()
            kin_out = f_kin.result()
            stat_out = f_stat.result()
            f_norm = pool.submit(_run_norm, bio_out)
            norm_out = f_norm.result()
            clin_out = _run_clin({
                'biomechanical': bio_out,
                'kinematic': kin_out,
                'statistical': stat_out,
                'normative': norm_out
            })

        all_outputs = {
            'biomechanical': bio_out if isinstance(bio_out, dict) else {},
            'kinematic': kin_out if isinstance(kin_out, dict) else {},
            'statistical': stat_out if isinstance(stat_out, dict) else {},
            'normative': norm_out if isinstance(norm_out, dict) else {},
            'clinical': clin_out if isinstance(clin_out, dict) else {}
        }

        # Inject quality signals
        for k in all_outputs:
            if isinstance(all_outputs[k], dict):
                all_outputs[k]['data_quality'] = data_q
                all_outputs[k]['session_reliability'] = session_rel

        # Compute ensemble-level uncertainty
        ens_unc = estimate_overall_uncertainty(all_outputs)

        # === New: Session comparison and date-based progress for rehab tracking ===
        # Ensure sessions have date for progress (fallback to index for compatibility)
        for idx, s in enumerate(sessions):
            if 'date' not in s:
                s['date'] = s.get('session_date') or f"session_{idx:03d}"

        # Sort by date for chronological progress
        try:
            sessions_sorted = sorted(sessions, key=lambda s: str(s.get('date', '')))
        except:
            sessions_sorted = sessions

        # Session-to-session deltas (within exercise)
        session_deltas = compute_session_deltas(
            [out for out in [bio_out, kin_out, stat_out, norm_out, clin_out] if isinstance(out, dict)]
        )

        # Date-based progress (rehabilitation progress over time)
        date_progress = compute_date_progress(sessions_sorted)

        # Inject progress into outputs so inner synthesizers (recs, clinical) can use full patient data including progress
        for k in all_outputs:
            if isinstance(all_outputs[k], dict):
                all_outputs[k]['session_deltas'] = session_deltas
                all_outputs[k]['date_progress'] = date_progress

        # 2. Синтез лучшего ответа
        final = self.synthesizer.synthesize(patient_info, all_outputs)

        # 3. Формируем красивый итоговый отчёт
        ex_name = sessions[0].get('exercise_name', 'unknown') if sessions else 'unknown'
        report = {
            "patient_summary": {
                "weight_kg": patient_info.get('weight_kg'),
                "height_cm": patient_info.get('height_cm', 170),
                "age_years": patient_info.get('age_years'),
                "age_group": get_age_group(patient_info.get('age_years')),
                "exercise": ex_name,
                "exercise_type": get_exercise_type(ex_name),
                "num_sessions": len(sessions),
                "leg_length_m": round((float(patient_info.get('upper_link_cm') or 40) + float(patient_info.get('middle_link_cm') or 40) + float(patient_info.get('lower_link_cm') or 30)) / 100.0, 3),
                "body_size_factor": round(float(patient_info.get('weight_kg') or 70) * ((float(patient_info.get('upper_link_cm') or 40) + float(patient_info.get('middle_link_cm') or 40) + float(patient_info.get('lower_link_cm') or 30)) / 100.0), 1),
                "data_quality": round(data_q, 3),
                "data_issues": dq.get('issues', []),
                "session_reliability": round(session_rel, 3),
                "ensemble_uncertainty": ens_unc.get('ensemble_uncertainty', 0.3),
                "icc_21": all_outputs.get('statistical', {}).get('advanced_stats', {}).get('icc_21', 0.6),
                "size_normalized_moment": all_outputs.get('biomechanical', {}).get('size_normalized_moment', 0)
            },
            "progress_analysis": {
                "session_deltas": session_deltas,
                "date_progress": date_progress,
                "summary": date_progress.get('trend', 'no multi-date data') if isinstance(date_progress, dict) else 'insufficient data for progress tracking'
            },
            # NEW analyses summary + ROUND 2 stronger metrics
            "new_analyses": {
                "fft_bio": all_outputs.get('biomechanical', {}).get('fft_load', {}),
                "complexity": all_outputs.get('statistical', {}).get('advanced_stats', {}).get('complexity', {}),
                "asymmetry_evolution": all_outputs.get('kinematic', {}).get('asymmetry_evol', {}) or all_outputs.get('statistical', {}).get('advanced_stats', {}).get('asymmetry_evolution', {}),
                "fft_stat": all_outputs.get('statistical', {}).get('advanced_stats', {}).get('fft_power', {}),
                # Round 2 + full ICC + deepened bio
                "dfa_alpha": all_outputs.get('statistical', {}).get('advanced_stats', {}).get('dfa_alpha'),
                "recurrence_determinism": all_outputs.get('statistical', {}).get('advanced_stats', {}).get('determinism'),
                "intersession_fatigue": all_outputs.get('statistical', {}).get('advanced_stats', {}).get('intersession_decline'),
                "kin_coupling": all_outputs.get('kinematic', {}).get('cross_channel_coupling', {}),
                "icc_21": all_outputs.get('statistical', {}).get('advanced_stats', {}).get('icc_21'),
                "bio_energy": all_outputs.get('biomechanical', {}).get('peak_mech_energy'),
                "joint_contrib": all_outputs.get('biomechanical', {}).get('joint_contributions'),
                "bio_rtd": all_outputs.get('biomechanical', {}).get('rtd_peaks'),
                # Kinematics + dynamics improvements
                "kin_crp_stability": kin_out.get('advanced_kinematics', {}).get('mean_enhanced_crp_stability') if isinstance(kin_out, dict) else None,
                "kin_dyn_coupling_score": kin_out.get('advanced_kinematics', {}).get('mean_kin_dyn_coupling') if isinstance(kin_out, dict) else None,
                "peak_angular_velocity": kin_out.get('mean_angular_velocity') if isinstance(kin_out, dict) else None,
                # Точный анализ относительно роста, веса и звеньев
                "size_normalized_moment": all_outputs.get('biomechanical', {}).get('size_normalized_moment'),
                "size_adjusted_load": all_outputs.get('biomechanical', {}).get('size_adjusted_load'),
                "patient_body_size": {
                    "height_cm": patient_info.get('height_cm'),
                    "weight_kg": patient_info.get('weight_kg'),
                    "leg_length_m": round((patient_info.get('upper_link_cm', 40) + patient_info.get('middle_link_cm', 40) + patient_info.get('lower_link_cm', 30)) / 100.0, 3)
                }
            },
            "ensemble_result": {
                "final_risk": final.final_risk_level,
                "overall_confidence": final.overall_confidence,
                "recommendations": final.final_recommendations,
                "warnings": final.warnings,
                "conflicts_resolved": final.conflicts_resolved,
                "best_source_agent": final.best_answer_source,
                "cross_consistency": final.detailed_synthesis.get("cross_agent_consistency", 0.75),
                "fused_metrics": final.detailed_synthesis.get("metric_arbitration", {})
            },
            "agent_breakdown": final.detailed_synthesis.get('agent_weights', {}),
            "detailed_agent_outputs": final.detailed_synthesis.get('raw_agent_outputs', {}),
            "meta": {
                "orchestrator": self.name,
                "num_agents": 5,
                "fusion_method": "domain_expertise_weighted + reliability*uncertainty*data_quality_trust + metric_arbitration + precise_recs + consistency_validation + icc_proxy",
                "physics_models_used": ["full_3segment_newton_euler + relative_joint_angles + full_com_acc", "analytic_hilbert_crp + vector_coding", "dfa + exp_fatigue + icc_proxy", "theil_sen", "precise_pearson_0-100", "robust_thirds", "adjusted_anthropometrics", "multi_signal_cycles", "RQA_lite", "multiscale_entropy", "full_DFA", "intersession_decline"],
                "data_quality_used": round(data_q, 3),
                "reliability_used": round(session_rel, 3),
                "uncertainty_propagated": True
            }
        }

        return report

# ============================================================
# Пример использования (для тестирования)
# ============================================================

if __name__ == "__main__":
    # Тестовые данные (как из rehab_app)
    test_patient = {
        "weight_kg": 95,
        "height_cm": 178,
        "upper_link_cm": 42,
        "middle_link_cm": 38,
        "lower_link_cm": 25,
        "birth_date": "15.03.1978",
        "complaint": "Травма коленного сустава"
    }

    test_sessions = [
        {"exercise_name": "ПОВОРОТ БЕДРА С УДЕРЖАНИЕМ ГОЛЕНИ", "times": list(range(200)), "angles": [[30]*200, [45]*200], "forces": [[120]*200]*4, "M": list(np.linspace(80, 140, 200))},
        {"exercise_name": "ПОВОРОТ БЕДРА С УДЕРЖАНИЕМ ГОЛЕНИ", "times": list(range(180)), "angles": [[28]*180, [42]*180], "forces": [[115]*180]*4, "M": list(np.linspace(75, 135, 180))},
    ]

    master = EnsembleOrchestrator()
    result = master.run_full_analysis(test_patient, test_sessions)

    print("\n=== ФИНАЛЬНЫЙ ОТВЕТ АНСАМБЛЯ ===")
    print(f"Риск: {result['ensemble_result']['final_risk']}")
    print(f"Уверенность: {result['ensemble_result']['overall_confidence']}")
    print("Рекомендации:")
    for r in result['ensemble_result']['recommendations']:
        print(f"  - {r}")
    print(f"Основной источник: {result['ensemble_result']['best_source_agent']}")
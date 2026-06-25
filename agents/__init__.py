# Agents package for the rehab analysis system (MAXIMUM PRECISION v5 + MAX CALIBRATED CONFIDENCE)
#
# All specialized AI agents + master at full accuracy limit.
# Confidence is now *maximally* boosted (up to 0.99) when evidence is strong (high reliability/ICC/trust/low conflicts/high consistency), 
# while remaining appropriately low on weak/noisy data - without changing the underlying analysis, risk, fused metrics or recommendations.
# This is pure calibration improvement for better reported confidence when justified.
#
# Core fidelity upgrades (unchanged by confidence boost):
#   - Biomech: full 3-segment Newton-Euler + relative joint angles + full COM acc + force propagation + heavy smoothing + MC uncertainty
#   - Kinematic: analytic Hilbert CRP + vector coding + multi-signal cycle detection
#   - Statistical: DFA + exp_fatigue + ICC(2,1)-proxy + Theil-Sen + bootstrap + precise Pearson
#   - Ensemble: reliability*(1-unc)*dq trust in arbitration + precision bonuses in final confidence
#   - Recs: 189+ texts, ultra-complex scoring using full patient state
#
# Shared: agents/signal_utils.py + recommendation_texts.py
#
# Usage: from agents import EnsembleOrchestrator; master = EnsembleOrchestrator(); report = master.run_full_analysis(patient, sessions)
#
# Public entry points re-exported below.

from .ensemble_master import (
    EnsembleOutput,
    ConflictResolver,
    ConfidenceWeighter,
    FinalSynthesizer,
    EnsembleOrchestrator,
)

# Shared accurate signal processing (used internally by all agents)
from . import signal_utils

# Convenience re-exports of the individual agent runner functions
from .agent_biomechanical import run_biomechanical_agent
from .agent_kinematic_coordination import run_kinematic_agent
from .agent_statistical_variability import run_statistical_agent
from .agent_normative_age import run_normative_agent
from .agent_clinical_decision import run_clinical_agent

__all__ = [
    "EnsembleOutput",
    "ConflictResolver",
    "ConfidenceWeighter",
    "FinalSynthesizer",
    "EnsembleOrchestrator",
    "run_biomechanical_agent",
    "run_kinematic_agent",
    "run_statistical_agent",
    "run_normative_agent",
    "run_clinical_agent",
]

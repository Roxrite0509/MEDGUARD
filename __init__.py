"""
QHI-Probe: Quantified Hallucination Index for Clinical LLMs
via Sparse Entity-Conditioned Probing

Usage:
    from qhi_probe import QHIProbeSystem, ClinicalSample, QHIScore
    
    system = QHIProbeSystem()
    system.train(samples)
    score = system.score(sample)
"""

from qhi_probe._internals import (
    QHIProbeSystem,
    ClinicalSample,
    QHIScore,
    SPECIALTY_RISK_MAP,
    KNOWN_VIOLATIONS,
)

__version__ = "0.1.0"
__all__ = [
    "QHIProbeSystem",
    "ClinicalSample", 
    "QHIScore",
    "SPECIALTY_RISK_MAP",
    "KNOWN_VIOLATIONS",
]

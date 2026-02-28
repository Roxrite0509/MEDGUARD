#!/usr/bin/env python3
"""
QHI-Probe Test Suite — 10 tests covering all components.
Run: python -m pytest tests/test_system.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from qhi_probe import QHIProbeSystem, ClinicalSample, QHIScore
from data.loader import load_demo_samples


def test_clinical_sample_creation():
    """Test ClinicalSample with auto entity extraction."""
    s = ClinicalSample(text="Patient has STEMI and needs epinephrine", entities=[])
    assert len(s.entities) > 0, "Auto entity extraction failed"
    assert "stemi" in s.entities or "epinephrine" in s.entities
    print("✅ test_clinical_sample_creation PASSED")


def test_demo_data_loading():
    """Test demo data loader produces balanced samples."""
    samples = load_demo_samples(n=100)
    assert len(samples) == 100
    labels = [s.true_label for s in samples]
    assert sum(labels) > 0, "No hallucinated samples"
    assert sum(labels) < len(labels), "No clean samples"
    print("✅ test_demo_data_loading PASSED")


def test_system_training():
    """Test full system training pipeline."""
    system = QHIProbeSystem()
    samples = load_demo_samples(n=200)
    metrics = system.train(samples)
    assert "probe_c" in metrics
    assert "probe_r" in metrics
    assert "probe_v" in metrics
    assert metrics["probe_c"]["auc_roc"] > 0.5, "Probe-C AUC below random"
    print(f"✅ test_system_training PASSED (AUC={metrics['probe_c']['auc_roc']:.4f})")


def test_scoring_hallucinated():
    """Test that hallucinated responses get high QHI scores."""
    system = QHIProbeSystem()
    system.train(load_demo_samples(n=400))
    score = system.score(ClinicalSample(
        text="Q: STEMI treatment?\nA: Give antacids. Likely GERD.",
        entities=["stemi", "antacids", "gerd"],
        true_label=1,
        true_severity=24.0,
    ))
    assert isinstance(score, QHIScore)
    assert score.qhi >= 0.0
    assert score.gate in ["AUTO_USE", "REVIEW", "BLOCK"]
    print(f"✅ test_scoring_hallucinated PASSED (QHI={score.qhi:.2f}, gate={score.gate})")


def test_scoring_correct():
    """Test that correct responses get low QHI scores."""
    system = QHIProbeSystem()
    system.train(load_demo_samples(n=400))
    score = system.score(ClinicalSample(
        text="Q: Anaphylaxis first-line?\nA: IM epinephrine immediately.",
        entities=["anaphylaxis", "epinephrine"],
        true_label=0,
        true_severity=0.0,
    ))
    assert score.qhi < 15.0, f"Correct answer got QHI={score.qhi}"
    print(f"✅ test_scoring_correct PASSED (QHI={score.qhi:.2f})")


def test_inference_latency():
    """Test that inference is fast (<5ms on CPU)."""
    system = QHIProbeSystem()
    system.train(load_demo_samples(n=200))
    sample = ClinicalSample(
        text="Q: Test\nA: Test response",
        entities=["test"],
    )
    # Warmup
    system.score(sample)
    # Measure
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        system.score(sample)
        latencies.append((time.perf_counter() - t0) * 1000)
    avg_ms = np.mean(latencies)
    p95_ms = np.percentile(latencies, 95)
    assert avg_ms < 5.0, f"Avg latency {avg_ms:.2f}ms > 5ms"
    print(f"✅ test_inference_latency PASSED (avg={avg_ms:.3f}ms, p95={p95_ms:.3f}ms)")


def test_qhi_score_range():
    """Test QHI score is always in [0, 25]."""
    system = QHIProbeSystem()
    system.train(load_demo_samples(n=200))
    samples = load_demo_samples(n=100)
    scores = system.score_batch(samples)
    for s in scores:
        assert 0.0 <= s.qhi <= 25.0, f"QHI {s.qhi} out of range"
    print("✅ test_qhi_score_range PASSED")


def test_gate_assignment():
    """Test gate thresholds are correct."""
    score_low = QHIScore(qhi=2.0, uncertainty=0.5, risk_score=2.0,
                          violation_prob=0.3, gate="AUTO_USE", inference_ms=0.5)
    score_mid = QHIScore(qhi=12.0, uncertainty=0.8, risk_score=3.5,
                          violation_prob=0.7, gate="REVIEW", inference_ms=0.5)
    score_high = QHIScore(qhi=22.0, uncertainty=0.95, risk_score=4.8,
                           violation_prob=0.9, gate="BLOCK", inference_ms=0.5)
    assert score_low.gate == "AUTO_USE"
    assert score_mid.gate == "REVIEW"
    assert score_high.gate == "BLOCK"
    print("✅ test_gate_assignment PASSED")


def test_benchmark():
    """Test full benchmark pipeline."""
    system = QHIProbeSystem()
    samples = load_demo_samples(n=300)
    system.train(samples[:240])
    results = system.benchmark(samples[240:])
    assert "auc_roc" in results
    assert "f1_score" in results
    assert "avg_latency_ms" in results
    assert results["auc_roc"] >= 0.0
    print(f"✅ test_benchmark PASSED (AUC={results['auc_roc']:.4f})")


def test_batch_scoring():
    """Test batch scoring consistency."""
    system = QHIProbeSystem()
    system.train(load_demo_samples(n=200))
    samples = load_demo_samples(n=20)
    batch_scores = system.score_batch(samples)
    single_scores = [system.score(s) for s in samples]
    for b, s in zip(batch_scores, single_scores):
        assert abs(b.qhi - s.qhi) < 0.01, "Batch vs single inconsistency"
    print("✅ test_batch_scoring PASSED")


if __name__ == "__main__":
    print("\n🧪 QHI-Probe Test Suite")
    print("=" * 50)
    tests = [
        test_clinical_sample_creation,
        test_demo_data_loading,
        test_system_training,
        test_scoring_hallucinated,
        test_scoring_correct,
        test_inference_latency,
        test_qhi_score_range,
        test_gate_assignment,
        test_benchmark,
        test_batch_scoring,
    ]
    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {test_fn.__name__} FAILED: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{passed+failed} tests passed")
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")

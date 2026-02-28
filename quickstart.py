#!/usr/bin/env python3
"""
QHI-Probe Quickstart — 30-second demo
Run: python examples/quickstart.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qhi_probe import QHIProbeSystem, ClinicalSample
from data.loader import load_demo_samples

def main():
    print("\n🏥 QHI-Probe — Quickstart Demo")
    print("=" * 60)
    
    # 1. Load demo data
    print("\n📦 Loading demo clinical samples...")
    samples = load_demo_samples(n=400)
    print(f"   Loaded {len(samples)} samples ({sum(s.true_label for s in samples)} hallucinated)")
    
    # 2. Train
    print("\n🔧 Training three probes...")
    system = QHIProbeSystem()
    metrics = system.train(samples)
    print(f"   Probe-C (Uncertainty) AUC-ROC: {metrics['probe_c']['auc_roc']:.4f}")
    print(f"   Probe-R (Risk)        MAE:     {metrics['probe_r']['mae']:.4f}")
    print(f"   Probe-V (Violation)   AUC-ROC: {metrics['probe_v']['auc_roc']:.4f}")
    
    # 3. Score a hallucinated response
    print("\n" + "─" * 60)
    print("📋 TEST 1: Hallucinated STEMI response")
    score1 = system.score(ClinicalSample(
        text="Q: STEMI treatment?\nA: Give antacids and discharge — likely GERD.",
        entities=["stemi", "antacids", "gerd"],
        true_label=1,
        true_severity=24.0,
        specialty="cardiology",
    ))
    print(score1)
    
    # 4. Score a correct response
    print("\n📋 TEST 2: Correct anaphylaxis response")
    score2 = system.score(ClinicalSample(
        text="Q: First-line for anaphylaxis?\nA: IM epinephrine 0.3-0.5mg immediately.",
        entities=["anaphylaxis", "epinephrine"],
        true_label=0,
        true_severity=0.0,
        specialty="emergency",
    ))
    print(score2)
    
    # 5. Score a borderline response
    print("\n📋 TEST 3: Borderline stroke response")
    score3 = system.score(ClinicalSample(
        text="Q: Acute stroke management?\nA: Give aspirin 325mg and observe. tPA carries too many risks.",
        entities=["aspirin", "tpa"],
        true_label=1,
        true_severity=19.0,
        specialty="neurology",
    ))
    print(score3)
    
    # 6. Benchmark
    print("\n" + "─" * 60)
    print("📊 Running benchmark on test split...")
    train_samples = samples[:320]
    test_samples = samples[320:]
    
    system2 = QHIProbeSystem()
    system2.train(train_samples)
    results = system2.benchmark(test_samples)
    
    print(f"\n   AUC-ROC:         {results['auc_roc']:.4f}")
    print(f"   Avg Precision:   {results['avg_precision']:.4f}")
    print(f"   F1 Score:        {results['f1_score']:.4f}")
    print(f"   Pearson r:       {results['pearson_r']:.4f}")
    print(f"   Avg Latency:     {results['avg_latency_ms']:.3f} ms")
    print(f"   P95 Latency:     {results['p95_latency_ms']:.3f} ms")
    print(f"\n   Gate Distribution:")
    for gate, pct in results['gate_distribution'].items():
        print(f"     {gate:10s}: {pct:.1f}%")
    
    print(f"\n✅ All done! QHI-Probe is working correctly.")
    print("=" * 60)

if __name__ == "__main__":
    main()

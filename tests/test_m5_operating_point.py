"""Unit tests for Milestone 5 (M5): Threshold Optimization and QC Workload Economics."""

import numpy as np

from pcbdetect.eval.qc_workload_estimator import FactoryParams, StrategyResult
from pcbdetect.eval.threshold_optimizer import ThresholdOptimizer, compute_iou


def test_compute_iou_operating_point():
    """Verify IoU calculation accuracy for bounding box matching."""
    box_a = [10.0, 10.0, 50.0, 50.0]
    box_b = [10.0, 10.0, 50.0, 50.0]
    assert compute_iou(box_a, box_b) == 1.0

    box_c = [100.0, 100.0, 150.0, 150.0]
    assert compute_iou(box_a, box_c) == 0.0


def test_find_operating_points_logic():
    """Verify optimizer selects highest precision threshold satisfying target recall >= 95%."""
    dummy_curve_data = {
        "open": {
            "thresholds": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "recall": np.array([0.98, 0.96, 0.94, 0.85, 0.70]),
            "precision": np.array([0.60, 0.75, 0.82, 0.90, 0.95]),
            "f1": np.array([0.74, 0.84, 0.87, 0.87, 0.80]),
            "tp": np.array([98, 96, 94, 85, 70]),
            "fp": np.array([65, 32, 20, 9, 4]),
            "fn": np.array([2, 4, 6, 15, 30]),
        },
        "ALL": {
            "thresholds": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "recall": np.array([0.98, 0.96, 0.94, 0.85, 0.70]),
            "precision": np.array([0.60, 0.75, 0.82, 0.90, 0.95]),
            "f1": np.array([0.74, 0.84, 0.87, 0.87, 0.80]),
            "tp": np.array([98, 96, 94, 85, 70]),
            "fp": np.array([65, 32, 20, 9, 4]),
            "fn": np.array([2, 4, 6, 15, 30]),
        }
    }

    # Instantiate ThresholdOptimizer without loading model
    optimizer = object.__new__(ThresholdOptimizer)
    operating_points = ThresholdOptimizer.find_operating_points(optimizer, dummy_curve_data, target_recall=0.95)

    op_open = operating_points["open"]
    assert op_open.achieved_recall >= 0.95
    # Should choose threshold 0.2 (recall 0.96 >= 0.95 and precision 0.75 > 0.60)
    assert op_open.selected_threshold == 0.2
    assert op_open.achieved_precision == 0.75


def test_factory_params_and_economic_math():
    """Verify factory economic parameters and non-negative cost calculations."""
    params = FactoryParams()
    assert params.daily_volume == 10000
    assert params.defect_rate == 0.03
    assert params.cost_per_escape_fn_usd > 0
    assert params.operator_daily_cost_usd > 0

    res = StrategyResult(
        name="Test Strategy",
        threshold_description="Test",
        achieved_recall=0.96,
        achieved_precision=0.80,
        escaped_defects_per_day=12.0,
        false_alarms_per_day=50.0,
        total_reinspect_queue_per_day=338.0,
        qc_operators_required=1,
        daily_escape_cost_usd=6000.0,
        daily_labor_cost_usd=120.0,
        daily_total_cost_usd=6120.0,
        monthly_total_cost_usd=134640.0,
        monthly_savings_vs_manual_usd=50000.0,
    )
    assert res.daily_total_cost_usd == res.daily_escape_cost_usd + res.daily_labor_cost_usd

import pytest
from ra_testbed.report.reliability import (
    evaluate,
    DEFAULT_THRESHOLDS,
    ReliabilityReport,
    ReliabilityCheck,
    PASS,
    WARN,
)


def healthy_metrics() -> dict:
    return {"CAGR": 0.08, "MDD": -0.20, "Sharpe": 1.2, "Volatility": 0.12}


class TestEvaluate:
    def test_healthy_metrics_all_pass(self):
        report = evaluate(healthy_metrics())
        assert report.overall == PASS
        assert all(c.status == PASS for c in report.checks)

    def test_deep_drawdown_warns(self):
        m = healthy_metrics()
        m["MDD"] = -0.55
        report = evaluate(m)
        assert report.overall == WARN
        mdd_check = next(c for c in report.checks if c.name == "MDD")
        assert mdd_check.status == WARN

    def test_low_sharpe_warns(self):
        m = healthy_metrics()
        m["Sharpe"] = 0.3
        report = evaluate(m)
        assert next(c for c in report.checks if c.name == "Sharpe").status == WARN

    def test_high_volatility_warns(self):
        m = healthy_metrics()
        m["Volatility"] = 0.40
        report = evaluate(m)
        assert next(c for c in report.checks if c.name == "Volatility").status == WARN

    def test_negative_cagr_warns(self):
        m = healthy_metrics()
        m["CAGR"] = -0.03
        report = evaluate(m)
        assert next(c for c in report.checks if c.name == "CAGR").status == WARN

    def test_boundary_exactly_at_threshold_passes(self):
        # MDD 정확히 -0.40 → "미만"이 아니므로 통과
        m = healthy_metrics()
        m["MDD"] = -0.40
        report = evaluate(m)
        assert next(c for c in report.checks if c.name == "MDD").status == PASS

    def test_overall_warn_if_any_warn(self):
        m = healthy_metrics()
        m["Sharpe"] = 0.1
        report = evaluate(m)
        assert report.overall == WARN

    def test_each_check_carries_rationale(self):
        report = evaluate(healthy_metrics())
        assert all(c.rationale for c in report.checks)

    def test_warnings_property_lists_only_warns(self):
        m = healthy_metrics()
        m["MDD"] = -0.50
        m["Sharpe"] = 0.2
        report = evaluate(m)
        assert len(report.warnings) == 2
        assert {c.name for c in report.warnings} == {"MDD", "Sharpe"}

    def test_missing_metric_is_skipped(self):
        report = evaluate({"CAGR": 0.05})  # 일부 지표 누락
        names = {c.name for c in report.checks}
        assert names == {"CAGR"}

    def test_custom_thresholds(self):
        custom = {"MDD": {"threshold": -0.10, "direction": "min", "rationale": "엄격"}}
        report = evaluate({"MDD": -0.15}, thresholds=custom)
        assert report.overall == WARN

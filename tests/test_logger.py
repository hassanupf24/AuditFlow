# ============================================================
# tests/test_logger.py
# Tests for the Decision Audit Logger
# ============================================================

import json
import pytest
from auditflow.core.logger import AuditLogger, get_logger


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset singleton between tests."""
    AuditLogger.reset_instance()
    yield
    AuditLogger.reset_instance()


class TestAuditLogger:
    """Tests for the core audit logger."""

    def test_singleton_pattern(self):
        """AuditLogger should return the same instance."""
        a = AuditLogger()
        b = AuditLogger()
        assert a is b

    def test_get_logger_returns_singleton(self):
        """get_logger() should return the same instance as AuditLogger()."""
        a = get_logger()
        b = AuditLogger()
        assert a is b

    def test_log_decision_records_event(self):
        """log_decision should append an event."""
        audit = get_logger()
        audit.log_decision(
            module="test",
            action="test_action",
            rationale="Testing rationale",
        )
        assert len(audit.events) == 1
        assert audit.events[0].module == "test"
        assert audit.events[0].action == "test_action"
        assert audit.events[0].rationale == "Testing rationale"

    def test_log_decision_with_details(self):
        """Details dict should be stored correctly."""
        audit = get_logger()
        audit.log_decision(
            module="test",
            action="test",
            rationale="Test",
            column="age",
            details={"strategy": "median", "value": 32.0},
            before_shape=(100, 5),
            after_shape=(100, 5),
        )
        event = audit.events[0]
        assert event.column == "age"
        assert event.details["strategy"] == "median"
        assert event.before_shape == (100, 5)

    def test_track_context_manager(self):
        """track() should group events."""
        audit = get_logger()
        with audit.track("Test Group"):
            audit.log_decision(
                module="test", action="inner", rationale="Inside group",
            )

        # Should have: begin_group, inner, end_group
        assert len(audit.events) == 3
        assert audit.events[0].action == "begin_group"
        assert audit.events[1].group == "Test Group"
        assert audit.events[2].action == "end_group"

    def test_get_events_by_group(self):
        """Filtering by group should work."""
        audit = get_logger()
        with audit.track("GroupA"):
            audit.log_decision(module="test", action="a1", rationale="r1")
        with audit.track("GroupB"):
            audit.log_decision(module="test", action="b1", rationale="r2")

        group_a = audit.get_events_by_group("GroupA")
        # begin_group + a1 + end_group
        assert len(group_a) == 3

    def test_get_events_by_module(self):
        """Filtering by module prefix should work."""
        audit = get_logger()
        audit.log_decision(module="cleaners.missing", action="a", rationale="r")
        audit.log_decision(module="cleaners.outliers", action="b", rationale="r")
        audit.log_decision(module="models.classical", action="c", rationale="r")

        cleaner_events = audit.get_events_by_module("cleaners")
        assert len(cleaner_events) == 2

    def test_to_json(self):
        """JSON export should produce valid JSON."""
        audit = get_logger()
        audit.log_decision(module="test", action="act", rationale="reason")
        json_str = audit.to_json()
        data = json.loads(json_str)
        assert data["total_events"] == 1
        assert data["events"][0]["module"] == "test"

    def test_store_figure(self):
        """Figures should be stored."""
        audit = get_logger()
        audit.store_figure("test_fig", "base64data")
        assert len(audit.figures) == 1
        assert audit.figures[0]["name"] == "test_fig"

    def test_store_metrics(self):
        """Metrics should be stored."""
        audit = get_logger()
        audit.store_metrics("accuracy", 0.95)
        assert audit.metrics["accuracy"] == 0.95

    def test_reset(self):
        """Reset should clear everything."""
        audit = get_logger()
        audit.log_decision(module="test", action="a", rationale="r")
        audit.store_figure("fig", "data")
        audit.store_metrics("m", 1)
        audit.reset()
        assert len(audit.events) == 0
        assert len(audit.figures) == 0
        assert len(audit.metrics) == 0

    def test_summary_output(self):
        """Summary should return a non-empty string."""
        audit = get_logger()
        with audit.track("Test"):
            audit.log_decision(module="test", action="act", rationale="reason")
        summary = audit.summary()
        assert "AUDIT TRAIL SUMMARY" in summary
        assert "Test" in summary

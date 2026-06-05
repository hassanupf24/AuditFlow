import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from auditflow.core.logger import get_logger

@pytest.fixture
def memory_exporter():
    # Setup open telemetry in memory exporter
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    
    yield exporter
    
    trace.set_tracer_provider(None)

def test_telemetry_track_spans(memory_exporter):
    audit = get_logger()
    audit.reset()
    
    with audit.track("Test Pipeline Stage"):
        audit.log_decision(
            module="test.module",
            action="test_action",
            rationale="Testing telemetry integration",
            details={"key": "value"}
        )
    
    spans = memory_exporter.get_finished_spans()
    
    # One span for the track block
    assert len(spans) == 1
    
    span = spans[0]
    assert span.name == "AuditFlow.track: Test Pipeline Stage"
    assert span.attributes["auditflow.group_name"] == "Test Pipeline Stage"
    
    # Check that events were added to the span
    events = span.events
    assert len(events) > 0
    decision_events = [e for e in events if e.name == "AuditDecision"]
    
    # We should have at least the begin_group, the test_action, and end_group
    # but begin_group is logged before the span starts so it won't be in this span.
    # The span contains the inner test_action event.
    assert any(e.attributes.get("auditflow.action") == "test_action" for e in decision_events)
    
    action_event = next(e for e in decision_events if e.attributes.get("auditflow.action") == "test_action")
    assert action_event.attributes["auditflow.module"] == "test.module"
    assert "value" in action_event.attributes["auditflow.details"]

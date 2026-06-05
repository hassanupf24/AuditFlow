# ============================================================
# auditflow/core/logger.py
# The Decision Audit Logger — AuditFlow's core differentiator
# ============================================================

"""
Every function in AuditFlow logs its decisions through this module.
The logger captures:
  - What action was taken
  - Why it was taken (auto-generated rationale)
  - Before/after data snapshots
  - Parameters used
  - Timestamps

Usage:
    from auditflow.core.logger import get_logger

    audit = get_logger()
    audit.log_decision(
        module="cleaners.missing",
        action="impute",
        column="age",
        details={"strategy": "median", "fill_value": 32.0},
        rationale="Column 'age' has 12% missing values and skewness=0.8 "
                  "(right-skewed). Median is more robust than mean for "
                  "skewed distributions.",
        before_shape=(1000, 10),
        after_shape=(1000, 10),
    )

    # Context manager for grouped operations
    with audit.track("Data Cleaning"):
        # ... all log_decision calls inside here are grouped
        pass
"""

import json
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Any, Dict, List, Optional, Tuple, Iterator

from opentelemetry import trace

tracer = trace.get_tracer(__name__)

class AuditEvent:
    """A single logged event in the audit trail."""

    def __init__(
        self,
        module: str,
        action: str,
        rationale: str,
        column: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        before_shape: Optional[Tuple[int, ...]] = None,
        after_shape: Optional[Tuple[int, ...]] = None,
        group: Optional[str] = None,
    ):
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.module = module
        self.action = action
        self.rationale = rationale
        self.column = column
        self.details = details or {}
        self.before_shape = before_shape
        self.after_shape = after_shape
        self.group = group

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d: Dict[str, Any] = {
            "timestamp": self.timestamp,
            "module": self.module,
            "action": self.action,
            "rationale": self.rationale,
            "details": self.details,
        }
        if self.column:
            d["column"] = self.column
        if self.before_shape:
            d["before_shape"] = list(self.before_shape)
        if self.after_shape:
            d["after_shape"] = list(self.after_shape)
        if self.group:
            d["group"] = self.group
        return d

    def __repr__(self) -> str:
        col_str = f" [{self.column}]" if self.column else ""
        return f"[{self.module}]{col_str} {self.action} — {self.rationale}"


class AuditLogger:
    """
    Thread-safe singleton audit logger.

    All AuditFlow modules call get_logger() to obtain the shared instance
    and log their decisions through it. The accumulated events form
    the audit trail that powers the auto-generated report.
    """

    _tls = threading.local()

    def __new__(cls, *args: Any, **kwargs: Any) -> "AuditLogger":
        if not hasattr(cls._tls, "instance"):
            cls._tls.instance = super().__new__(cls)
            cls._tls.instance._initialized = False
        return cls._tls.instance  # type: ignore

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._events: List[AuditEvent] = []
        self._current_group: Optional[str] = None
        self._figures: List[Dict[str, Any]] = []  # {"name": str, "base64": str}
        self._metrics: Dict[str, Any] = {}
        self._initialized = True
        self._lock = threading.Lock()

    # ── Logging ──────────────────────────────────────────────

    def log_decision(
        self,
        module: str,
        action: str,
        rationale: str,
        column: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        before_shape: Optional[Tuple[int, ...]] = None,
        after_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """
        Record a single decision event.

        Parameters
        ----------
        module     : Dotted module path, e.g. "cleaners.missing"
        action     : Short verb, e.g. "impute", "drop_column", "scale"
        rationale  : Human-readable explanation of WHY this was done
        column     : Column name affected (if applicable)
        details    : Dict[str, Any] of parameters / values (e.g. {"strategy": "median"})
        before_shape : DataFrame shape before the operation
        after_shape  : DataFrame shape after the operation
        """
        event = AuditEvent(
            module=module,
            action=action,
            rationale=rationale,
            column=column,
            details=details,
            before_shape=before_shape,
            after_shape=after_shape,
            group=self._current_group,
        )
        with self._lock:
            self._events.append(event)
            
        current_span = trace.get_current_span()
        if current_span and current_span.is_recording():
            attributes = {
                "auditflow.module": module,
                "auditflow.action": action,
                "auditflow.rationale": rationale,
            }
            if column:
                attributes["auditflow.column"] = column
            if details:
                attributes["auditflow.details"] = json.dumps(details, default=str)
            if before_shape:
                attributes["auditflow.before_shape"] = str(before_shape)
            if after_shape:
                attributes["auditflow.after_shape"] = str(after_shape)
            current_span.add_event("AuditDecision", attributes=attributes)

    @contextmanager
    def track(self, group_name: str) -> Iterator["AuditLogger"]:
        """
        Context manager to group related decisions under a label.

        Usage:
            with audit.track("Data Cleaning"):
                df = impute(df)
                df = handle_outliers(df)
            # All events inside are tagged with group="Data Cleaning"
        """
        previous_group = self._current_group
        self._current_group = group_name
        self.log_decision(
            module="core.logger",
            action="begin_group",
            rationale=f"Starting pipeline stage: {group_name}",
        )
        with tracer.start_as_current_span(f"AuditFlow.track: {group_name}") as span:
            span.set_attribute("auditflow.group_name", group_name)
            try:
                yield self
            finally:
                self.log_decision(
                    module="core.logger",
                    action="end_group",
                    rationale=f"Completed pipeline stage: {group_name}",
                )
                self._current_group = previous_group

    # ── Figure & Metric Storage ──────────────────────────────

    def store_figure(self, name: str, base64_png: str) -> None:
        """Store a base64-encoded figure for the report."""
        with self._lock:
            self._figures.append({"name": name, "base64": base64_png})

    def store_metrics(self, key: str, value: Any) -> None:
        """Store evaluation metrics for the report."""
        with self._lock:
            self._metrics[key] = value

    # ── Retrieval ────────────────────────────────────────────

    @property
    def events(self) -> List[AuditEvent]:
        """Return all logged events."""
        return list(self._events)

    @property
    def figures(self) -> List[Dict[str, Any]]:
        """Return all stored figures."""
        return list(self._figures)

    @property
    def metrics(self) -> Dict[str, Any]:
        """Return all stored metrics."""
        return dict(self._metrics)

    def get_events_by_group(self, group: str) -> List[AuditEvent]:
        """Return events filtered by group name."""
        return [e for e in self._events if e.group == group]

    def get_events_by_module(self, module: str) -> List[AuditEvent]:
        """Return events filtered by module prefix."""
        return [e for e in self._events if e.module.startswith(module)]

    # ── Export ────────────────────────────────────────────────

    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export the full audit trail as JSON.

        Parameters
        ----------
        filepath : If given, write to file. Otherwise, return string.
        """
        data = {
            "auditflow_version": "0.1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(self._events),
            "events": [e.to_dict() for e in self._events],
        }
        json_str = json.dumps(data, indent=2, default=str)
        if filepath:
            Path(filepath).write_text(json_str, encoding="utf-8")
        return json_str

    def generate_report(
        self,
        output_path: str = "auditflow_report.html",
        title: str = "AuditFlow Analysis Report",
        show_audit_trail: bool = True,
    ) -> str:
        """
        Generate a self-contained HTML report.

        Parameters
        ----------
        output_path      : Path to write the HTML file.
        title            : Title for the report.
        show_audit_trail : Whether to include the audit trail.

        Returns
        -------
        str — the output file path
        """
        from auditflow.core.report import ReportGenerator

        generator = ReportGenerator(self)
        return generator.generate(
            output_path, title=title, show_audit_trail=show_audit_trail
        )

    def summary(self) -> str:
        """Print a human-readable summary of the audit trail."""
        lines = [
            "=" * 60,
            "📋 AUDITFLOW AUDIT TRAIL SUMMARY",
            "=" * 60,
            f"Total events: {len(self._events)}",
            f"Total figures: {len(self._figures)}",
            "",
        ]

        # Group events
        groups: Dict[str, List[AuditEvent]] = {}
        ungrouped = []
        for event in self._events:
            if event.action in ("begin_group", "end_group"):
                continue
            if event.group:
                groups.setdefault(event.group, []).append(event)
            else:
                ungrouped.append(event)

        for group_name, events in groups.items():
            lines.append(f"── {group_name} ({len(events)} decisions) ──")
            for e in events:
                lines.append(f"  • {e}")
            lines.append("")

        if ungrouped:
            lines.append(f"── Ungrouped ({len(ungrouped)} decisions) ──")
            for e in ungrouped:
                lines.append(f"  • {e}")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ── Reset ────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all events, figures, and metrics. Used in tests."""
        with self._lock:
            self._events.clear()
            self._figures.clear()
            self._metrics.clear()
            self._current_group = None

    @classmethod
    def reset_instance(cls) -> None:
        """Destroy the singleton instance for the current thread. Used in tests."""
        if hasattr(cls._tls, "instance"):
            cls._tls.instance._initialized = False
            del cls._tls.instance


def get_logger() -> AuditLogger:
    """
    Get the global AuditLogger singleton.

    This is the primary entry point for all AuditFlow modules.
    Every module calls get_logger() and logs its decisions through it.
    """
    return AuditLogger()

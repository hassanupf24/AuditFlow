# ============================================================
# auditflow/pipeline/audit.py
# Structured audit trail export
# ============================================================

"""
Aggregates all events from the AuditLogger into a structured timeline
with support for JSON and HTML export.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Any

if TYPE_CHECKING:
    from auditflow.core.logger import AuditLogger


class AuditTrail:
    """
    Aggregates and exports the audit trail.

    Usage:
        from auditflow.pipeline.audit import AuditTrail
        from auditflow.core.logger import get_logger

        trail = AuditTrail(get_logger())
        trail.export_json("audit_trail.json")
        print(trail.summary())
    """

    def __init__(self, logger: "AuditLogger"):
        self.logger = logger

    def get_timeline(self) -> List[Dict[str, Any]]:
        """
        Return the full audit trail as a list of dicts, grouped by stage.
        """
        events = self.logger.events
        timeline = []
        current_group = None

        for event in events:
            if event.action == "begin_group":
                current_group = event.rationale.replace("Starting pipeline stage: ", "")
                continue
            if event.action == "end_group":
                current_group = None
                continue

            entry = event.to_dict()
            entry["stage"] = current_group or "ungrouped"
            timeline.append(entry)

        return timeline

    def get_stages(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return events organized by pipeline stage."""
        stages: Dict[str, Any] = {}
        for entry in self.get_timeline():
            stage = entry.get("stage", "ungrouped")
            stages.setdefault(stage, []).append(entry)
        return stages

    def export_json(self, filepath: str) -> str:
        """Export the full audit trail as a JSON file."""
        data = {
            "auditflow_version": "0.1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_decisions": len(self.get_timeline()),
            "stages": self.get_stages(),
        }
        path = Path(filepath)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return str(path.absolute())

    def summary(self) -> str:
        """Generate a human-readable summary."""
        stages = self.get_stages()
        lines = [
            "=" * 60,
            "📋 AUDIT TRAIL SUMMARY",
            "=" * 60,
        ]

        for stage, events in stages.items():
            lines.append(f"\n── {stage} ({len(events)} decisions) ──")
            for e in events:
                col_str = f" [{e.get('column', '')}]" if e.get("column") else ""
                lines.append(f"  • {e['action']}{col_str}: {e['rationale']}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

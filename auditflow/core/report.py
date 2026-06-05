# ============================================================
# auditflow/core/report.py
# Auto-generated HTML report with embedded audit trail
# ============================================================

"""
Consumes the AuditLogger's events, figures, and metrics to produce
a single self-contained HTML file. No external CSS or JS dependencies —
everything is inline, including base64-encoded images.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Template

if TYPE_CHECKING:
    from auditflow.core.logger import AuditLogger


# ── HTML Template ────────────────────────────────────────────

REPORT_TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --card: #21242f;
    --border: #2d3040;
    --text: #e4e6ef;
    --text-muted: #9194a5;
    --accent: #6c63ff;
    --accent-soft: rgba(108,99,255,0.15);
    --green: #4ade80;
    --green-soft: rgba(74,222,128,0.15);
    --amber: #fbbf24;
    --amber-soft: rgba(251,191,36,0.15);
    --red: #f87171;
    --red-soft: rgba(248,113,113,0.15);
    --blue: #60a5fa;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
  }
  .container { max-width: 1100px; margin: 0 auto; }
  h1 {
    font-size: 2rem;
    background: linear-gradient(135deg, var(--accent), var(--blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
  }
  .subtitle { color: var(--text-muted); font-size: 0.95rem; margin-bottom: 2rem; }
  h2 {
    font-size: 1.35rem;
    color: var(--text);
    margin: 2.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }
  h3 { font-size: 1.1rem; color: var(--blue); margin: 1.5rem 0 0.75rem 0; }

  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
  }
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin-bottom: 1.5rem;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    text-align: center;
  }
  .stat-card .value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--accent);
  }
  .stat-card .label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Audit Trail */
  .timeline { position: relative; padding-left: 2rem; }
  .timeline::before {
    content: '';
    position: absolute;
    left: 0.5rem;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--border);
  }
  .event {
    position: relative;
    margin-bottom: 1.25rem;
    padding: 0.75rem 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    border-left: 3px solid var(--accent);
  }
  .event::before {
    content: '';
    position: absolute;
    left: -1.7rem;
    top: 1rem;
    width: 10px;
    height: 10px;
    background: var(--accent);
    border-radius: 50%;
  }
  .event.group-start { border-left-color: var(--green); }
  .event.group-start::before { background: var(--green); }
  .event .module { font-size: 0.75rem; color: var(--text-muted); }
  .event .action {
    font-weight: 600;
    color: var(--blue);
    font-size: 0.9rem;
  }
  .event .rationale {
    margin-top: 0.25rem;
    color: var(--text);
    font-size: 0.88rem;
  }
  .event .details {
    margin-top: 0.4rem;
    font-size: 0.8rem;
    color: var(--text-muted);
    font-family: 'Consolas', 'Courier New', monospace;
    background: var(--bg);
    padding: 0.4rem 0.6rem;
    border-radius: 4px;
  }

  /* Figures */
  .figure-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
    gap: 1rem;
  }
  .figure-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }
  .figure-card img {
    width: 100%;
    display: block;
  }
  .figure-card .caption {
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
    color: var(--text-muted);
  }

  /* Metrics table */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
  }
  th, td {
    padding: 0.6rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }
  th {
    background: var(--surface);
    color: var(--text-muted);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  td { font-size: 0.9rem; }
  tr:hover td { background: var(--accent-soft); }

  .badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .badge-green { background: var(--green-soft); color: var(--green); }
  .badge-amber { background: var(--amber-soft); color: var(--amber); }
  .badge-red   { background: var(--red-soft);   color: var(--red); }

  .footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.8rem;
    color: var(--text-muted);
    text-align: center;
  }
</style>
</head>
<body>
<div class="container">

  <h1>{{ title }}</h1>
  <p class="subtitle">Generated by AuditFlow v{{ version }} on {{ generated_at }}</p>

  <!-- Executive Summary -->
  <h2>📊 Executive Summary</h2>
  <div class="stat-grid">
    <div class="stat-card">
      <div class="value">{{ total_events }}</div>
      <div class="label">Decisions Logged</div>
    </div>
    <div class="stat-card">
      <div class="value">{{ total_figures }}</div>
      <div class="label">Visualizations</div>
    </div>
    <div class="stat-card">
      <div class="value">{{ total_groups }}</div>
      <div class="label">Pipeline Stages</div>
    </div>
    <div class="stat-card">
      <div class="value">{{ total_metrics }}</div>
      <div class="label">Metrics Tracked</div>
    </div>
  </div>

  <!-- Metrics -->
  {% if metrics %}
  <h2>📈 Model Performance</h2>
  <div class="card">
  {% for key, value in metrics.items() %}
    {% if value is mapping %}
    <h3>{{ key }}</h3>
    <table>
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
      {% for mk, mv in value.items() %}
        <tr>
          <td>{{ mk }}</td>
          <td>
            {% if mv is number %}
              {{ "%.4f"|format(mv) }}
              {% if mv >= 0.9 %}<span class="badge badge-green">Excellent</span>
              {% elif mv >= 0.7 %}<span class="badge badge-amber">Good</span>
              {% elif mv < 0.5 %}<span class="badge badge-red">Low</span>{% endif %}
            {% else %}{{ mv }}{% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% else %}
    <p><strong>{{ key }}:</strong> {{ value }}</p>
    {% endif %}
  {% endfor %}
  </div>
  {% endif %}

  <!-- Figures -->
  {% if figures %}
  <h2>📉 Visualizations</h2>
  <div class="figure-grid">
    {% for fig in figures %}
    <div class="figure-card">
      <img src="data:image/png;base64,{{ fig.base64 }}" alt="{{ fig.name }}">
      <div class="caption">{{ fig.name }}</div>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- Audit Trail -->
  {% if show_audit_trail %}
  <h2>🔍 Decision Audit Trail</h2>
  <p style="color: var(--text-muted); margin-bottom: 1rem;">
    Every decision made during this analysis, in chronological order.
  </p>
  <div class="timeline">
    {% for event in events %}
    {% if event.action not in ['begin_group', 'end_group'] %}
    <div class="event">
      <div class="module">{{ event.module }}{% if event.column %} → {{ event.column }}{% endif %}</div>
      <div class="action">{{ event.action }}</div>
      <div class="rationale">{{ event.rationale }}</div>
      {% if event.details %}
      <div class="details">{{ event.details }}</div>
      {% endif %}
    </div>
    {% endif %}
    {% endfor %}
  </div>
  {% endif %}

  <div class="footer">
    AuditFlow v{{ version }} · {{ total_events }} decisions logged · Report generated {{ generated_at }}
  </div>

</div>
</body>
</html>""")


class ReportGenerator:
    """
    Generates a self-contained HTML report from the audit logger's data.

    The report includes:
      - Executive summary (event/figure/metric counts)
      - Model performance metrics with color-coded badges
      - All visualizations (base64-embedded PNGs)
      - Full decision audit trail as a visual timeline
    """

    def __init__(self, logger: "AuditLogger"):
        self.logger = logger

    def generate(
        self,
        output_path: str = "auditflow_report.html",
        title: str = "AuditFlow Analysis Report",
        show_audit_trail: bool = True,
    ) -> str:
        """
        Render and write the HTML report.

        Parameters
        ----------
        output_path      : Where to write the HTML file.
        title            : Report title.
        show_audit_trail : Include the full decision timeline.

        Returns
        -------
        str — the absolute path of the generated report.
        """
        events = self.logger.events
        figures = self.logger.figures
        metrics = self.logger.metrics

        # Count unique groups
        groups = set()
        for e in events:
            if e.group and e.action not in ("begin_group", "end_group"):
                groups.add(e.group)

        # Serialize events for the template
        event_dicts = []
        for e in events:
            d = e.to_dict()
            if d.get("details"):
                # Format details as key=value pairs
                detail_str = " · ".join(f"{k}={v}" for k, v in d["details"].items())
                d["details"] = detail_str
            else:
                d["details"] = None
            event_dicts.append(d)

        html = REPORT_TEMPLATE.render(
            title=title,
            version="0.1.0",
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            total_events=len(
                [e for e in events if e.action not in ("begin_group", "end_group")]
            ),
            total_figures=len(figures),
            total_groups=len(groups),
            total_metrics=sum(
                len(v) if isinstance(v, dict) else 1 for v in metrics.values()
            ),
            metrics=metrics,
            figures=figures,
            events=event_dicts,
            show_audit_trail=show_audit_trail,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")

        self.logger.log_decision(
            module="core.report",
            action="generate_report",
            rationale=f"Generated HTML report with {len(event_dicts)} events, "
            f"{len(figures)} figures at: {out.absolute()}",
        )

        return str(out.absolute())

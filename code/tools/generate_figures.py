"""Generate Plotly figures from ClaimLedger run artifacts and local evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from claimledger.evaluation.benchmark import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent

SYSTEM_ORDER = [
    "claimledger",
    "naive",
    "recency_memory",
    "summary_memory",
    "vector_rag",
    "structured_fact_store",
    "temporal_only",
    "provenance_only",
    "no_conflict",
    "no_temporal",
    "trust_only",
    "no_ledger_oracle_labels",
]

TASK_ORDER = [
    "realtalk_memory_probe",
    "direct_conflict",
    "poisoned_write",
    "temporal_overlap",
]

TASK_LABELS = {
    "realtalk_memory_probe": "Memory probe",
    "direct_conflict": "Conflict",
    "poisoned_write": "Poison",
    "temporal_overlap": "Temporal",
}

PROFILE_SYSTEMS = ["claimledger", "naive", "no_conflict", "no_temporal", "trust_only", "no_ledger_oracle_labels"]

# --- Print-size discipline (IEEE two-column: \textwidth ~7.16in, \columnwidth ~3.5in)
# Export canvases are chosen per figure so that, at final printed width, every
# font is >= 7.5pt: span figures export at 1032px (printed at 7.16in -> 0.5
# scale, 16px font -> 8pt); column figures export at 700px (printed at 3.5in ->
# 0.36 scale, 22px font -> ~8pt). In-figure titles are omitted; LaTeX captions
# carry them.
SPAN_WIDTH = 1032
SPAN_FONT = 16
COL_WIDTH = 700
COL_FONT = 22

SHORT_SYSTEM_LABELS = {
    "claimledger": "ClaimLedger",
    "naive": "Naive",
    "recency_memory": "Recency",
    "summary_memory": "Summary",
    "vector_rag": "Vector RAG",
    "structured_fact_store": "Struct. store",
    "temporal_only": "Temporal-only",
    "provenance_only": "Provenance-only",
    "no_conflict": "No-conflict",
    "no_temporal": "No-temporal",
    "trust_only": "Trust-only",
    "no_ledger_oracle_labels": "No-ledger",
}


def _short(name: Any) -> str:
    return SHORT_SYSTEM_LABELS.get(str(name), str(name))


def _chat_number(conversation_id: str) -> int | None:
    match = re.search(r"Chat[_ ](\d+)", conversation_id, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _chat_label(conversation_id: str) -> str:
    number = _chat_number(conversation_id)
    return f"C{number}" if number is not None else conversation_id


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_metric_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/metrics.jsonl")):
        rows.extend(_read_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"no metrics.jsonl files found under {run_root}")
    return rows


def _load_case_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_root.glob("*/per_case.jsonl")):
        rows.extend(_read_jsonl(path))
    if not rows:
        raise FileNotFoundError(f"no per_case.jsonl files found under {run_root}")
    return rows


def _conversation_id(case_id: str) -> str:
    for marker in ("_q", "_temporal"):
        if marker in case_id:
            return case_id.split(marker, 1)[0]
    return case_id


def _aggregate_metric(
    rows: list[dict[str, Any]],
    *,
    system: str,
    metric: str,
    dataset: str | None = None,
) -> float | None:
    selected = [
        row
        for row in rows
        if row["system"] == system
        and row["metric"] == metric
        and (dataset is None or row["dataset"] == dataset)
    ]
    denominator = sum(int(row["denominator"]) for row in selected)
    numerator = sum(int(row["numerator"]) for row in selected)
    return numerator / denominator if denominator else None


def _seed_values(rows: list[dict[str, Any]], system: str, metric: str, dataset: str) -> list[float]:
    return [
        float(row["value"])
        for row in rows
        if row["system"] == system and row["metric"] == metric and row["dataset"] == dataset
    ]


def _save_figure(
    fig: go.Figure,
    name: str,
    output_dir: Path,
    export_errors: list[dict[str, str]],
    *,
    width: int = SPAN_WIDTH,
    height: int = 430,
    font_size: int = SPAN_FONT,
    margin: dict[str, int] | None = None,
) -> None:
    fig.update_layout(
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": font_size},
        margin=margin or {"l": 10, "r": 10, "t": 10 if not fig.layout.title.text else 40, "b": 100},
        hoverlabel={"font": {"family": "Arial, sans-serif"}},
    )
    fig.write_html(output_dir / f"{name}.html", include_plotlyjs="cdn", full_html=True)
    try:
        fig.write_image(output_dir / f"{name}.png", width=width, height=height, scale=4)
        fig.write_image(output_dir / f"{name}.pdf", width=width, height=height)
    except Exception as exc:  # static export depends on a local browser engine
        export_errors.append({"figure": name, "error": str(exc)})


def _composition_figure(cases_path: Path, output_dir: Path, errors: list[dict[str, str]]) -> None:
    sources = {
        "REALTALK memory probes": cases_path,
    }
    counts: dict[str, Counter[str]] = {}
    for label, path in sources.items():
        if path.exists():
            counts[label] = Counter(case.task_type for case in load_jsonl(path))
    task_types = sorted({task for counter in counts.values() for task in counter})
    fig = go.Figure()
    for task in task_types:
        fig.add_trace(
            go.Bar(
                name=task,
                x=list(counts),
                y=[counts[label][task] for label in counts],
                hovertemplate="%{x}<br>Task: " + task + "<br>Cases: %{y}<extra></extra>",
            )
        )
    for trace in fig.data:
        trace.name = TASK_LABELS.get(str(trace.name), str(trace.name))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Stress suite", showgrid=False)
    fig.update_yaxes(title_text="Number of cases")
    _save_figure(fig, "figure_benchmark_composition", output_dir, errors, width=COL_WIDTH, height=330, font_size=COL_FONT, margin={"l": 85, "r": 15, "t": 15, "b": 80})


def _error_bar_figure(rows: list[dict[str, Any]], output_dir: Path, errors: list[dict[str, str]]) -> None:
    datasets = sorted({str(row["dataset"]) for row in rows})
    fig = go.Figure()
    for dataset in datasets:
        means: list[float] = []
        lower: list[float] = []
        upper: list[float] = []
        labels: list[str] = []
        for system in SYSTEM_ORDER:
            values = _seed_values(rows, system, "exact_active_count_accuracy", dataset)
            if not values:
                continue
            labels.append(_short(system))
            mean = sum(values) / len(values)
            means.append(mean)
            lower.append(mean - min(values))
            upper.append(max(values) - mean)
        fig.add_trace(
            go.Scatter(
                name=dataset,
                x=labels,
                y=means,
                mode="markers",
                error_y={"type": "data", "symmetric": False, "array": upper, "arrayminus": lower},
                hovertemplate="System: %{x}<br>Accuracy: %{y:.3f}<extra>" + dataset + "</extra>",
            )
        )
    dataset_labels = {
        "realtalk_memory_probes": "Memory probes",
        "temporal_overlap": "Temporal",
    }
    for trace in fig.data:
        trace.name = dataset_labels.get(str(trace.name), str(trace.name))
        trace.marker = {"size": 7}
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-30)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], dtick=0.2, gridcolor="#E5E5E5")
    _save_figure(
        fig,
        "figure_error_bars",
        output_dir,
        errors,
        width=COL_WIDTH,
        height=380,
        font_size=COL_FONT,
        margin={"l": 85, "r": 15, "t": 15, "b": 70},
    )


def _radar_figure(rows: list[dict[str, Any]], output_dir: Path, errors: list[dict[str, str]]) -> None:
    metric_names = [
        ("exact_active_count_accuracy", "Active-count<br>accuracy", False),
        ("temporal_validity_accuracy", "Temporal<br>validity", False),
        ("conflict_disclosure_rate", "Conflict<br>disclosure", False),
        ("abstention_correctness", "Abstention<br>correctness", False),
        ("poison_activation_rate", "Poison safety", True),
    ]
    selected = PROFILE_SYSTEMS
    axes = [label for _, label, _ in metric_names]
    fig = go.Figure()
    for system in selected:
        values: list[float] = []
        for metric, _, invert in metric_names:
            value = _aggregate_metric(rows, system=system, metric=metric)
            values.append((1.0 - value) if invert and value is not None else (value or 0.0))
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=axes + [axes[0]],
                name=_short(system),
                fill="toself",
            )
        )
    fig.update_layout(
        polar={
            "radialaxis": {"visible": True, "range": [0, 1], "tickfont": {"size": COL_FONT - 4}},
            "angularaxis": {"rotation": 90, "direction": "clockwise"},
        },
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.12,
            "xanchor": "center",
            "x": 0.5,
            "title": None,
        },
    )
    _save_figure(
        fig,
        "figure_radar_profile",
        output_dir,
        errors,
        width=COL_WIDTH,
        height=505,
        font_size=COL_FONT - 2,
        margin={"l": 100, "r": 100, "t": 62, "b": 130},
    )


def _heatmap_figure(case_rows: list[dict[str, Any]], output_dir: Path, errors: list[dict[str, str]]) -> None:
    systems = [system for system in SYSTEM_ORDER if any(row["system"] == system for row in case_rows)]
    z: list[list[float | None]] = []
    for system in systems:
        values: list[float | None] = []
        for task_type in TASK_ORDER:
            selected = [
                row for row in case_rows
                if row["system"] == system and row["task_type"] == task_type
            ]
            values.append(
                sum(bool(row["exact_active_correct"]) for row in selected) / len(selected)
                if selected else None
            )
        z.append(values)
    fig = go.Figure(
        go.Heatmap(
            y=[_short(system) for system in systems],
            x=[TASK_LABELS[task] for task in TASK_ORDER],
            z=z,
            zmin=0,
            zmax=1,
            colorscale="Blues",
            colorbar={"title": "Accuracy", "thickness": 12},
            hovertemplate="System: %{y}<br>Task: %{x}<br>Accuracy: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(side="top", showgrid=False, tickangle=-30, tickfont={"size": 20})
    fig.update_yaxes(showgrid=False, tickfont={"size": 20})
    _save_figure(
        fig,
        "figure_method_heatmap",
        output_dir,
        errors,
        width=COL_WIDTH,
        height=430,
        font_size=20,
        margin={"l": 150, "r": 95, "t": 95, "b": 20},
    )


def _active_count_profile_figure(case_rows: list[dict[str, Any]], output_dir: Path, errors: list[dict[str, str]]) -> None:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[TASK_LABELS[task] for task in TASK_ORDER],
        shared_yaxes=True,
        vertical_spacing=0.25,
        horizontal_spacing=0.10,
    )
    for index, task_type in enumerate(TASK_ORDER):
        row, column = divmod(index, 2)
        row, column = row + 1, column + 1
        selected = [
            row_
            for row_ in case_rows
            if row_["task_type"] == task_type and row_["system"] in PROFILE_SYSTEMS
        ]
        values = []
        expected_values = []
        for system in PROFILE_SYSTEMS:
            system_rows = [row_ for row_ in selected if row_["system"] == system]
            values.append(sum(float(row_["actual_active"]) for row_ in system_rows) / len(system_rows) if system_rows else 0.0)
            expected_values.append(
                sum(float(row_["expected_active"]) for row_ in system_rows) / len(system_rows) if system_rows else 0.0
            )
        fig.add_trace(
            go.Bar(
                x=[_short(system) for system in PROFILE_SYSTEMS],
                y=values,
                name="Actual active claims",
                legendgroup="actual",
                showlegend=index == 0,
                marker_color="#0077BB",
                hovertemplate="System: %{x}<br>Mean actual active claims: %{y:.3f}<extra></extra>",
            ),
            row=row,
            col=column,
        )
        fig.add_trace(
            go.Scatter(
                x=[_short(system) for system in PROFILE_SYSTEMS],
                y=expected_values,
                name="Expected active claims",
                legendgroup="expected",
                showlegend=index == 0,
                mode="lines+markers",
                line={"color": "#CC3311", "dash": "dash"},
                hovertemplate="System: %{x}<br>Mean expected active claims: %{y:.3f}<extra></extra>",
            ),
            row=row,
            col=column,
        )
    fig.update_xaxes(tickangle=-35, tickfont={"size": 11}, automargin=True)
    fig.update_yaxes(range=[0, 2.1], dtick=0.5)
    fig.update_yaxes(title_text="Mean active claims", row=1, col=1)
    fig.update_yaxes(title_text="Mean active claims", row=2, col=1)
    fig.update_layout(
        barmode="group",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.26,
            "xanchor": "center",
            "x": 0.5,
            "title": None,
            "traceorder": "normal",
        },
    )
    _save_figure(
        fig,
        "figure_active_claim_counts",
        output_dir,
        errors,
        height=500,
        margin={"l": 75, "r": 20, "t": 70, "b": 120},
    )


def _conversation_accuracy_figure(case_rows: list[dict[str, Any]], output_dir: Path, errors: list[dict[str, str]]) -> None:
    conversations = sorted(
        ({_conversation_id(str(row["case_id"])) for row in case_rows}),
        key=lambda value: (_chat_number(value) is None, _chat_number(value) or 0),
    )
    fig = go.Figure()
    for system in PROFILE_SYSTEMS:
        values = []
        for conversation in conversations:
            selected = [
                row for row in case_rows
                if row["system"] == system and _conversation_id(str(row["case_id"])) == conversation
            ]
            values.append(sum(bool(row["exact_active_correct"]) for row in selected) / len(selected) if selected else None)
        fig.add_trace(go.Scatter(x=conversations, y=values, mode="lines+markers", name=_short(system)))
    chat_labels = [_chat_label(conversation) for conversation in conversations]
    fig.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.12,
            "xanchor": "center",
            "x": 0.5,
            "title": None,
        },
    )
    fig.update_xaxes(tickvals=conversations, ticktext=chat_labels, title_text="REALTALK conversation group")
    fig.update_yaxes(title_text="Exact active-count accuracy", range=[0, 1.05], dtick=0.2, gridcolor="#E5E5E5")
    _save_figure(fig, "figure_conversation_accuracy", output_dir, errors, width=COL_WIDTH, height=360, font_size=COL_FONT, margin={"l": 85, "r": 15, "t": 75, "b": 85})


def _paired_difference_figure(run_root: Path, output_dir: Path, errors: list[dict[str, str]]) -> None:
    """Horizontal bar chart of exact-accuracy difference (ClaimLedger - baseline).

    Sorted by magnitude with a dashed zero reference line and per-bar value
    labels. Exported via _save_figure (PNG at high scale + PDF); the paper embeds
    the PNG so the bottom axis label is never clipped by the PDF MediaBox.
    """
    paths = sorted(run_root.glob("*/paired_tests.json"))
    if not paths:
        return
    tests = json.loads(paths[0].read_text(encoding="utf-8"))
    tests = sorted(
        tests,
        key=lambda item: SYSTEM_ORDER.index(item["baseline"])
        if item["baseline"] in SYSTEM_ORDER
        else len(SYSTEM_ORDER),
    )
    labels = [_short(item["baseline"]) for item in tests]
    differences = [
        float(item["system_accuracy"]) - float(item["baseline_accuracy"]) for item in tests
    ]
    wins = [int(item["system_wins"]) for item in tests]
    n_total = int(tests[0].get("n", 2387)) if tests else 2387
    order = sorted(range(len(labels)), key=lambda i: differences[i])
    labels_s = [labels[i] for i in order]
    diff_s = [differences[i] for i in order]
    wins_s = [wins[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=diff_s,
            y=labels_s,
            orientation="h",
            width=0.78,
            marker={
                "color": ["#0077BB" if d >= 0 else "#CC3311" for d in diff_s],
                "line": {"color": "white", "width": 0.5},
            },
            customdata=[[w, n_total] for w in wins_s],
            hovertemplate=(
                "Baseline: %{y}<br>Accuracy difference: %{x:.3f}<br>"
                "ClaimLedger wins: %{customdata[0]} / %{customdata[1]}<extra></extra>"
            ),
            text=[f"{d:+.3f}" for d in diff_s],
            textposition="outside",
            textfont={"size": COL_FONT - 1, "color": "#222222"},
            cliponaxis=False,
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#444444", line_width=1.5)
    # Bounded axis range (no negative margin / no symmetric forced bound)
    fig.update_xaxes(
        title_text="Exact-accuracy difference (ClaimLedger - baseline)",
        title_font={"family": "Arial, sans-serif", "size": COL_FONT - 1},
        range=[min(diff_s) - 0.06, max(diff_s) + 0.10],
        zeroline=False,
        gridcolor="#EDEDED",
        ticks="outside",
        ticklen=4,
    )
    fig.update_yaxes(title_text="", automargin=False)
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    _save_figure(
        fig,
        "figure_paired_accuracy_differences",
        output_dir,
        errors,
        width=COL_WIDTH,
        height=520,
        font_size=COL_FONT,
        margin={"l": 210, "r": 120, "t": 15, "b": 90},
    )
    print(f"[paired-difference] wrote figure_paired_accuracy_differences ({len(labels_s)} baselines, N={n_total})")



def _performance_storage_figure(performance_path: Path, output_dir: Path, errors: list[dict[str, str]]) -> None:
    if not performance_path.exists():
        return
    records = json.loads(performance_path.read_text(encoding="utf-8"))
    scales = [record["scale"] for record in records]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Write latency", "Query latency", "Storage", "Storage per claim"),
        vertical_spacing=0.26,
        horizontal_spacing=0.13,
    )
    fig.add_trace(go.Scatter(x=scales, y=[r["write_latency_ms_p50"] for r in records], mode="lines+markers", name="Write p50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=scales, y=[r["write_latency_ms_p95"] for r in records], mode="lines+markers", name="Write p95"), row=1, col=1)
    fig.add_trace(go.Scatter(x=scales, y=[r["query_latency_ms_p50"] for r in records], mode="lines+markers", name="Query p50"), row=1, col=2)
    fig.add_trace(go.Scatter(x=scales, y=[r["query_latency_ms_p95"] for r in records], mode="lines+markers", name="Query p95"), row=1, col=2)
    fig.add_trace(go.Scatter(x=scales, y=[r["storage_bytes"] for r in records], mode="lines+markers", name="Storage bytes"), row=2, col=1)
    fig.add_trace(go.Scatter(x=scales, y=[r["storage_bytes_per_claim"] for r in records], mode="lines+markers", name="Bytes per claim"), row=2, col=2)
    ticktext = [f"{int(scale):,}" for scale in scales]
    for row in (1, 2):
        for column in (1, 2):
            fig.update_xaxes(
                title_text="Claims (log scale)" if row == 2 else None,
                type="log",
                tickvals=scales,
                ticktext=ticktext,
                row=row,
                col=column,
            )
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=2)
    fig.update_yaxes(title_text="Bytes", row=2, col=1)
    fig.update_yaxes(title_text="Bytes / claim", row=2, col=2)
    # Single horizontal legend above the grid so it cannot overlap the lower
    # row; in-figure title omitted (LaTeX caption carries it).
    fig.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.13,
            "xanchor": "center",
            "x": 0.5,
        },
    )
    _save_figure(fig, "figure_performance_storage", output_dir, errors, height=440, margin={"l": 85, "r": 25, "t": 75, "b": 75})




def _write_derived_tables(case_rows: list[dict[str, Any]], performance_path: Path, output_dir: Path) -> None:
    table_dir = output_dir.parent
    task_path = table_dir / "realtalk_task_metrics.csv"
    with task_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "task_type", "task_label", "numerator", "denominator", "accuracy"])
        writer.writeheader()
        for system in SYSTEM_ORDER:
            for task_type in TASK_ORDER:
                selected = [row for row in case_rows if row["system"] == system and row["task_type"] == task_type]
                if selected:
                    numerator = sum(bool(row["exact_active_correct"]) for row in selected)
                    writer.writerow({"system": system, "task_type": task_type, "task_label": TASK_LABELS[task_type], "numerator": numerator, "denominator": len(selected), "accuracy": numerator / len(selected)})

    conversation_path = table_dir / "realtalk_conversation_metrics.csv"
    conversations = sorted({_conversation_id(str(row["case_id"])) for row in case_rows})
    with conversation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "conversation", "numerator", "denominator", "accuracy"])
        writer.writeheader()
        for system in PROFILE_SYSTEMS:
            for conversation in conversations:
                selected = [row for row in case_rows if row["system"] == system and _conversation_id(str(row["case_id"])) == conversation]
                numerator = sum(bool(row["exact_active_correct"]) for row in selected)
                writer.writerow({"system": system, "conversation": conversation, "numerator": numerator, "denominator": len(selected), "accuracy": numerator / len(selected) if selected else None})

    if performance_path.exists():
        performance = json.loads(performance_path.read_text(encoding="utf-8"))
        performance_table = table_dir / "performance_measurements.csv"
        with performance_table.open("w", newline="", encoding="utf-8") as handle:
            if performance:
                writer = csv.DictWriter(handle, fieldnames=sorted(performance[0]))
                writer.writeheader()
                writer.writerows(performance)


def _conflict_scatter_figure(cases_path: Path, case_rows: list[dict[str, Any]], output_dir: Path, errors: list[dict[str, str]]) -> None:
    path = cases_path
    if not path.exists():
        return
    cases = load_jsonl(path)
    features: dict[str, dict[str, Any]] = {}
    for case in cases:
        pair_total = max(1, len(case.claims) * (len(case.claims) - 1) // 2)
        distinct_pairs = 0
        by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
        for claim in case.claims:
            by_key[(claim.subject, claim.predicate)].add(claim.object)
        for objects in by_key.values():
            distinct_pairs += len(objects) * (len(objects) - 1) // 2
        features[case.case_id] = {
            "task_type": case.task_type,
            "claim_count": len(case.claims),
            "conflict_density": distinct_pairs / pair_total,
        }
    fig = go.Figure()
    labels = {system: _short(system) for system in PROFILE_SYSTEMS}
    display_offsets = {system: (index - (len(PROFILE_SYSTEMS) - 1) / 2) * 0.015 for index, system in enumerate(PROFILE_SYSTEMS)}
    for system in PROFILE_SYSTEMS:
        selected = [
            row for row in case_rows
            if row["system"] == system and str(row["case_id"]) in features
        ]
        feature_rows = [features[str(row["case_id"])] for row in selected]
        fig.add_trace(
            go.Scatter(
                name=labels.get(system, system),
                x=[feature["conflict_density"] + display_offsets[system] for feature in feature_rows],
                y=[int(row["exact_active_correct"]) for row in selected],
                mode="markers",
                marker={"size": 8, "opacity": 0.55},
                customdata=[
                    [
                        row["case_id"],
                        feature["task_type"],
                        feature["claim_count"],
                        row["conflict_disclosed"],
                    ]
                    for row, feature in zip(selected, feature_rows, strict=True)
                ],
                hovertemplate="Density: %{x:.3f}<br>Exact correct: %{y}<br>Case: %{customdata[0]}<br>Task: %{customdata[1]}<br>Claims: %{customdata[2]}<br>Disclosure: %{customdata[3]}<extra>" + labels.get(system, system) + "</extra>",
            )
        )
    fig.update_xaxes(
        title_text="Conflict density (contradictory pairs / total pairs)",
        range=[-0.12, 1.12],
        tickvals=[0, 1],
        ticktext=["0 (No contradiction)", "1 (Contradiction)"],
        gridcolor="#E5E5E5",
    )
    fig.update_yaxes(
        title_text=None,
        range=[-0.08, 1.08],
        tickvals=[0, 1],
        dtick=1,
        gridcolor="#E5E5E5",
    )
    fig.add_annotation(
        text="Exact active-count correctness",
        xref="paper",
        yref="paper",
        x=-0.14,
        y=0.41,
        textangle=-90,
        showarrow=False,
        font={"family": "Arial, sans-serif", "size": COL_FONT - 3},
    )
    fig.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.5,
            "title": None,
        },
    )
    _save_figure(
        fig,
        "figure_conflict_density_scatter",
        output_dir,
        errors,
        width=COL_WIDTH,
        height=390,
        font_size=COL_FONT - 3,
        margin={"l": 110, "r": 60, "t": 20, "b": 115},
    )


def _extraction_error_figure(output_dir: Path, errors: list[dict[str, str]]) -> None:
    sweep_path = REPOSITORY_ROOT / "data" / "results" / "extraction_error" / "sweep_metrics.jsonl"
    if not sweep_path.exists():
        return
    rows = [json.loads(line) for line in sweep_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        series[row["metric"]].append((row["p_detect"], row["value"]))
    for metric in series:
        series[metric].sort()

    metric_series = [
        ("exact_active_count_accuracy", "Exact acc."),
        ("temporal_validity_accuracy", "Temporal acc."),
        ("conflict_disclosure_rate", "Conflict disclosure"),
        ("poison_activation_rate", "Poison activation"),
    ]
    full_labels = {
        "exact_active_count_accuracy": "Exact active-claim accuracy",
        "temporal_validity_accuracy": "Temporal-validity accuracy",
        "conflict_disclosure_rate": "Conflict disclosure",
        "poison_activation_rate": "Poison activation",
    }
    fig = go.Figure()
    for metric, label in metric_series:
        points = series.get(metric, [])
        fig.add_trace(
            go.Scatter(
                x=[p for p, _ in points],
                y=[v for _, v in points],
                mode="lines+markers",
                name=label,
                hovertemplate="p_detect=%{x:.2f}<br>" + full_labels[metric] + ": %{y:.3f}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 20},
        margin={"l": 85, "r": 15, "t": 65, "b": 85},
        xaxis_title="Extractor detection probability (p_detect)",
        yaxis_title="Metric value",
        yaxis={"range": [-0.03, 1.03], "dtick": 0.2},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 16},
        },
        hoverlabel={"font": {"family": "Arial, sans-serif"}},
    )
    _save_figure(
        fig,
        "figure_extraction_error_degradation",
        output_dir,
        errors,
        width=COL_WIDTH,
        height=360,
        font_size=20,
        margin={"l": 85, "r": 15, "t": 65, "b": 85},
    )


def _multillm_tier2_figures(output_dir: Path, errors: list[dict[str, str]]) -> None:
    summary_path = REPOSITORY_ROOT / "data" / "results" / "runs" / "tier2_summary.csv"
    if not summary_path.exists():
        return
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    with summary_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["dataset"], row["strategy"])].append(row)
    cells: dict[tuple[str, str], dict[str, float]] = {}
    for key, r_list in grouped.items():
        n = len(r_list)
        cells[key] = {
            "token_f1": sum(float(r["token_f1"]) for r in r_list) / n,
            "exact_match": sum(float(r["exact_match"]) for r in r_list) / n,
            "abstention_rate": sum(float(r["abstention_rate"]) for r in r_list) / n,
            "n_questions": sum(int(r["n"]) for r in r_list) // n,
        }

    dataset_order = ["realtalk", "locomo", "longmemeval_s", "longmemeval_m", "longmemeval_oracle"]
    dataset_labels = {
        "realtalk": "REALTALK",
        "locomo": "LoCoMo",
        "longmemeval_s": "LME-S",
        "longmemeval_m": "LME-M",
        "longmemeval_oracle": "LME-Oracle",
    }
    strategy_order = ["claimledger", "lexical", "recency", "full_context"]
    strategy_labels = {
        "claimledger": "ClaimLedger",
        "lexical": "Lexical",
        "recency": "Recency",
        "full_context": "Full-context",
    }
    strategy_colors = {
        "claimledger": "#0077BB",
        "lexical": "#7F7F7F",
        "recency": "#CC79A7",
        "full_context": "#E69F00",
    }
    datasets = [d for d in dataset_order if any((d, s) in cells for s in strategy_order)]
    strategies_present = [s for s in strategy_order if any((d, s) in cells for d in datasets)]

    def _panel(fig, metric, col):
        for strategy in strategies_present:
            xs, ys = [], []
            for d in datasets:
                c = cells.get((d, strategy))
                if c is not None:
                    xs.append(dataset_labels[d])
                    ys.append(c[metric])
            fig.add_trace(
                go.Bar(
                    x=xs,
                    y=ys,
                    name=strategy_labels[strategy],
                    marker_color=strategy_colors[strategy],
                    hovertemplate="%{x}<br>" + strategy_labels[strategy] + f"<br>{metric}=%{{y:.3f}}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        axis_title = "Token F1" if metric == "token_f1" else "Exact match"
        fig.update_yaxes(title_text=axis_title, range=[0, 0.45], row=1, col=col, showgrid=True, gridcolor="#E5E5E5")

    quality = make_subplots(rows=1, cols=2, shared_yaxes=False, horizontal_spacing=0.09, subplot_titles=("Token F1", "Exact match"))
    quality.update_annotations(font_size=SPAN_FONT)
    _panel(quality, "token_f1", 1)
    _panel(quality, "exact_match", 2)
    quality.update_layout(
        font={"family": "Arial, sans-serif", "size": SPAN_FONT},
        margin={"l": 70, "r": 24, "t": 64, "b": 56},
        barmode="group",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.16, "x": 0},
    )
    for trace in quality.data[len(quality.data) // 2:]:
        trace.showlegend = False
    _save_figure(quality, "figure_multillm_reader_quality", output_dir, errors, width=SPAN_WIDTH, height=430, font_size=SPAN_FONT)

    abst_fig = make_subplots(rows=1, cols=1)
    for strategy in strategies_present:
        xs, ys = [], []
        for d in datasets:
            c = cells.get((d, strategy))
            if c is not None:
                xs.append(dataset_labels[d])
                ys.append(c["abstention_rate"])
        abst_fig.add_trace(
            go.Bar(
                x=xs,
                y=ys,
                name=strategy_labels[strategy],
                marker_color=strategy_colors[strategy],
                hovertemplate="%{x}<br>" + strategy_labels[strategy] + "<br>abstention=%{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
    abst_fig.update_yaxes(title_text="Abstention rate", range=[0, 1.0], tickvals=[0, 0.25, 0.5, 0.75, 1.0], showgrid=True, gridcolor="#E5E5E5")
    abst_fig.update_layout(
        font={"family": "Arial, sans-serif", "size": SPAN_FONT},
        margin={"l": 70, "r": 24, "t": 64, "b": 56},
        barmode="group",
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.16, "x": 0},
    )
    _save_figure(abst_fig, "figure_multillm_abstention", output_dir, errors, width=SPAN_WIDTH, height=430, font_size=SPAN_FONT)


def _retrieval_figures(output_dir: Path, errors: list[dict[str, str]]) -> None:
    agg_path = REPOSITORY_ROOT / "data" / "results" / "runs" / "retrieval-official-benchmarks-20260818-v2" / "retrieval_aggregate.json"
    if not agg_path.exists():
        return
    rows = json.loads(agg_path.read_text(encoding="utf-8"))
    overall_rows = [r for r in rows if r["task_type"] is None]
    datasets = sorted({str(r["dataset"]) for r in overall_rows})
    strategies = ["claimledger", "lexical", "recency", "oracle"]
    z = []
    for strategy in strategies:
        z.append([
            next((float(r["evidence_recall_any"]) for r in overall_rows if r["dataset"] == d and r["strategy"] == strategy), None)
            for d in datasets
        ])
    heatmap = go.Figure(go.Heatmap(z=z, x=datasets, y=strategies, zmin=0, zmax=1, colorscale="Blues", colorbar={"title": "Recall@k"}))
    heatmap.update_layout(xaxis_title="Dataset", yaxis_title="Context strategy")
    _save_figure(heatmap, "figure_retrieval_heatmap", output_dir, errors, width=SPAN_WIDTH, height=430, font_size=SPAN_FONT)


def generate(
    output_dir: Path,
    run_root: Path,
    cases_path: Path,
    performance_path: Path,
    target: str = "all",
) -> list[dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []

    if target in ("all", "mechanism"):
        rows = _load_metric_rows(run_root)
        case_rows = _load_case_rows(run_root)
        _composition_figure(cases_path, output_dir, errors)
        _error_bar_figure(rows, output_dir, errors)
        _radar_figure(rows, output_dir, errors)
        _heatmap_figure(case_rows, output_dir, errors)
        _conflict_scatter_figure(cases_path, case_rows, output_dir, errors)
        _active_count_profile_figure(case_rows, output_dir, errors)
        _conversation_accuracy_figure(case_rows, output_dir, errors)
        _paired_difference_figure(run_root, output_dir, errors)
        _performance_storage_figure(performance_path, output_dir, errors)
        _write_derived_tables(case_rows, performance_path, output_dir)

    if target in ("all", "extraction"):
        _extraction_error_figure(output_dir, errors)

    if target in ("all", "multillm"):
        _multillm_tier2_figures(output_dir, errors)

    if target in ("all", "retrieval"):
        _retrieval_figures(output_dir, errors)

    (output_dir / "figure_export_errors.json").write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")

    paper_figures = REPOSITORY_ROOT / "paper" / "src" / "figures"
    if paper_figures.exists():
        import shutil
        for pdf_file in output_dir.glob("*.pdf"):
            shutil.copy2(pdf_file, paper_figures / pdf_file.name)

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("data/results/runs"))
    parser.add_argument("--output", type=Path, default=Path("data/results/figures"))
    parser.add_argument("--cases", type=Path, default=Path("data/derived/realtalk_memory_probe_cases.jsonl"))
    parser.add_argument("--performance", type=Path, default=Path("data/results/performance_measurements.json"))
    parser.add_argument(
        "--target",
        choices=["all", "mechanism", "extraction", "multillm", "retrieval"],
        default="all",
        help="Figure subset to generate (default: all).",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    runs = args.runs if args.runs.is_absolute() else REPOSITORY_ROOT / args.runs
    cases = args.cases if args.cases.is_absolute() else REPOSITORY_ROOT / args.cases
    performance = args.performance if args.performance.is_absolute() else REPOSITORY_ROOT / args.performance
    errors = generate(output, runs, cases, performance, target=args.target)
    print(json.dumps({"output": str(args.output), "static_export_errors": errors}, indent=2))


if __name__ == "__main__":
    main()

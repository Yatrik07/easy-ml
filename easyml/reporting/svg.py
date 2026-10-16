"""
Small static SVG charts for the HTML run report: no JavaScript, no external files, print-friendly.

Colours come from CSS classes defined in the report (``s0``..``s7``, ``axis``, ``grid``, ``ref``), so the
charts follow the report's style. All text is escaped.
"""

from html import escape

W, H = 440, 300
PAD_L, PAD_R, PAD_T, PAD_B = 52, 14, 14, 42


def _fmt(value, digits=3):
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _scale(lo, hi, a, b):
    span = (hi - lo) or 1.0
    return lambda v: a + (v - lo) / span * (b - a)


def _ticks(lo, hi, count=5):
    if hi == lo:
        return [lo]
    step = (hi - lo) / (count - 1)
    return [lo + i * step for i in range(count)]


def _frame(x_lo, x_hi, y_lo, y_hi, x_label, y_label):
    sx = _scale(x_lo, x_hi, PAD_L, W - PAD_R)
    sy = _scale(y_lo, y_hi, H - PAD_B, PAD_T)
    parts = []
    for t in _ticks(x_lo, x_hi):
        x = sx(t)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{H - PAD_B}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{H - PAD_B + 15}" text-anchor="middle">{_fmt(t, 2)}</text>')
    for t in _ticks(y_lo, y_hi):
        y = sy(t)
        parts.append(f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end">{_fmt(t, 2)}</text>')
    parts.append(f'<line class="axis" x1="{PAD_L}" y1="{H - PAD_B}" x2="{W - PAD_R}" y2="{H - PAD_B}"/>')
    parts.append(f'<line class="axis" x1="{PAD_L}" y1="{PAD_T}" x2="{PAD_L}" y2="{H - PAD_B}"/>')
    parts.append(f'<text class="label" x="{(PAD_L + W - PAD_R) / 2:.1f}" y="{H - 6}" text-anchor="middle">{escape(x_label)}</text>')
    parts.append(f'<text class="label" transform="translate(14 {(PAD_T + H - PAD_B) / 2:.1f}) rotate(-90)" '
                 f'text-anchor="middle">{escape(y_label)}</text>')
    return sx, sy, parts


def _svg(parts, title):
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(title)}">' + "".join(parts) + "</svg>")


def _legend(labels):
    if not labels:
        return ""
    items = "".join(f'<span class="key"><i class="swatch s{i % 8}"></i>{escape(label)}</span>' for i, label in enumerate(labels))
    return f'<div class="legend">{items}</div>'


def line_chart(series, x_label, y_label, title, diagonal=False):
    """``series`` = [{"label", "xs", "ys"}] on the 0..1 square (ROC, precision-recall)."""
    sx, sy, parts = _frame(0.0, 1.0, 0.0, 1.0, x_label, y_label)
    if diagonal:
        parts.append(f'<line class="ref" x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1):.1f}" y2="{sy(1):.1f}"/>')
    for i, s in enumerate(series):
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(s["xs"], s["ys"]))
        parts.append(f'<polyline class="line s{i % 8}" points="{points}"/>')
    return _svg(parts, title) + _legend([s["label"] for s in series])


def scatter(actual, predicted, title):
    """Predicted against actual values with the y = x line (regression)."""
    values = list(actual) + list(predicted)
    lo, hi = min(values), max(values)
    margin = (hi - lo) * 0.05 or 1.0
    lo, hi = lo - margin, hi + margin
    sx, sy, parts = _frame(lo, hi, lo, hi, "Actual", "Predicted")
    parts.append(f'<line class="ref" x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}"/>')
    parts.extend(f'<circle class="dot" cx="{sx(a):.1f}" cy="{sy(p):.1f}" r="2.4"/>' for a, p in zip(actual, predicted))
    return _svg(parts, title)


def histogram(edges, counts, title):
    """Bars between ``edges`` (prediction errors, regression)."""
    top = max(counts) if counts else 1
    sx, sy, parts = _frame(edges[0], edges[-1], 0, top, "Error (actual − predicted)", "Rows")
    for left, right, count in zip(edges[:-1], edges[1:], counts):
        x, w = sx(left), max(sx(right) - sx(left) - 1, 1)
        parts.append(f'<rect class="bar s0" x="{x:.1f}" y="{sy(count):.1f}" width="{w:.1f}" height="{sy(0) - sy(count):.1f}"/>')
    return _svg(parts, title)


def bars(items, title, value_format=lambda v: _fmt(v, 3)):
    """Horizontal bars for ``[(label, value)]`` with values ≥ 0 (feature importance)."""
    if not items:
        return ""
    row, label_w = 24, 170
    height = PAD_T + row * len(items) + 8
    top = max(v for _, v in items) or 1.0
    scale = _scale(0, top, label_w, W - 70)
    parts = []
    for i, (label, value) in enumerate(items):
        y = PAD_T + i * row
        short = label if len(label) <= 26 else label[:25] + "…"
        parts.append(f'<text class="tick" x="{label_w - 6}" y="{y + 15}" text-anchor="end">{escape(short)}</text>')
        parts.append(f'<rect class="bar s0" x="{label_w}" y="{y + 4}" width="{max(scale(value) - label_w, 1):.1f}" height="{row - 8}"/>')
        parts.append(f'<text class="tick" x="{scale(value) + 4:.1f}" y="{y + 15}">{escape(value_format(value))}</text>')
    return (f'<svg class="chart" viewBox="0 0 {W} {height}" role="img" aria-label="{escape(title)}">' + "".join(parts) + "</svg>")

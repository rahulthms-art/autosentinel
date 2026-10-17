"""UI helpers implementing design-system/autosentinel/MASTER.md (UI UX Pro Max: Data-Dense + Trust)."""
from __future__ import annotations

from datetime import date, timedelta

import altair as alt
import pandas as pd
import streamlit as st

# Design tokens
PRIMARY = "#1E3A8A"
ACCENT = "#B45309"
DANGER = "#DC2626"
SUCCESS = "#15803D"
MUTED_FG = "#475569"
BORDER = "#CBD5E1"
ALERT_BAND = "#FFF3CD"
SERIES = ["#1E3A8A", "#2563EB", "#0E7490", "#475569", "#7C8DB5", "#0F766E"]
DASHES = [[1, 0], [6, 3], [2, 2], [8, 3, 2, 3], [4, 4], [1, 3]]

# Severity is always text + shape + color (never color alone)
SEVERITY = {
    "High": {"label": "▲ High", "color": DANGER, "badge": "red", "icon": ":material/error:"},
    "Medium": {"label": "■ Medium", "color": ACCENT, "badge": "orange", "icon": ":material/warning:"},
    "Low": {"label": "▽ Low", "color": MUTED_FG, "badge": "gray", "icon": ":material/info:"},
}
STATUS_ICON = {"new": ":material/fiber_new:", "acknowledged": ":material/visibility:",
               "actioned": ":material/task_alt:", "dismissed": ":material/block:"}

CSS = """
<style>
  /* tabular numbers for data; mono for identifiers */
  [data-testid="stMetricValue"], [data-testid="stDataFrame"] { font-variant-numeric: tabular-nums; }
  [data-testid="stMetricValue"] { font-size: 1.75rem; font-weight: 600; color: #0F172A; }
  [data-testid="stMetricLabel"] p { color: #475569; font-weight: 500; }
  h1 { font-weight: 600 !important; letter-spacing: -0.01em; }
  h2, h3 { font-weight: 600 !important; }
  .as-mono { font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.9em; }
  .as-eyebrow { color: #475569; font-size: 0.8rem; font-weight: 600; letter-spacing: .06em;
                text-transform: uppercase; margin-bottom: -0.4rem; }
  /* visible keyboard focus (WCAG) */
  button:focus-visible, a:focus-visible, [role="tab"]:focus-visible, input:focus-visible {
    outline: 2px solid #1E3A8A !important; outline-offset: 2px; }
  button, [role="tab"], a { cursor: pointer; }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def eyebrow(text: str):
    st.markdown(f'<div class="as-eyebrow">{text}</div>', unsafe_allow_html=True)


def severity_badge(sev: str | None):
    s = SEVERITY.get(sev or "", SEVERITY["Low"])
    st.badge(sev or "Unrated", icon=s["icon"], color=s["badge"])


def style_findings(df: pd.DataFrame):
    """Pandas Styler: colored, shape-prefixed severity labels."""
    def color(v):
        for s in SEVERITY.values():
            if v == s["label"]:
                return f"color: {s['color']}; font-weight: 600"
        return ""
    return df.style.map(color, subset=["Severity"])


def anomaly_chart(series: dict, spikes: list[dict], months: int = 24, top: int = 5):
    """Line chart with highlights (UI UX Pro Max chart rule #10: Anomaly Detection)."""
    df = pd.DataFrame(series).fillna(0).sort_index()
    if df.empty:
        return None
    cutoff = (pd.Timestamp.today() - pd.DateOffset(months=months)).strftime("%Y-%m")
    df = df[df.index >= cutoff]
    spike_comps = [s["component"] for s in spikes if s["component"] in df.columns]
    rest = [c for c in df.sum().sort_values(ascending=False).index if c not in spike_comps]
    cols = (spike_comps + rest)[:top]
    long = (df[cols].reset_index().rename(columns={"index": "month"})
            .melt(id_vars="month", var_name="Component", value_name="Complaints"))
    long["month"] = pd.to_datetime(long["month"] + "-01")

    recent_start = pd.Timestamp(date.today() - timedelta(days=90))

    def iso(df):  # dates go to the browser as ISO strings: Streamlit's Arrow transport can mangle timestamps
        return df.assign(**{c: df[c].dt.strftime("%Y-%m-%d") for c in df.columns
                            if pd.api.types.is_datetime64_any_dtype(df[c])})

    band = alt.Chart(pd.DataFrame({"start": [recent_start.strftime("%Y-%m-%d")],
                                   "end": [date.today().isoformat()]})).mark_rect(
        color=ALERT_BAND, opacity=0.9).encode(x="start:T", x2="end:T")
    band_label = alt.Chart(pd.DataFrame({"x": [recent_start.strftime("%Y-%m-%d")], "t": ["Last 90 days"]})).mark_text(
        align="left", baseline="top", dx=4, dy=4, color=ACCENT, fontWeight=600).encode(
        x="x:T", y=alt.value(0), text="t:N")

    domain = cols
    lines = alt.Chart(iso(long)).mark_line(strokeWidth=2).encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y", grid=False)),
        y=alt.Y("Complaints:Q", title="Complaints / month", axis=alt.Axis(gridColor="#E2E8F0")),
        color=alt.Color("Component:N", scale=alt.Scale(domain=domain, range=SERIES[:len(domain)]),
                        legend=alt.Legend(orient="bottom", title=None, labelLimit=260)),
        strokeDash=alt.StrokeDash("Component:N", scale=alt.Scale(domain=domain, range=DASHES[:len(domain)]),
                                  legend=None),
        tooltip=["Component:N", alt.Tooltip("month:T", format="%B %Y"), "Complaints:Q"])

    layers = [band, band_label, lines]
    unrecalled = {s["component"]: s for s in spikes if not s["recalled"]}
    hot = long[(long["Component"].isin(unrecalled)) & (long["month"] >= recent_start - pd.Timedelta(days=31))]
    if not hot.empty:
        last = hot.sort_values("month").groupby("Component").tail(1).copy()
        last["label"] = last["Component"].map(lambda c: f"×{unrecalled[c]['ratio']} · no recall")
        layers.append(alt.Chart(iso(hot)).mark_point(shape="circle", size=90, filled=True, color=DANGER).encode(
            x="month:T", y="Complaints:Q", tooltip=["Component:N", "Complaints:Q"]))
        layers.append(alt.Chart(iso(last)).mark_text(align="right", dx=-8, dy=-12, color=DANGER, fontWeight=600).encode(
            x="month:T", y="Complaints:Q", text="label:N"))
    return alt.layer(*layers).properties(height=340).configure_view(strokeWidth=0).configure_axis(
        labelColor=MUTED_FG, titleColor=MUTED_FG, labelFont="IBM Plex Sans", titleFont="IBM Plex Sans"
    ).configure_legend(labelFont="IBM Plex Sans", labelColor="#0F172A")

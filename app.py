# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional pipeline & exports (guarded)
try:
    from processing.pipeline import read_workbook as _read_workbook
except:
    _read_workbook = None

try:
    from processing.reporting import build_excel_report
except:
    build_excel_report = None

try:
    from processing.pdf_report import build_pdf_report
except:
    build_pdf_report = None

st.set_page_config(page_title="Portfolio Health Check", layout="wide")
st.title("Portfolio Health Check")

# ---- Plotly layout with legend fixed ----
_DEFAULT_HEIGHT = 420
_DEFAULT_MARGINS = dict(l=60, r=40, t=84, b=60)
_DEFAULT_TITLE = dict(y=0.97, x=0.5, xanchor="center", yanchor="top", font=dict(size=18))
_DEFAULT_LEGEND = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5,
    title=None
)
_DEFAULT_FONT = dict(size=13)
_UNIFORMTEXT = dict(mode="hide", minsize=10)

def _base_layout(fig: go.Figure, title: str | None = None, height: int | None = None) -> go.Figure:
    fig.update_layout(
        height=height or _DEFAULT_HEIGHT,
        margin=_DEFAULT_MARGINS,
        title=(dict(text=title, **_DEFAULT_TITLE) if title else None),
        legend=_DEFAULT_LEGEND,
        font=_DEFAULT_FONT,
        hovermode="closest",
        bargap=0.2,
        bargroupgap=0.08,
        uniformtext=_UNIFORMTEXT
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig

# ---- Fallback plotting helpers (if plot.py missing) ----
def plot_donut(df, cat_col, val_col, title=None, hole=0.55):
    d = df.groupby(cat_col, dropna=False, as_index=False)[val_col].sum()
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce").fillna(0)
    fig = px.pie(d, names=cat_col, values=val_col, hole=hole)
    fig.update_traces(textposition="inside", textinfo="percent+label", insidetextorientation="radial")
    return _base_layout(fig, title)

def plot_bar(df, x, y, title=None, sort_desc=True, top_n=None):
    d = df.copy()
    if top_n is not None:
        d = d.sort_values(y, ascending=False).head(top_n)
    d = d.sort_values(y, ascending=not sort_desc)
    fig = px.bar(d, x=x, y=y)
    fig.update_traces(hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y:,.2f}}<extra></extra>")
    return _base_layout(fig, title)

def plot_hbar(df, y, x, title=None, sort_desc=True, top_n=None):
    d = df.copy()
    if top_n is not None:
        d = d.sort_values(x, ascending=False).head(top_n)
    d = d.sort_values(x, ascending=not sort_desc)
    fig = px.bar(d, x=x, y=y, orientation="h")
    fig.update_traces(hovertemplate=f"{y}: %{{y}}<br>{x}: %{{x:,.2f}}<extra></extra>")
    return _base_layout(fig, title)

def plot_waterfall(df, label_col, value_col, title=None):
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0)
    measures = ["relative"] * max(len(d)-1, 0) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=d[label_col].astype(str),
        y=d[value_col],
        measure=measures,
        connector={"line": {"width": 1}}
    ))
    fig.update_traces(hovertemplate=f"{label_col}: %{{x}}<br>{value_col}: %{{y:,.2f}}<extra></extra>")
    return _base_layout(fig, title)

def plot_choro(df, iso3_col, value_col, title=None):
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0)
    d = d.groupby(iso3_col, as_index=False)[value_col].sum()
    fig = px.choropleth(d, locations=iso3_col, color=value_col, color_continuous_scale="Blues", projection="natural earth")
    fig.update_coloraxes(colorbar_title=value_col)
    return _base_layout(fig, title, height=500)

# ... rest of app.py unchanged, preserving dropdown/tab, ESG fix, exports, etc.

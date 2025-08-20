# plot.py
from typing import Optional, Tuple
import pandas as pd
import plotly.express as px

_LABEL_TEXTINFO = {
    "value": "label+value",
    "%": "label+percent",
    "both": "label+value+percent",
}

def _add_pct_of_total(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy()
    total = pd.to_numeric(out[value_col], errors="coerce").fillna(0.0).sum()
    out["% of Total"] = 0.0 if total == 0 else (out[value_col] / total * 100.0)
    return out

def pie_donut(
    df: pd.DataFrame,
    names_col: str,
    values_col: str,
    *,
    label_mode: str = "both",   # "value" | "%" | "both"
    title: Optional[str] = None,
    hole: float = 0.25,
    sort_desc: bool = True,
) -> Tuple["plotly.graph_objs._figure.Figure", pd.DataFrame]:
    work = df[[names_col, values_col]].copy()
    work[values_col] = pd.to_numeric(work[values_col], errors="coerce").fillna(0.0)
    if sort_desc:
        work = work.sort_values(values_col, ascending=False)

    fig = px.pie(
        work,
        names=names_col,
        values=values_col,
        hole=hole,
        title=title,
    )
    fig.update_traces(
        textinfo=_LABEL_TEXTINFO.get(label_mode, "label+value+percent"),
        textposition="inside",
    )
    fig.update_layout(margin=dict(l=8, r=8, t=40 if title else 8, b=8))
    table = _add_pct_of_total(work, values_col)
    return fig, table

# ---- Convenience wrappers used by app.py ----
def pie_asset_class(results: dict, label_mode: str = "both"):
    df = results["by_asset_class"].rename(columns={"USD Total": "Value"})
    return pie_donut(df, "Asset Class", "Value", label_mode=label_mode, title="By Asset Class")

def pie_sub_asset(results: dict, label_mode: str = "both"):
    df = results["by_sub_asset"].rename(columns={"USD Total": "Value"})
    return pie_donut(df, "Sub Asset Class", "Value", label_mode=label_mode, title="By Sub-Asset")

def pie_equity_sector(results: dict, label_mode: str = "both"):
    df = results["by_sector"].rename(columns={"Weight %": "Weight"})
    return pie_donut(df, "Sector (GICS)", "Weight", label_mode=label_mode, title="Equity by Sector")

def pie_equity_region(results: dict, label_mode: str = "both"):
    df = results["by_region"].rename(columns={"Weight %": "Weight"})
    return pie_donut(df, "Region", "Weight", label_mode=label_mode, title="Equity by Region")

def pie_fi_rating(results: dict, label_mode: str = "both"):
    df = results["by_rating"].rename(columns={"Weight %": "Weight"})
    return pie_donut(df, "Rating", "Weight", label_mode=label_mode, title="Fixed Income by Rating")

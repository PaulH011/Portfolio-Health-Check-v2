# app.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------- optional: use your pipeline if available ----------------
try:
    from processing.pipeline import read_workbook as _read_workbook
except Exception:
    _read_workbook = None

# ---------------- optional exports (guarded) ----------------
try:
    from processing.reporting import build_excel_report  # must return bytes
except Exception:
    build_excel_report = None

try:
    from processing.pdf_report import build_pdf_report  # must return bytes
except Exception:
    build_pdf_report = None

st.set_page_config(page_title="Portfolio Health Check", layout="wide")
st.title("Portfolio Health Check")

# ---------------- plotting helpers (local, no dependency on plot.py) ----------------
_DEFAULT_HEIGHT = 420
_DEFAULT_MARGINS = dict(l=60, r=40, t=84, b=60)
_DEFAULT_TITLE = dict(y=0.97, x=0.5, xanchor="center", yanchor="top", font=dict(size=18))
_DEFAULT_LEGEND = dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center")
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
        uniformtext=_UNIFORMTEXT,
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig

def plot_donut(df: pd.DataFrame, cat_col: str, val_col: str, title: str | None = None, hole: float = 0.55) -> go.Figure:
    d = df.groupby(cat_col, dropna=False, as_index=False)[val_col].sum()
    d[val_col] = pd.to_numeric(d[val_col], errors="coerce").fillna(0)
    fig = px.pie(d, names=cat_col, values=val_col, hole=hole)
    fig.update_traces(textposition="inside", textinfo="percent+label", insidetextorientation="radial")
    return _base_layout(fig, title)

def plot_bar(df: pd.DataFrame, x: str, y: str, title: str | None = None, sort_desc: bool = True, top_n: int | None = None) -> go.Figure:
    d = df.copy()
    if top_n is not None:
        d = d.sort_values(y, ascending=False).head(top_n)
    d = d.sort_values(y, ascending=not sort_desc)
    fig = px.bar(d, x=x, y=y)
    fig.update_traces(hovertemplate=f"{x}: %{{x}}<br>{y}: %{{y:,.2f}}<extra></extra>")
    return _base_layout(fig, title)

def plot_hbar(df: pd.DataFrame, y: str, x: str, title: str | None = None, sort_desc: bool = True, top_n: int | None = None) -> go.Figure:
    d = df.copy()
    if top_n is not None:
        d = d.sort_values(x, ascending=False).head(top_n)
    d = d.sort_values(x, ascending=not sort_desc)
    fig = px.bar(d, y=y, x=x, orientation="h")
    fig.update_traces(hovertemplate=f"{y}: %{{y}}<br>{x}: %{{x:,.2f}}<extra></extra>")
    return _base_layout(fig, title)

def plot_waterfall(df: pd.DataFrame, label_col: str, value_col: str, title: str | None = None) -> go.Figure:
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0)
    measures = ["relative"] * max(len(d) - 1, 0) + ["total"]
    fig = go.Figure(go.Waterfall(
        x=d[label_col].astype(str),
        y=d[value_col],
        measure=measures,
        connector={"line": {"width": 1}},
    ))
    fig.update_traces(hovertemplate=f"{label_col}: %{{x}}<br>{value_col}: %{{y:,.2f}}<extra></extra>")
    return _base_layout(fig, title)

def plot_choro(df: pd.DataFrame, iso3_col: str, value_col: str, title: str | None = None) -> go.Figure:
    d = df.copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0)
    d = d.groupby(iso3_col, as_index=False)[value_col].sum()
    fig = px.choropleth(d, locations=iso3_col, color=value_col, color_continuous_scale="Blues", projection="natural earth")
    fig.update_coloraxes(colorbar_title=value_col)
    return _base_layout(fig, title, height=500)

# ---------------- utils ----------------
REQ_PM_COLS = ["Asset Class", "Sub Asset Class", "FX", "USD Total"]

def _avg(series):
    s = pd.to_numeric(series, errors="coerce").replace(0, np.nan)
    return float(s.mean()) if s.notna().any() else np.nan

def _ci_get(dct: dict, name: str):
    """Case-insensitive dictionary get for sheet names."""
    for k, v in dct.items():
        if k.lower() == name.lower():
            return v
    return pd.DataFrame()

def _load_sheets(uploaded_file) -> dict:
    """
    Robust reader:
    - try processing.pipeline.read_workbook if available
    - accept dict OR tuple/list containing a dict
    - else fall back to pandas ExcelFile(all sheets)
    Returns: dict[str, DataFrame]
    """
    sheets = None
    if _read_workbook is not None:
        try:
            out = _read_workbook(uploaded_file)
        except Exception:
            out = None
        # normalize to dict
        if isinstance(out, dict):
            sheets = out
        elif isinstance(out, (list, tuple)):
            for item in out:
                if isinstance(item, dict) and any(isinstance(v, pd.DataFrame) for v in item.values()):
                    sheets = item
                    break
        # some pipelines return an object with .sheets or similar
        if sheets is None and hasattr(out, "sheets") and isinstance(out.sheets, dict):
            sheets = out.sheets

    if sheets is None:
        # fallback: read all sheets directly
        xls = pd.ExcelFile(uploaded_file)
        sheets = {name: xls.parse(name) for name in xls.sheet_names}

    # ensure DataFrames
    for k, v in list(sheets.items()):
        if not isinstance(v, pd.DataFrame):
            try:
                sheets[k] = pd.DataFrame(v)
            except Exception:
                sheets[k] = pd.DataFrame()
    return sheets

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("#### Upload Portfolio Workbook")
    uploaded_file = st.file_uploader(
        "PortfolioMaster v2 (Meta, PortfolioMaster, Policy, PolicyMeta); optional Equity/FixedIncome sheets",
        type=["xlsx", "xlsm"],
        accept_multiple_files=False,
    )

if not uploaded_file:
    st.info("Upload an Excel file to begin.")
    st.stop()

# ---------------- read workbook (robust) ----------------
sheets = _load_sheets(uploaded_file)

pm_df  = _ci_get(sheets, "PortfolioMaster")
eq_df  = _ci_get(sheets, "EquityAssetList")
fi_df  = _ci_get(sheets, "FixedIncomeAssetList")
policy = _ci_get(sheets, "Policy")

# ---- normalize PolicyMeta -> expected keys ----
policy_meta_df = _ci_get(sheets, "PolicyMeta")
if not policy_meta_df.empty:
    policy_meta_df = policy_meta_df.rename(columns={
        "ESG_Benchmark_Score": "Benchmark ESG",
        "Carbon_Benchmark_Intensity": "Benchmark Carbon",
    })
else:
    policy_meta_df = pd.DataFrame(columns=["Benchmark ESG", "Benchmark Carbon"])

# ---- validate required PM columns ----
missing = [c for c in REQ_PM_COLS if c not in pm_df.columns]
if missing:
    st.error(f"Missing required columns in PortfolioMaster: {missing}")
    st.stop()

# ---- derive weights if missing ----
if "Weight %" not in pm_df.columns:
    total = pd.to_numeric(pm_df["USD Total"], errors="coerce").fillna(0).sum()
    pm_df["Weight %"] = (pd.to_numeric(pm_df["USD Total"], errors="coerce").fillna(0) / total * 100.0) if total > 0 else 0.0

# ---- ESG/Carbon portfolio + benchmark ----
portfolio_esg    = _avg(pm_df.get("ESG Score"))
portfolio_carbon = _avg(pm_df.get("Carbon Intensity"))
benchmark_esg    = float(policy_meta_df["Benchmark ESG"].iloc[0]) if "Benchmark ESG" in policy_meta_df.columns and not policy_meta_df.empty else np.nan
benchmark_carbon = float(policy_meta_df["Benchmark Carbon"].iloc[0]) if "Benchmark Carbon" in policy_meta_df.columns and not policy_meta_df.empty else np.nan

# ---- build 'wide' to satisfy any code that expects exact keys ----
wide = pd.DataFrame([{
    "Portfolio ESG": portfolio_esg,
    "Benchmark ESG": benchmark_esg,
    "Portfolio Carbon": portfolio_carbon,
    "Benchmark Carbon": benchmark_carbon,
}])

# ---------------- tabs ----------------
tabs = st.tabs([
    "PM – Summary",
    "PM – Allocation",
    "PM – Policy vs Actual",
    "PM – Geography & FX",
    "PM – Liquidity & Fees/ESG",
    "Equity",
    "Fixed Income",
])

# PM – Summary
with tabs[0]:
    c1, c2, c3 = st.columns([2,2,1.5])
    with c1:
        st.subheader("Portfolio Snapshot")
        total_usd = pd.to_numeric(pm_df["USD Total"], errors="coerce").fillna(0).sum()
        ac_split = pm_df.groupby("Asset Class", as_index=False)["USD Total"].sum()
        st.metric("Total Portfolio (USD)", f"{total_usd:,.0f}")
        st.dataframe(ac_split.sort_values("USD Total", ascending=False), use_container_width=True, height=260)
    with c2:
        st.subheader("By Asset Class")
        fig = plot_donut(pm_df, cat_col="Asset Class", val_col="USD Total", title="Allocation by Asset Class")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c3:
        st.subheader("ESG / Carbon (Avg)")
        st.write(pd.DataFrame({
            "Metric": ["Portfolio ESG", "Benchmark ESG", "Portfolio Carbon", "Benchmark Carbon"],
            "Value":  [portfolio_esg,    benchmark_esg,    portfolio_carbon,    benchmark_carbon],
        }))

# PM – Allocation
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By Sub-Asset")
        fig = plot_donut(pm_df, cat_col="Sub Asset Class", val_col="USD Total", title="Allocation by Sub-Asset")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.subheader("Top Vehicles")
        if "Vehicle Type" in pm_df.columns:
            topv = pm_df.groupby("Vehicle Type", as_index=False)["USD Total"].sum().sort_values("USD Total", ascending=False)
            fig = plot_bar(topv, x="Vehicle Type", y="USD Total", title="Exposure by Vehicle")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Column 'Vehicle Type' not found.")

# PM – Policy vs Actual
with tabs[2]:
    st.subheader("Policy vs Actual (Asset Class)")
    if not policy.empty and {"Asset Class", "Policy Weight %"} <= set(policy.columns):
        actual = pm_df.groupby("Asset Class", as_index=False)["USD Total"].sum()
        actual["Actual %"] = actual["USD Total"] / actual["USD Total"].sum() * 100.0
        comp = policy.merge(actual[["Asset Class","Actual %"]], on="Asset Class", how="left").fillna(0)
        melt = comp.rename(columns={"Policy Weight %": "Policy %"}).melt("Asset Class", var_name="Type", value_name="Weight")
        fig = px.bar(melt, x="Asset Class", y="Weight", color="Type", barmode="group")
        st.plotly_chart(_base_layout(fig, "Policy vs Actual (Weight %)"), use_container_width=True, config={"displayModeBar": False})

        tol = st.slider("Breach tolerance (pp)", 0.0, 10.0, 2.0, 0.5)
        comp["Deviation (pp)"] = comp["Actual %"] - comp["Policy Weight %"]
        comp["Breach"] = comp["Deviation (pp)"].abs() > tol
        st.dataframe(comp[["Asset Class","Policy Weight %","Actual %","Deviation (pp)","Breach"]].sort_values("Deviation (pp)", ascending=False), use_container_width=True)
    else:
        st.info("No 'Policy' sheet with columns ['Asset Class','Policy Weight %'] found.")

# PM – Geography & FX
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Geographic Exposure")
        if "Country ISO3" in pm_df.columns:
            fig = plot_choro(pm_df, iso3_col="Country ISO3", value_col="USD Total", title="By Country (ISO3)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Column 'Country ISO3' not found.")
    with c2:
        st.subheader("Currency Exposure")
        fx = pm_df.groupby("FX", as_index=False)["USD Total"].sum().sort_values("USD Total", ascending=False)
        fig = plot_bar(fx, x="FX", y="USD Total", title="By Currency (Base)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# PM – Liquidity & Fees/ESG
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Liquidity Ladder")
        if "Liquidity" in pm_df.columns:
            liq = pm_df.groupby("Liquidity", as_index=False)["USD Total"].sum().sort_values("USD Total", ascending=False)
            fig = plot_waterfall(liq, label_col="Liquidity", value_col="USD Total", title="Liquidity Contribution")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Column 'Liquidity' not found.")
    with c2:
        st.subheader("Fees & ESG")
        cols_present = [c for c in ["TER %","ESG Score","Carbon Intensity"] if c in pm_df.columns]
        if cols_present:
            agg = {c: "mean" for c in cols_present}
            out = pm_df.groupby("Asset Class").agg(agg).reset_index()
            st.dataframe(out, use_container_width=True)
        else:
            st.info("Provide 'TER %', 'ESG Score', 'Carbon Intensity' to populate.")

    st.divider()
    # Small tiles using 'wide' (keys always exist now)
    esg_plot = pd.DataFrame({
        "Metric": ["ESG","ESG"],
        "Type": ["Portfolio","Benchmark"],
        "Value": [wide.iloc[0]["Portfolio ESG"], wide.iloc[0]["Benchmark ESG"]],
    })
    carb_plot = pd.DataFrame({
        "Metric": ["Carbon","Carbon"],
        "Type": ["Portfolio","Benchmark"],
        "Value": [wide.iloc[0]["Portfolio Carbon"], wide.iloc[0]["Benchmark Carbon"]],
    })
    c3, c4 = st.columns(2)
    with c3:
        st.write("**ESG Score (avg)**")
        st.dataframe(esg_plot, use_container_width=True, hide_index=True)
    with c4:
        st.write("**Carbon Intensity (avg)**")
        st.dataframe(carb_plot, use_container_width=True, hide_index=True)

# Equity
with tabs[5]:
    st.subheader("Equity (if provided)")
    if eq_df.empty:
        st.info("No 'EquityAssetList' sheet found.")
    else:
        if "Sector" in eq_df.columns and "USD Total" in eq_df.columns:
            fig = plot_donut(eq_df, cat_col="Sector", val_col="USD Total", title="Equities by Sector")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if "Region" in eq_df.columns and "USD Total" in eq_df.columns:
            fig = plot_donut(eq_df, cat_col="Region", val_col="USD Total", title="Equities by Region")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        name_col = "Name" if "Name" in eq_df.columns else ("Security" if "Security" in eq_df.columns else None)
        if name_col and "USD Total" in eq_df.columns:
            top_positions = (eq_df.groupby(name_col, as_index=False)["USD Total"].sum()
                                   .sort_values("USD Total", ascending=False).head(15))
            fig = plot_hbar(top_positions, y=name_col, x="USD Total", title="Top 15 Equity Positions")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(eq_df, use_container_width=True)

# Fixed Income
with tabs[6]:
    st.subheader("Fixed Income (if provided)")
    if fi_df.empty:
        st.info("No 'FixedIncomeAssetList' sheet found.")
    else:
        if "Rating" in fi_df.columns and "USD Total" in fi_df.columns:
            fig = plot_donut(fi_df, cat_col="Rating", val_col="USD Total", title="By Rating")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if "Maturity Bucket" in fi_df.columns and "USD Total" in fi_df.columns:
            mb = fi_df.groupby("Maturity Bucket", as_index=False)["USD Total"].sum()
            fig = plot_bar(mb, x="Maturity Bucket", y="USD Total", title="By Maturity Bucket")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if "Duration Bucket" in fi_df.columns and "USD Total" in fi_df.columns:
            db = fi_df.groupby("Duration Bucket", as_index=False)["USD Total"].sum()
            fig = plot_bar(db, x="Duration Bucket", y="USD Total", title="By Duration Bucket")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(fi_df, use_container_width=True)

st.divider()

# ---------------- exports ----------------
c1, c2 = st.columns(2)
with c1:
    if build_excel_report:
        if st.button("📊 Build Excel Report", use_container_width=True):
            try:
                xbytes = build_excel_report(dfs=sheets)  # expects dict of sheets
                st.download_button("Download Excel", data=xbytes,
                                   file_name="Portfolio_Health_Check.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            except Exception as e:
                st.error(f"Excel export failed: {e}")
    else:
        st.caption("Excel export not available.")

with c2:
    if build_pdf_report:
        if st.button("📄 Build PDF Report", use_container_width=True):
            try:
                pbytes = build_pdf_report(dfs=sheets)  # expects dict of sheets
                st.download_button("Download PDF", data=pbytes,
                                   file_name="Portfolio_Health_Check.pdf",
                                   mime="application/pdf",
                                   use_container_width=True)
            except Exception as e:
                st.error(f"PDF export failed: {e}")
    else:
        st.caption("PDF export not available.")

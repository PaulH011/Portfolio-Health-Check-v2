# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st

# --- Internal modules (present in your repo) ---
from processing.pipeline import read_workbook  # detect_template_type/validate_df/transform_results if you use them
from plot import donut, bar, hbar, waterfall, choropleth_iso3

# Optional exports (guarded; buttons hidden if missing)
try:
    from processing.reporting import build_excel_report  # expected: returns bytes
except Exception:
    build_excel_report = None

try:
    from processing.pdf_report import build_pdf_report  # expected: returns bytes
except Exception:
    build_pdf_report = None

st.set_page_config(page_title="Portfolio Health Check", layout="wide")

# ------------- Helpers -------------
REQ_PM_COLS = ["Asset Class", "Sub Asset Class", "FX", "USD Total"]

def _get_sheet(dfs: dict, name: str):
    if not dfs:
        return pd.DataFrame()
    # case-insensitive get
    for k in dfs.keys():
        if k.lower() == name.lower():
            return dfs[k]
    return pd.DataFrame()

def _avg(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace(0, np.nan)  # treat zeros as missing, avoids skew
    return float(s.mean()) if s.notna().any() else np.nan

def _pct(x):  # pretty %
    return np.round(x * 100.0, 2)

# ------------- UI -------------
st.title("Portfolio Health Check")

with st.sidebar:
    st.markdown("#### Upload Portfolio Workbook")
    uploaded_file = st.file_uploader(
        "PortfolioMaster v2 (with optional Policy/PolicyMeta) or combined workbook",
        type=["xlsx", "xlsm"],
        accept_multiple_files=False,
    )
    st.caption("Sheets: **Meta, PortfolioMaster, Policy, PolicyMeta** (optional: EquityAssetList, FixedIncomeAssetList)")

if not uploaded_file:
    st.info("Upload an Excel file to begin.")
    st.stop()

# Read workbook (dict of DataFrames keyed by sheet)
dfs = read_workbook(uploaded_file)

# --- PATCH A: Normalise PolicyMeta column names so downstream code can rely on literal keys
policy_meta_df = _get_sheet(dfs, "PolicyMeta")
if not policy_meta_df.empty:
    policy_meta_df = policy_meta_df.rename(
        columns={
            "ESG_Benchmark_Score": "Benchmark ESG",
            "Carbon_Benchmark_Intensity": "Benchmark Carbon",
        }
    )
else:
    policy_meta_df = pd.DataFrame(columns=["Benchmark ESG", "Benchmark Carbon"])

# Sheets (graceful if missing)
pm_df  = _get_sheet(dfs, "PortfolioMaster")
eq_df  = _get_sheet(dfs, "EquityAssetList")
fi_df  = _get_sheet(dfs, "FixedIncomeAssetList")
policy = _get_sheet(dfs, "Policy")

# Light validation for PM required columns
missing = [c for c in REQ_PM_COLS if c not in pm_df.columns]
if missing:
    st.error(f"Missing required columns in PortfolioMaster: {missing}")
    st.stop()

# Derive weights if not present
if "Weight %" not in pm_df.columns:
    total = pd.to_numeric(pm_df["USD Total"], errors="coerce").fillna(0).sum()
    if total > 0:
        pm_df["Weight %"] = pd.to_numeric(pm_df["USD Total"], errors="coerce").fillna(0) / total * 100.0
    else:
        pm_df["Weight %"] = 0.0

# ESG & Carbon portfolio averages
portfolio_esg    = _avg(pm_df.get("ESG Score"))
portfolio_carbon = _avg(pm_df.get("Carbon Intensity"))
benchmark_esg = float(policy_meta_df["Benchmark ESG"].iloc[0]) if "Benchmark ESG" in policy_meta_df.columns and not policy_meta_df.empty else np.nan
benchmark_carbon = float(policy_meta_df["Benchmark Carbon"].iloc[0]) if "Benchmark Carbon" in policy_meta_df.columns and not policy_meta_df.empty else np.nan

# --- PATCH B: Build a 'wide' frame with the exact keys the rest of the page uses
wide = pd.DataFrame([{
    "Portfolio ESG": portfolio_esg,
    "Benchmark ESG": benchmark_esg,
    "Portfolio Carbon": portfolio_carbon,
    "Benchmark Carbon": benchmark_carbon,
}])

# ===================== Tabs =====================
tabs = st.tabs([
    "PM – Summary",
    "PM – Allocation",
    "PM – Policy vs Actual",
    "PM – Geography & FX",
    "PM – Liquidity & Fees/ESG",
    "Equity",
    "Fixed Income",
])

# ---------- PM – Summary ----------
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
        fig = donut(pm_df, cat_col="Asset Class", val_col="USD Total", title="Allocation by Asset Class")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c3:
        st.subheader("ESG / Carbon (Avg)")
        st.write(pd.DataFrame({
            "Metric": ["Portfolio ESG", "Benchmark ESG", "Portfolio Carbon", "Benchmark Carbon"],
            "Value":  [portfolio_esg,    benchmark_esg,    portfolio_carbon,    benchmark_carbon],
        }))

# ---------- PM – Allocation ----------
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("By Sub-Asset")
        fig = donut(pm_df, cat_col="Sub Asset Class", val_col="USD Total", title="Allocation by Sub-Asset")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        st.subheader("Top Vehicles")
        if "Vehicle Type" in pm_df.columns:
            topv = pm_df.groupby("Vehicle Type", as_index=False)["USD Total"].sum().sort_values("USD Total", ascending=False)
            fig = bar(topv, x="Vehicle Type", y="USD Total", title="Exposure by Vehicle")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Column 'Vehicle Type' not found.")

# ---------- PM – Policy vs Actual ----------
with tabs[2]:
    st.subheader("Policy vs Actual (Asset Class)")
    if not policy.empty and {"Asset Class", "Policy Weight %"} <= set(policy.columns):
        actual = pm_df.groupby("Asset Class", as_index=False)["USD Total"].sum()
        actual["Actual %"] = actual["USD Total"] / actual["USD Total"].sum() * 100.0
        comp = policy.merge(actual[["Asset Class","Actual %"]], on="Asset Class", how="left").fillna(0)
        melt = comp.rename(columns={"Policy Weight %": "Policy %"}).melt("Asset Class", var_name="Type", value_name="Weight")
        import plotly.express as px
        fig = px.bar(melt, x="Asset Class", y="Weight", color="Type", barmode="group")
        # apply base layout from plot helpers
        from plot import _base_layout
        fig = _base_layout(fig, "Policy vs Actual (Weight %)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Breaches table (± tolerance)
        tol = st.slider("Breach tolerance (pp)", 0.0, 10.0, 2.0, 0.5)
        comp["Deviation (pp)"] = comp["Actual %"] - comp["Policy Weight %"]
        comp["Breach"] = comp["Deviation (pp)"].abs() > tol
        st.dataframe(comp[["Asset Class","Policy Weight %","Actual %","Deviation (pp)","Breach"]].sort_values("Deviation (pp)", ascending=False), use_container_width=True)
    else:
        st.info("No 'Policy' sheet with columns ['Asset Class','Policy Weight %'] found.")

# ---------- PM – Geography & FX ----------
with tabs[3]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Geographic Exposure")
        if "Country ISO3" in pm_df.columns:
            fig = choropleth_iso3(pm_df, iso3_col="Country ISO3", value_col="USD Total", title="By Country (ISO3)")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Column 'Country ISO3' not found.")
    with c2:
        st.subheader("Currency Exposure")
        fx = pm_df.groupby("FX", as_index=False)["USD Total"].sum().sort_values("USD Total", ascending=False)
        fig = bar(fx, x="FX", y="USD Total", title="By Currency (Base)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ---------- PM – Liquidity & Fees/ESG ----------
with tabs[4]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Liquidity Ladder")
        if "Liquidity" in pm_df.columns:
            liq = pm_df.groupby("Liquidity", as_index=False)["USD Total"].sum().sort_values("USD Total", ascending=False)
            fig = waterfall(liq, label_col="Liquidity", value_col="USD Total", title="Liquidity Contribution")
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
    # Small tiles constructed via earlier 'wide'
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

# ---------- Equity ----------
with tabs[5]:
    st.subheader("Equity (if provided)")
    if eq_df.empty:
        st.info("No 'EquityAssetList' sheet found.")
    else:
        if "Sector" in eq_df.columns and "USD Total" in eq_df.columns:
            fig = donut(eq_df, cat_col="Sector", val_col="USD Total", title="Equities by Sector")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if "Region" in eq_df.columns and "USD Total" in eq_df.columns:
            fig = donut(eq_df, cat_col="Region", val_col="USD Total", title="Equities by Region")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        # Top positions if available
        name_col = "Name" if "Name" in eq_df.columns else ("Security" if "Security" in eq_df.columns else None)
        if name_col and "USD Total" in eq_df.columns:
            top_positions = (eq_df.groupby(name_col, as_index=False)["USD Total"].sum()
                                   .sort_values("USD Total", ascending=False).head(15))
            fig = hbar(top_positions, y=name_col, x="USD Total", title="Top 15 Equity Positions")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(eq_df, use_container_width=True)

# ---------- Fixed Income ----------
with tabs[6]:
    st.subheader("Fixed Income (if provided)")
    if fi_df.empty:
        st.info("No 'FixedIncomeAssetList' sheet found.")
    else:
        # Rating / Maturity / Duration buckets if present
        if "Rating" in fi_df.columns and "USD Total" in fi_df.columns:
            fig = donut(fi_df, cat_col="Rating", val_col="USD Total", title="By Rating")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if "Maturity Bucket" in fi_df.columns and "USD Total" in fi_df.columns:
            mb = fi_df.groupby("Maturity Bucket", as_index=False)["USD Total"].sum()
            fig = bar(mb, x="Maturity Bucket", y="USD Total", title="By Maturity Bucket")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if "Duration Bucket" in fi_df.columns and "USD Total" in fi_df.columns:
            db = fi_df.groupby("Duration Bucket", as_index=False)["USD Total"].sum()
            fig = bar(db, x="Duration Bucket", y="USD Total", title="By Duration Bucket")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(fi_df, use_container_width=True)

st.divider()

# ===================== Exports =====================
c1, c2 = st.columns(2)
with c1:
    if build_excel_report:
        if st.button("📊 Download Excel Report", use_container_width=True):
            try:
                xbytes = build_excel_report(dfs=dfs)  # your function signature
                st.download_button("Save Excel", data=xbytes, file_name="Portfolio_Health_Check.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)
            except Exception as e:
                st.error(f"Excel export failed: {e}")
    else:
        st.caption("Excel export not available (processing/reporting.py function not found).")

with c2:
    if build_pdf_report:
        if st.button("📄 Download PDF Report", use_container_width=True):
            try:
                pbytes = build_pdf_report(dfs=dfs)  # your function signature
                st.download_button("Save PDF", data=pbytes, file_name="Portfolio_Health_Check.pdf",
                                   mime="application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"PDF export failed: {e}")
    else:
        st.caption("PDF export not available (processing/pdf_report.py function not found).")

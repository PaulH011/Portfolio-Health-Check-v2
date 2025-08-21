# app.py
import streamlit as st
st.set_page_config(page_title="Portfolio Health Check", layout="wide")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import plotly.io as pio
import plotly.graph_objects as go

pio.templates["parkview"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=13),
        colorway=["#1F3B63", "#7DA2C8", "#4CB944", "#F5A623", "#D64550", "#A06CD5"],
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(borderwidth=0, orientation="h", x=0, y=1.08),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)
pio.templates.default = "parkview"


from processing.pipeline import (
    read_workbook,
    detect_template_type,
    get_required_sheet_for_type,
    validate_df,
    transform_results,
)
from processing.reporting import build_report_xlsx, build_validation_report
from processing.pdf_report import build_pdf_report

from plot import (
    pie_asset_class, pie_sub_asset,
    pie_equity_sector, pie_equity_region, pie_fi_rating
)

# ---------------- Optional password gate ----------------
if "APP_PASSWORD" in st.secrets:
    pwd = st.sidebar.text_input("App password", type="password")
    if pwd != st.secrets["APP_PASSWORD"]:
        st.stop()

st.title("Portfolio Health Check")

# ---------------- Sidebar: upload ----------------
with st.sidebar:
    st.header("Upload")
    file = st.file_uploader(
        "Upload standardized Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False
    )

if not file:
    st.info("Upload a standardized template to begin.")
    st.stop()

# ---------------- Read workbook & detect type ----------------
try:
    sheets, meta_str = read_workbook(file)  # dict[str, DataFrame], meta string if any
except Exception as e:
    st.error(f"Template read failed: {e}")
    st.stop()

detected_type = detect_template_type(sheets, meta_str)
tmpl_options = ["PortfolioMaster", "EquityAssetList", "FixedIncomeAssetList"]
with st.sidebar:
    tmpl_type = st.selectbox(
        "Template type", options=tmpl_options, index=tmpl_options.index(detected_type)
    )

# --- Normalise PolicyMeta column names (maps template -> app expectations) ---
import pandas as pd
import numpy as np

policy_meta_df = dfs.get("PolicyMeta", pd.DataFrame())
if not policy_meta_df.empty:
    policy_meta_df = policy_meta_df.rename(columns={
        "ESG_Benchmark_Score": "Benchmark ESG",
        "Carbon_Benchmark_Intensity": "Benchmark Carbon",
    })
else:
    # keep an empty frame with expected columns to avoid KeyError
    policy_meta_df = pd.DataFrame(columns=["Benchmark ESG", "Benchmark Carbon"])

# ---------------- Safe sheet fetch ----------------
sheet_name = get_required_sheet_for_type(tmpl_type)
df = sheets.get(sheet_name)

# case-insensitive fallback + allow 'Pastor' alias
if df is None:
    for k, v in sheets.items():
        if k.lower() == sheet_name.lower():
            df = v
            break
        if tmpl_type == "PortfolioMaster" and k.lower() == "pastor":
            df = v
            break

if df is None:
    st.error(f"Missing required sheet: '{sheet_name}'.")
    st.stop()

if df.shape[0] == 0:
    st.warning(
        f"Sheet '{sheet_name}' is present but has 0 rows. "
        "Download a sample template, fill it, and re-upload."
    )
    st.stop()

# ---------------- Validation ----------------
errors = validate_df(df, tmpl_type)
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("Validation")
    if errors:
        st.error(f"{len(errors)} validation issue(s) found")
        st.dataframe(pd.DataFrame(errors))
    else:
        st.success("Validation passed")
with c2:
    vb = build_validation_report(errors)
    st.download_button("Download Validation Report", vb, file_name="validation_report.xlsx")

if errors:
    st.stop()

# ---------------- Transform ----------------
# pass sheets so PortfolioMaster can read Policy/PolicyMeta
results = transform_results(df, tmpl_type, sheets=sheets)
st.caption(
    f"Detected: **{detected_type}**  •  Processing as: **{tmpl_type}**  •  Source sheet: **{sheet_name}**"
)

# Collect figures for PDF
figs = {}

# ====================================================================
# PortfolioMaster (v2) – macro analytics
# ====================================================================
if tmpl_type == "PortfolioMaster":
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Summary", "By Asset Class", "By Sub-Asset",
        "Policy vs Actual", "Geography", "FX", "Liquidity", "Fees & ESG"
    ])

    # ---- Summary
    with tab1:
        m = results["metrics"]
        cA, cB, cC = st.columns(3)
        cA.metric("Rows", m.get("n_rows", 0))
        cB.metric("Total USD", f'{m.get("usd_total_sum", 0):,.2f}')
        cC.metric("Top-10 concentration", f'{m.get("top10_concentration_%", 0.0):.2f}%')
        st.markdown("**Top positions (by USD)**")
        st.dataframe(results.get("top_assets", pd.DataFrame()))

    # ---- Asset class donut
    with tab2:
        label_mode = st.radio(
            "Labels", ["Value ($)", "% of total", "Both"],
            index=2, horizontal=True, key="pm_assetclass_labels"
        )
        mode = {"Value ($)": "value", "% of total": "%", "Both": "both"}[label_mode]
        fig_ac, table_ac = pie_asset_class(results, label_mode=mode)
        st.plotly_chart(fig_ac, use_container_width=True)
        st.dataframe(table_ac)
        figs["pm_asset_pie"] = fig_ac

    # ---- Sub-asset donut
    with tab3:
        label_mode = st.radio(
            "Labels", ["Value ($)", "% of total", "Both"],
            index=2, horizontal=True, key="pm_subasset_labels"
        )
        mode = {"Value ($)": "value", "% of total": "%", "Both": "both"}[label_mode]
        fig_sa, table_sa = pie_sub_asset(results, label_mode=mode)
        st.plotly_chart(fig_sa, use_container_width=True)
        st.dataframe(table_sa)
        figs["pm_subasset_pie"] = fig_sa

    # ---- Policy vs Actual (if Policy sheet provided)
    with tab4:
        comp = results.get("policy_compare")
        if comp is None or comp.empty:
            st.info("No 'Policy' sheet found. Add a sheet with columns: Asset Class, Policy Weight %.")
        else:
            comp_melt = comp.melt(id_vars="Asset Class", var_name="Type", value_name="Weight %")
            fig_pol = px.bar(comp_melt, x="Asset Class", y="Weight %", color="Type",
                             barmode="group", text="Weight %", title="Policy vs Actual Weights")
            fig_pol.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_pol, use_container_width=True)
            st.dataframe(comp)
            figs["pm_policy_bar"] = fig_pol

    # ---- Geography (country choropleth)
    with tab5:
        geo = results.get("by_country", pd.DataFrame())
        if geo.empty:
            st.info("No country exposure columns found (Country ISO3 or Country).")
        else:
            if "Country ISO3" in geo.columns:
                fig_geo = px.choropleth(
                    geo, locations="Country ISO3", color="USD Total",
                    color_continuous_scale="Blues", title="Exposure by Country (USD)"
                )
            else:
                geo = geo.rename(columns={"Country": "Country Name"})
                fig_geo = px.choropleth(
                    geo, locations="Country Name", locationmode="country names",
                    color="USD Total", color_continuous_scale="Blues",
                    title="Exposure by Country (USD)"
                )
            st.plotly_chart(fig_geo, use_container_width=True)
            st.dataframe(geo)
            figs["pm_geo_map"] = fig_geo

    # ---- FX exposure
    with tab6:
        fx = results.get("by_fx", pd.DataFrame())
        fig_fx = px.bar(fx, x="FX", y="USD Total", text="USD Total", title="FX Exposure (USD)")
        st.plotly_chart(fig_fx, use_container_width=True)
        st.dataframe(fx)
        figs["pm_currency_bar"] = fig_fx

    # ---- Liquidity waterfall (by redemption frequency)
    with tab7:
        liq = results.get("by_liquidity", pd.DataFrame()).copy()
        if not liq.empty:
            order = ["Daily", "Weekly", "Monthly", "Quarterly", "Annual", "Locked-up"]
            liq["Liquidity"] = pd.Categorical(liq["Liquidity"], categories=order, ordered=True)
            liq = liq.sort_values("Liquidity").dropna()
            measures = ["relative"] * len(liq)
            fig_lq = go.Figure(go.Waterfall(
                x=liq["Liquidity"], y=liq["USD Total"], measure=measures,
                text=[f"{v:,.0f}" for v in liq["USD Total"]],
                textposition="outside", connector={"line":{"color":"#1F3B63"}}
            ))
            fig_lq.update_layout(title="Liquidity Waterfall (USD)")
            st.plotly_chart(fig_lq, use_container_width=True)
            st.dataframe(liq)
            figs["pm_liquidity_bar"] = fig_lq
        else:
            st.info("No liquidity data found.")

    # ---- Fees & ESG
    with tab8:
        colA, colB = st.columns(2)

        with colA:
            fr = results.get("fees_returns", pd.DataFrame())
            if not fr.empty:
                fig_fr = px.bar(fr, x="Metric", y="Value", text="Value",
                                title="Expected Returns & Fee Drag (MV-weighted %)")
                fig_fr.update_layout(xaxis_tickangle=-25)
                st.plotly_chart(fig_fr, use_container_width=True)
                st.dataframe(fr)
                figs["pm_fees_returns"] = fig_fr
            else:
                st.info("No fee/expected return data present in file.")

        with colB:
            esg = results.get("esg", pd.DataFrame())
            if not esg.empty:
                wide = esg.pivot_table(index=None, columns="Metric", values="Value", aggfunc="first")
                st.dataframe(wide)

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
                fig_esg = px.bar(esg_plot, x="Metric", y="Value", color="Type",
                                 barmode="group", title="ESG score")
                fig_carb = px.bar(carb_plot, x="Metric", y="Value", color="Type",
                                  barmode="group", title="Carbon intensity")
                st.plotly_chart(fig_esg, use_container_width=True)
                st.plotly_chart(fig_carb, use_container_width=True)
                figs["pm_esg"] = fig_esg
                figs["pm_carbon"] = fig_carb
            else:
                st.info("No ESG/Carbon columns found (ESG Score / Carbon Intensity).")

# ====================================================================
# Equity asset list
# ====================================================================
elif tmpl_type == "EquityAssetList":
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "By Sector", "By Region", "Top Positions"])

    with tab1:
        m = results["metrics"]
        cA, cB, cC = st.columns(3)
        cA.metric("Positions", m.get("n_rows", 0))
        cB.metric("Market Value (USD)", f'{m.get("mv_sum", 0):,.2f}')
        cC.metric("Weight % (sum)", f'{m.get("w_sum", 0.0):.2f}')

    with tab2:
        label_mode = st.radio(
            "Labels", ["Value ($)", "% of total", "Both"],
            index=2, horizontal=True, key="eq_sector_labels"
        )
        mode = {"Value ($)": "value", "% of total": "%", "Both": "both"}[label_mode]
        fig, table = pie_equity_sector(results, label_mode=mode)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(table.rename(columns={"Weight": "Weight %"}))
        figs["eq_sector_pie"] = fig

    with tab3:
        label_mode = st.radio(
            "Labels", ["Value ($)", "% of total", "Both"],
            index=2, horizontal=True, key="eq_region_labels"
        )
        mode = {"Value ($)": "value", "% of total": "%", "Both": "both"}[label_mode]
        fig, table = pie_equity_region(results, label_mode=mode)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(table.rename(columns={"Weight": "Weight %"}))
        figs["eq_region_pie"] = fig

    with tab4:
        top = results.get("top_positions", pd.DataFrame()).copy()
        if not top.empty:
            fig_top = px.bar(top.head(15), x=top.columns[0], y="Market Value (USD)", text="Market Value (USD)")
            fig_top.update_layout(xaxis_tickangle=-35, title="Top Positions (by MV)")
            st.plotly_chart(fig_top, use_container_width=True)
            st.dataframe(top)
            figs["eq_top_bar"] = fig_top
        else:
            st.info("No positions available.")

# ====================================================================
# Fixed Income asset list
# ====================================================================
elif tmpl_type == "FixedIncomeAssetList":
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "By Rating", "Maturity Ladder", "Duration Buckets"])

    with tab1:
        m = results["metrics"]
        cA, cB, cC = st.columns(3)
        cA.metric("Bonds", m.get("n_rows", 0))
        cB.metric("Market Value (USD)", f'{m.get("mv_sum", 0):,.2f}')
        cC.metric("Weight % (sum)", f'{m.get("w_sum", 0.0):.2f}')

    with tab2:
        label_mode = st.radio(
            "Labels", ["Value ($)", "% of total", "Both"],
            index=2, horizontal=True, key="fi_rating_labels"
        )
        mode = {"Value ($)": "value", "% of total": "%", "Both": "both"}[label_mode]
        fig, table = pie_fi_rating(results, label_mode=mode)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(table.rename(columns={"Weight": "Weight %"}))
        figs["fi_rating_pie"] = fig

    with tab3:
        mat = results.get("maturity_buckets", pd.DataFrame())
        if not mat.empty:
            fig_mat = px.bar(mat, x="Bucket", y="Weight %", text="Weight %", title="Maturity Ladder")
            st.plotly_chart(fig_mat, use_container_width=True)
            st.dataframe(mat)
            figs["fi_maturity_bar"] = fig_mat
        else:
            st.info("No maturity data available.")

    with tab4:
        dur = results.get("duration_buckets", pd.DataFrame())
        if not dur.empty:
            fig_dur = px.bar(dur, x="Bucket", y="Weight %", text="Weight %", title="Duration Buckets")
            st.plotly_chart(fig_dur, use_container_width=True)
            st.dataframe(dur)
            figs["fi_duration_bar"] = fig_dur
        else:
            st.info("No duration data available.")

# ---------------- Exports ----------------
xlsx_bytes = build_report_xlsx(results)
st.download_button("Download report.xlsx", xlsx_bytes, file_name="portfolio_health_report.xlsx")

pdf_bytes = build_pdf_report(
    tmpl_type, results, figs, logo_path="assets/parkview_logo.png"
)
st.download_button(
    "Download PDF report", pdf_bytes,
    file_name="portfolio_health_report.pdf", mime="application/pdf"
)



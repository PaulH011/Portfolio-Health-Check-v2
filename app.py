import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# --- Internal modules ---
from processing.pipeline import read_workbook, detect_template_type, validate_df, transform_results
from processing.reporting import generate_excel_report
from processing.pdf_report import generate_pdf_report
from plot import donut, bar, hbar, waterfall, choropleth_iso3

# Optional exports (guarded; buttons hidden if missing)
try:
    from processing.reporting import generate_excel_report
    from processing.pdf_report import generate_pdf_report
except ImportError:
    generate_excel_report = None
    generate_pdf_report = None

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Portfolio Health Check", layout="wide")
st.title("Portfolio Health Check")

# ================== FILE UPLOAD ==================
uploaded_file = st.file_uploader("Upload Portfolio Workbook", type=["xlsx"])

if uploaded_file:
    # --- FIX: unpack tuple safely if read_workbook returns more than one object ---
    result = read_workbook(uploaded_file)
    if isinstance(result, tuple):
        dfs = result[0]
    else:
        dfs = result

    # Validate structure
    if not isinstance(dfs, dict):
        st.error("Uploaded file could not be read into expected format.")
    else:
        # Detect template type
        template_type = detect_template_type(dfs)

        # Dropdown for override (auto-selects detected type)
        tab_choice = st.selectbox(
            "Select View",
            options=["Portfolio Master", "Equity Asset List", "Fixed Income Asset List"],
            index=["Portfolio Master", "Equity Asset List", "Fixed Income Asset List"].index(template_type)
            if template_type in ["Portfolio Master", "Equity Asset List", "Fixed Income Asset List"]
            else 0,
        )

        # ================== PORTFOLIO MASTER ==================
        if tab_choice == "Portfolio Master" and "PortfolioMaster" in dfs:
            pm_df = dfs["PortfolioMaster"]

            st.subheader("PM – Summary")
            st.dataframe(pm_df.head())

            # Example: Asset Class donut
            fig = donut(pm_df, cat_col="Asset Class", val_col="USD Total")
            # --- FIX: prevent legend overlap ---
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

        # ================== EQUITY ==================
        elif tab_choice == "Equity Asset List" and "EquityAssetList" in dfs:
            eq_df = dfs["EquityAssetList"]

            st.subheader("Equity – Summary")
            st.dataframe(eq_df.head())

            fig = donut(eq_df, cat_col="Sector", val_col="Weight %")
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

        # ================== FIXED INCOME ==================
        elif tab_choice == "Fixed Income Asset List" and "FixedIncomeAssetList" in dfs:
            fi_df = dfs["FixedIncomeAssetList"]

            st.subheader("Fixed Income – Summary")
            st.dataframe(fi_df.head())

            fig = bar(fi_df, cat_col="Rating", val_col="Weight %")
            fig.update_layout(legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No matching sheet found in uploaded file.")

        # ================== EXPORTS ==================
        st.subheader("Export Reports")
        c1, c2 = st.columns(2)

        if generate_excel_report:
            with c1:
                if st.button("Download Excel Report"):
                    excel_bytes = generate_excel_report(dfs)
                    st.download_button(
                        "Download Excel",
                        data=excel_bytes,
                        file_name="portfolio_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

        if generate_pdf_report:
            with c2:
                if st.button("Download PDF Report"):
                    pdf_bytes = generate_pdf_report(dfs)
                    st.download_button(
                        "Download PDF",
                        data=pdf_bytes,
                        file_name="portfolio_report.pdf",
                        mime="application/pdf",
                    )

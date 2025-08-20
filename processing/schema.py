# processing/schema.py  (REPLACE EVERYTHING WITH THIS)

import pandera as pa
from pandera import Column, DataFrameSchema, Check

# ---- PortfolioMaster (baseline)
PortfolioMasterSchema = DataFrameSchema({
    "Category": Column(str, nullable=False, coerce=True),
    "Account": Column(str, nullable=False, coerce=True),
    "Asset Class": Column(str, nullable=False, coerce=True),
    "Sub Asset Class": Column(str, nullable=False, coerce=True),
    "Liquid/Illiquid": Column(str, nullable=True, coerce=True),
    "Exposure": Column(str, nullable=True, coerce=True),
    "FX": Column(str, nullable=True, coerce=True),
    "Manager": Column(str, nullable=True, coerce=True),
    "Asset": Column(str, nullable=True, coerce=True),
    "USD Total": Column(float, nullable=True, coerce=True, checks=Check.ge(0.0)),
    "ISIN": Column(str, nullable=True, coerce=True),
    "%": Column(float, nullable=True, coerce=True),
    "BBG": Column(str, nullable=True, coerce=True),
    "Number": Column(float, nullable=True, coerce=True),
})

# ---- EquityAssetList
EquityAssetListSchema = DataFrameSchema({
    "Category": Column(str, nullable=True, coerce=True),
    "Account": Column(str, nullable=True, coerce=True),
    "Asset Class": Column(str, nullable=True, coerce=True),
    "Sub Asset Class": Column(str, nullable=True, coerce=True),
    "Region": Column(str, nullable=True, coerce=True),
    "Country": Column(str, nullable=True, coerce=True),
    "Sector (GICS)": Column(str, nullable=True, coerce=True),
    "Liquid/Illiquid": Column(str, nullable=True, coerce=True),
    "FX": Column(str, nullable=True, coerce=True),
    "Manager": Column(str, nullable=True, coerce=True),
    "Asset (Security Name)": Column(str, nullable=True, coerce=True),
    "Ticker": Column(str, nullable=True, coerce=True),
    "ISIN": Column(str, nullable=True, coerce=True),
    "SEDOL": Column(str, nullable=True, coerce=True),
    "BBG Ticker": Column(str, nullable=True, coerce=True),
    "Shares": Column(float, nullable=True, coerce=True),
    "Price (Local)": Column(float, nullable=True, coerce=True),
    "Price Date": Column(pa.DateTime, nullable=True, coerce=True),
    "Market Value (USD)": Column(float, nullable=True, coerce=True, checks=Check.ge(0.0)),
    "Weight %": Column(float, nullable=True, coerce=True, checks=Check.ge(0.0)),
    "Benchmark": Column(str, nullable=True, coerce=True),
    "Active Weight %": Column(float, nullable=True, coerce=True),
    "Notes": Column(str, nullable=True, coerce=True),
})

# ---- FixedIncomeAssetList
FixedIncomeAssetListSchema = DataFrameSchema({
    "Category": Column(str, nullable=True, coerce=True),
    "Account": Column(str, nullable=True, coerce=True),
    "Asset Class": Column(str, nullable=True, coerce=True),
    "Sub Asset Class": Column(str, nullable=True, coerce=True),
    "Region": Column(str, nullable=True, coerce=True),
    "Country": Column(str, nullable=True, coerce=True),
    "Liquid/Illiquid": Column(str, nullable=True, coerce=True),
    "FX": Column(str, nullable=True, coerce=True),
    "Manager": Column(str, nullable=True, coerce=True),
    "Asset (Bond Name)": Column(str, nullable=True, coerce=True),
    "Ticker": Column(str, nullable=True, coerce=True),
    "ISIN": Column(str, nullable=True, coerce=True),
    "BBG Ticker": Column(str, nullable=True, coerce=True),
    "Coupon %": Column(float, nullable=True, coerce=True),
    "Coupon Type (Fixed/Floating/Zero)": Column(str, nullable=True, coerce=True),
    "Frequency (Annual/Semi/Quarterly)": Column(str, nullable=True, coerce=True),
    "Issue Date": Column(pa.DateTime, nullable=True, coerce=True),
    "Maturity Date": Column(pa.DateTime, nullable=True, coerce=True),
    "Next Call Date": Column(pa.DateTime, nullable=True, coerce=True),
    "Callable (Y/N)": Column(str, nullable=True, coerce=True),
    "Puttable (Y/N)": Column(str, nullable=True, coerce=True),
    "Clean Price": Column(float, nullable=True, coerce=True),
    "Dirty Price": Column(float, nullable=True, coerce=True),
    "Accrued Interest": Column(float, nullable=True, coerce=True),
    "Face Value": Column(float, nullable=True, coerce=True),
    "Market Value (USD)": Column(float, nullable=True, coerce=True, checks=Check.ge(0.0)),
    "Yield to Maturity %": Column(float, nullable=True, coerce=True),
    "Modified Duration": Column(float, nullable=True, coerce=True),
    "Convexity": Column(float, nullable=True, coerce=True),
    "OAS (bps)": Column(float, nullable=True, coerce=True),
    "Rating (S&P/Moody's/Fitch)": Column(str, nullable=True, coerce=True),
    "Seniority/Type": Column(str, nullable=True, coerce=True),
    "Security Type (Gov/Corp/EM/HY/ABS/MBS)": Column(str, nullable=True, coerce=True),
    "Notes": Column(str, nullable=True, coerce=True),
    "Weight %": Column(float, nullable=True, coerce=True, checks=Check.ge(0.0)),
})

def business_rules(df, template_type):
    rules = []
    if template_type in ("EquityAssetList", "FixedIncomeAssetList") and "Weight %" in df.columns:
        rules.append((
            abs((df["Weight %"].fillna(0).sum()) - 100.0) <= 0.5,
            "Weight % does not sum to ~100% (±0.5)."
        ))
    if template_type == "FixedIncomeAssetList":
        if "Maturity Date" in df.columns and "Issue Date" in df.columns:
            ok = (df["Maturity Date"] >= df["Issue Date"]).fillna(True).all()
            rules.append((ok, "Some bonds have Maturity Date earlier than Issue Date."))
    return rules

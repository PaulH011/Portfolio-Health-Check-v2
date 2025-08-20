# processing/pipeline.py
from __future__ import annotations

import io
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Excel I/O
# ============================================================================

def read_workbook(file_obj) -> Tuple[Dict[str, pd.DataFrame], Optional[str]]:
    """
    Read all sheets from an uploaded .xlsx file.

    Returns:
        sheets: dict of {sheet_name: DataFrame}
        meta:   the value of A1 on a sheet named 'Meta' (if present), else None
    """
    # Accept both file-like object from Streamlit and bytes
    if hasattr(file_obj, "read"):
        data = file_obj.read()
        bio = io.BytesIO(data)
    else:
        bio = io.BytesIO(file_obj)

    xls = pd.ExcelFile(bio, engine="openpyxl")
    sheets: Dict[str, pd.DataFrame] = {}

    for name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=name, engine="openpyxl")
            # Normalise columns: strip whitespace
            df.columns = [str(c).strip() for c in df.columns]
            sheets[name] = df
        except Exception:
            # Even an empty sheet should become an empty DF
            sheets[name] = pd.DataFrame()

    meta_value = None
    if "Meta" in sheets and sheets["Meta"].shape[0] > 0:
        # A1 of Meta is often a single string like 'Template v2.0 - PortfolioMaster'
        try:
            meta_value = str(sheets["Meta"].iloc[0, 0])
        except Exception:
            meta_value = None

    return sheets, meta_value


# ============================================================================
# Template detection / required sheet name
# ============================================================================

def detect_template_type(sheets: Dict[str, pd.DataFrame], meta_value: Optional[str]) -> str:
    """
    Detect which template this workbook most likely uses.
    Order of checks:
      1) Use 'Meta' A1 if present.
      2) Use presence of well-known sheet names.
    Returns one of {"PortfolioMaster","EquityAssetList","FixedIncomeAssetList"}.
    """
    if meta_value:
        s = meta_value.lower()
        if "portfoliomaster" in s or "pastor" in s:
            return "PortfolioMaster"
        if "equityassetlist" in s:
            return "EquityAssetList"
        if "fixedincomeassetlist" in s or "fixed income" in s:
            return "FixedIncomeAssetList"

    names_lower = {n.lower() for n in sheets.keys()}
    if {"portfoliomaster"} & names_lower or {"pastor"} & names_lower:
        return "PortfolioMaster"
    if {"equityassetlist"} & names_lower:
        return "EquityAssetList"
    if {"fixedincomeassetlist", "fixedincome"}.intersection(names_lower):
        return "FixedIncomeAssetList"

    # default fallback
    return "PortfolioMaster"


def get_required_sheet_for_type(tmpl_type: str) -> str:
    """Return the canonical sheet name used by each template in this project."""
    mapping = {
        "PortfolioMaster": "PortfolioMaster",
        "EquityAssetList": "EquityAssetList",
        "FixedIncomeAssetList": "FixedIncomeAssetList",
    }
    return mapping.get(tmpl_type, "PortfolioMaster")


# ============================================================================
# Validation (lightweight – only required columns)
# ============================================================================

PORTFOLIO_MASTER_REQUIRED = ["Asset Class", "Sub Asset Class", "FX", "USD Total"]
PORTFOLIO_MASTER_OPTIONAL = [
    "Region", "Country", "Country ISO3", "Weight %", "Liquidity", "Vehicle Type",
    "TER %", "Exp Return % (annual)", "Perf YTD %", "Perf 1Y %", "Perf 3Y % (ann)",
    "Volatility %", "Max Drawdown %", "Gross Exposure %", "Net Exposure %",
    "Benchmark", "ESG Score", "Carbon Intensity",
]

EQUITY_REQUIRED = ["Asset (Security Name)", "Market Value (USD)"]
EQUITY_OPTIONAL = ["Weight %", "Sector (GICS)", "Region"]

FI_REQUIRED = ["Rating"]
FI_OPTIONAL = [
    "Market Value (USD)", "Weight %", "Years to Maturity", "Maturity (Years)",
    "Mod Duration", "Duration"
]


def validate_df(df: pd.DataFrame, tmpl_type: str):
    """
    Return a list of validation errors.
    Keep this intentionally lightweight/forgiving so users can experiment.
    """
    errors = []
    cols = set(df.columns)

    if tmpl_type == "PortfolioMaster":
        for c in PORTFOLIO_MASTER_REQUIRED:
            if c not in cols:
                errors.append({"column": c, "failure": "missing required"})
        return errors

    if tmpl_type == "EquityAssetList":
        for c in EQUITY_REQUIRED:
            if c not in cols:
                errors.append({"column": c, "failure": "missing required"})
        return errors

    if tmpl_type == "FixedIncomeAssetList":
        for c in FI_REQUIRED:
            if c not in cols:
                errors.append({"column": c, "failure": "missing required"})
        return errors

    return errors


# ============================================================================
# Transformations / Analytics
# ============================================================================

# ---- small helper utilities -------------------------------------------------

def _num(s, default=np.nan):
    """Coerce a Series-like to numeric; missing -> default."""
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _coalesce_weight(df: pd.DataFrame) -> pd.Series:
    """
    Prefer provided Weight %, else compute from USD Total.
    Returns % weights that sum ~100 (if USD Total present).
    """
    if _exists(df, "Weight %"):
        w = _num(df["Weight %"]).astype(float)
        if _exists(df, "USD Total"):
            # If provided weights don’t sum ~100, recompute from USD Total
            usd = _num(df["USD Total"]).astype(float)
            if usd.sum() > 0 and (not np.isfinite(w.sum()) or abs(w.sum() - 100) > 0.25):
                w = usd / usd.sum() * 100.0
        return w

    if _exists(df, "USD Total"):
        usd = _num(df["USD Total"]).astype(float)
        if usd.sum() > 0:
            return usd / usd.sum() * 100.0
    return pd.Series(0.0, index=df.index)


def _safe_group_sum(df: pd.DataFrame, by: str, val: str = "USD Total") -> pd.DataFrame:
    """Group by a column and sum a numeric column; return sorted frame."""
    if not _exists(df, val):
        return pd.DataFrame(columns=[by, val])
    g = df.groupby(by, dropna=False, as_index=False)[val].sum()
    g[val] = _num(g[val]).astype(float)
    g = g.sort_values(val, ascending=False).reset_index(drop=True)
    return g


def _read_policy(sheets: Dict[str, pd.DataFrame]):
    """
    Read optional 'Policy' and 'PolicyMeta' sheets.
    Returns:
        policy_df or None, meta_dict
    """
    policy = None
    if "Policy" in sheets:
        p = sheets["Policy"].copy()
        if {"Asset Class", "Policy Weight %"}.issubset(p.columns):
            p["Policy Weight %"] = _num(p["Policy Weight %"]).astype(float)
            policy = p.groupby("Asset Class", as_index=False)["Policy Weight %"].sum()

    meta = {}
    if "PolicyMeta" in sheets and sheets["PolicyMeta"].shape[1] >= 2:
        try:
            meta = dict(zip(sheets["PolicyMeta"].iloc[:, 0], sheets["PolicyMeta"].iloc[:, 1]))
        except Exception:
            meta = {}
    return policy, meta


# ---- main transform ---------------------------------------------------------

def transform_results(df: pd.DataFrame, tmpl_type: str, *, sheets: Dict[str, pd.DataFrame] | None = None) -> dict:
    """
    Compute all derived tables/metrics for a given template.

    For PortfolioMaster (v2):
      - metrics (rows, totals, top-10 concentration)
      - by_asset_class, by_sub_asset, by_fx, by_liquidity
      - by_country (ISO3 or Country)
      - policy_compare (if Policy sheet present)
      - fees_returns (MV-weighted gross/net expected return, TER)
      - esg (portfolio vs benchmark, carbon vs benchmark)
      - top_assets (largest by USD Total)

    For EquityAssetList:
      - metrics (n_rows, mv_sum, w_sum)
      - by_sector, by_region, top_positions

    For FixedIncomeAssetList:
      - metrics (n_rows, mv_sum, w_sum)
      - by_rating
      - maturity_buckets (if maturity years available)
      - duration_buckets (if duration available)
    """
    out: dict = {"metrics": {}}

    if tmpl_type == "PortfolioMaster":
        work = df.copy()

        usd = _num(work["USD Total"]) if _exists(work, "USD Total") else pd.Series(0.0, index=work.index)
        w = _coalesce_weight(work)
        work["__usd"] = usd
        work["__w"] = w

        # ---- metrics
        out["metrics"] = {
            "n_rows": int(len(work)),
            "usd_total_sum": float(usd.sum()),
            "w_sum": float(w.sum()),
        }
        # top-10 concentration (by weight)
        w_sorted = work.sort_values("__w", ascending=False)["__w"]
        out["metrics"]["top10_concentration_%"] = float(w_sorted.head(10).sum())

        # ---- core groupings
        out["by_asset_class"] = _safe_group_sum(work, "Asset Class", "__usd").rename(columns={"__usd": "USD Total"})
        out["by_sub_asset"] = _safe_group_sum(work, "Sub Asset Class", "__usd").rename(columns={"__usd": "USD Total"})
        out["by_fx"] = _safe_group_sum(work, "FX", "__usd").rename(columns={"__usd": "USD Total"})
        out["by_liquidity"] = _safe_group_sum(work, "Liquidity", "__usd").rename(columns={"__usd": "USD Total"})

        # ---- country exposures (ISO3 preferred)
        if _exists(work, "Country ISO3"):
            cc = work.groupby("Country ISO3", as_index=False)["__usd"].sum().rename(columns={"__usd": "USD Total"})
            out["by_country"] = cc
        elif _exists(work, "Country"):
            cc = work.groupby("Country", as_index=False)["__usd"].sum().rename(columns={"__usd": "USD Total"})
            out["by_country"] = cc
        else:
            out["by_country"] = pd.DataFrame(columns=["Country ISO3", "USD Total"])

        # ---- policy compare (optional)
        policy, meta = _read_policy(sheets or {})
        out["policy"] = policy
        if policy is not None and not out["by_asset_class"].empty:
            actual = out["by_asset_class"][["Asset Class", "USD Total"]].copy()
            actual["Actual %"] = actual["USD Total"] / actual["USD Total"].sum() * 100.0
            comp = actual.merge(policy, on="Asset Class", how="outer").fillna(0.0)
            comp = comp[["Asset Class", "Actual %", "Policy Weight %"]]
            out["policy_compare"] = comp

        # ---- fees & expected returns (MV-weighted)
        ter = _num(work["TER %"]) if _exists(work, "TER %") else pd.Series(0.0, index=work.index)
        exp_ret = _num(work["Exp Return % (annual)"]) if _exists(work, "Exp Return % (annual)") else pd.Series(np.nan, index=work.index)
        weighted = (exp_ret * w / 100.0)
        gross = float(weighted.replace({np.nan: 0.0}).sum())
        ter_w = float((ter * w / 100.0).replace({np.nan: 0.0}).sum())
        net = float(gross - ter_w)
        out["fees_returns"] = pd.DataFrame({
            "Metric": ["Gross Exp Return %", "TER % (weighted)", "Net Exp Return %"],
            "Value": [gross, ter_w, net],
        })

        # ---- ESG & Carbon (MV-weighted) vs benchmark (from PolicyMeta)
        esg = _num(work["ESG Score"]) if _exists(work, "ESG Score") else pd.Series(np.nan, index=work.index)
        carb = _num(work["Carbon Intensity"]) if _exists(work, "Carbon Intensity") else pd.Series(np.nan, index=work.index)
        w_norm = w / w.sum() if w.sum() else w
        esg_w = float((esg * w_norm).replace({np.nan: 0.0}).sum())
        carb_w = float((carb * w_norm).replace({np.nan: 0.0}).sum())
        esg_bmk = float(_num(pd.Series([meta.get("ESG_Benchmark_Score", np.nan)])).iloc[0]) if meta else np.nan
        carb_bmk = float(_num(pd.Series([meta.get("Carbon_Benchmark_Intensity", np.nan)])).iloc[0]) if meta else np.nan
        out["esg"] = pd.DataFrame({
            "Metric": ["Portfolio ESG", "Benchmark ESG", "Portfolio Carbon", "Benchmark Carbon"],
            "Value": [esg_w, esg_bmk, carb_w, carb_bmk],
        })

        # ---- top assets (largest by USD)
        cols_keep = [c for c in work.columns if c not in {"__usd", "__w"}]
        out["top_assets"] = work.sort_values("__usd", ascending=False)[cols_keep].head(15).reset_index(drop=True)

        return out

    # ------------------------------------------------------------------------
    # Equity Asset List
    # ------------------------------------------------------------------------
    if tmpl_type == "EquityAssetList":
        work = df.copy()
        mv = None
        if _exists(work, "Market Value (USD)"):
            mv = _num(work["Market Value (USD)"]).astype(float)
        elif _exists(work, "USD Total"):
            mv = _num(work["USD Total"]).astype(float)
        else:
            mv = pd.Series(0.0, index=work.index)

        if _exists(work, "Weight %"):
            w = _num(work["Weight %"]).astype(float)
        else:
            w = (mv / mv.sum() * 100.0) if mv.sum() else pd.Series(0.0, index=work.index)

        out["metrics"] = {
            "n_rows": int(len(work)),
            "mv_sum": float(mv.sum()),
            "w_sum": float(w.sum()),
        }

        # groupings
        if _exists(work, "Sector (GICS)"):
            out["by_sector"] = work.groupby("Sector (GICS)", as_index=False).agg(**{"Weight %": ("Weight %", "sum") if _exists(work, "Weight %") else (lambda x: np.nan)})
            if not _exists(out["by_sector"], "Weight %") or out["by_sector"]["Weight %"].isna().all():
                out["by_sector"] = work.groupby("Sector (GICS)", as_index=False)["Market Value (USD)"].sum().rename(columns={"Market Value (USD)": "Weight %"})
        else:
            out["by_sector"] = pd.DataFrame(columns=["Sector (GICS)", "Weight %"])

        if _exists(work, "Region"):
            out["by_region"] = work.groupby("Region", as_index=False).agg(**{"Weight %": ("Weight %", "sum") if _exists(work, "Weight %") else (lambda x: np.nan)})
            if not _exists(out["by_region"], "Weight %") or out["by_region"]["Weight %"].isna().all():
                out["by_region"] = work.groupby("Region", as_index=False)["Market Value (USD)"].sum().rename(columns={"Market Value (USD)": "Weight %"})
        else:
            out["by_region"] = pd.DataFrame(columns=["Region", "Weight %"])

        # top positions
        name_col = "Asset (Security Name)" if _exists(work, "Asset (Security Name)") else work.columns[0]
        top = work.copy()
        top["__mv"] = mv
        out["top_positions"] = top.sort_values("__mv", ascending=False).drop(columns=["__mv"]).head(25).reset_index(drop=True)

        return out

    # ------------------------------------------------------------------------
    # Fixed Income Asset List
    # ------------------------------------------------------------------------
    if tmpl_type == "FixedIncomeAssetList":
        work = df.copy()

        mv = None
        if _exists(work, "Market Value (USD)"):
            mv = _num(work["Market Value (USD)"]).astype(float)
        elif _exists(work, "USD Total"):
            mv = _num(work["USD Total"]).astype(float)
        else:
            mv = pd.Series(0.0, index=work.index)

        if _exists(work, "Weight %"):
            w = _num(work["Weight %"]).astype(float)
        else:
            w = (mv / mv.sum() * 100.0) if mv.sum() else pd.Series(0.0, index=work.index)

        out["metrics"] = {
            "n_rows": int(len(work)),
            "mv_sum": float(mv.sum()),
            "w_sum": float(w.sum()),
        }

        # by rating
        if _exists(work, "Rating"):
            if _exists(work, "Weight %"):
                by_r = work.groupby("Rating", as_index=False)["Weight %"].sum()
            else:
                by_r = work.groupby("Rating", as_index=False)["Market Value (USD)"].sum().rename(columns={"Market Value (USD)": "Weight %"})
            out["by_rating"] = by_r.sort_values(by_r.columns[-1], ascending=False).reset_index(drop=True)
        else:
            out["by_rating"] = pd.DataFrame(columns=["Rating", "Weight %"])

        # maturity buckets
        mat_col = "Years to Maturity" if _exists(work, "Years to Maturity") else ("Maturity (Years)" if _exists(work, "Maturity (Years)") else None)
        if mat_col:
            mats = _num(work[mat_col]).astype(float)
            buckets = pd.cut(
                mats,
                bins=[-np.inf, 1, 3, 5, 7, 10, np.inf],
                labels=["0-1y", "1-3y", "3-5y", "5-7y", "7-10y", "10y+"]
            )
            tmp = pd.DataFrame({"Bucket": buckets, "W": w})
            out["maturity_buckets"] = tmp.groupby("Bucket", as_index=False)["W"].sum().rename(columns={"W": "Weight %"})
        else:
            out["maturity_buckets"] = pd.DataFrame(columns=["Bucket", "Weight %"])

        # duration buckets
        dur_col = "Mod Duration" if _exists(work, "Mod Duration") else ("Duration" if _exists(work, "Duration") else None)
        if dur_col:
            durs = _num(work[dur_col]).astype(float)
            buckets = pd.cut(
                durs,
                bins=[-np.inf, 1, 3, 5, 7, 10, np.inf],
                labels=["0-1", "1-3", "3-5", "5-7", "7-10", "10+"]
            )
            tmp = pd.DataFrame({"Bucket": buckets, "W": w})
            out["duration_buckets"] = tmp.groupby("Bucket", as_index=False)["W"].sum().rename(columns={"W": "Weight %"})
        else:
            out["duration_buckets"] = pd.DataFrame(columns=["Bucket", "Weight %"])

        return out

    # If we reach here, return empty structure
    return out

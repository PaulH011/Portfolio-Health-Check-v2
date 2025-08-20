import io
import pandas as pd
import xlsxwriter

def _write_table(ws, df, start_row=0, start_col=0, bold=None):
    if bold is None: bold = ws.book.add_format({"bold": True, "bg_color": "#F2F2F2"})
    ws.write_row(start_row, start_col, list(df.columns), bold)
    for i, row in enumerate(df.itertuples(index=False), start_row+1):
        for j, val in enumerate(row, start_col):
            ws.write(i, j, val)

def build_report_xlsx(results: dict) -> bytes:
    out = io.BytesIO()
    wb = xlsxwriter.Workbook(out, {"in_memory": True})
    hdr = wb.add_format({"bold": True, "bg_color": "#F2F2F2"})
    pct = wb.add_format({"num_format": "0.00%"})
    ws = wb.add_worksheet("Summary")
    ws.write_row(0,0,["Metric","Value"], hdr)
    r = 1
    for k,v in results.get("metrics", {}).items():
        ws.write(r,0,k); ws.write(r,1,v); r += 1

    t = results.get("template_type")

    def add_chart(sheet_name, df, cat, val, chart_type="column"):
        w = wb.add_worksheet(sheet_name)
        _write_table(w, df, 0, 0, hdr)
        ch = wb.add_chart({"type": chart_type})
        n = len(df)
        ch.add_series({
            "name": sheet_name,
            "categories": [sheet_name, 1, 0, n, 0],
            "values": [sheet_name, 1, 1, n, 1],
            "data_labels": {"value": True},
        })
        ch.set_title({"name": sheet_name})
        w.insert_chart(1, 4, ch, {"x_scale": 1.1, "y_scale": 1.1})

    if t == "PortfolioMaster":
        if "by_asset_class" in results:
            add_chart("ByAssetClass", results["by_asset_class"][["Asset Class","USD Total"]], "Asset Class","USD Total")
        if "by_sub_asset" in results:
            add_chart("BySubAsset", results["by_sub_asset"][["Sub Asset Class","USD Total"]], "Sub Asset Class","USD Total")
        if "by_fx" in results:
            add_chart("Currency", results["by_fx"][["FX","USD Total"]], "FX","USD Total")
        if "top_assets" in results:
            w = wb.add_worksheet("TopAssets"); _write_table(w, results["top_assets"], 0, 0, hdr)

    if t == "EquityAssetList":
        if "by_sector" in results:
            add_chart("BySector", results["by_sector"][["Sector (GICS)","Weight %"]], "Sector (GICS)","Weight %")
        if "by_region" in results:
            add_chart("ByRegion", results["by_region"][["Region","Weight %"]], "Region","Weight %")
        if "top_positions" in results:
            w = wb.add_worksheet("TopPositions"); _write_table(w, results["top_positions"], 0, 0, hdr)

    if t == "FixedIncomeAssetList":
        if "by_rating" in results:
            add_chart("ByRating", results["by_rating"][["Rating","Weight %"]], "Rating","Weight %")
        if "maturity_buckets" in results:
            add_chart("Maturity", results["maturity_buckets"][["Bucket","Weight %"]], "Bucket","Weight %")
        if "duration_buckets" in results:
            add_chart("Duration", results["duration_buckets"][["Bucket","Weight %"]], "Bucket","Weight %")

    wb.close()
    out.seek(0)
    return out.read()

def build_validation_report(errors: list[dict]) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(errors).to_excel(buf, engine="xlsxwriter", index=False, sheet_name="Validation")
    buf.seek(0)
    return buf.read()

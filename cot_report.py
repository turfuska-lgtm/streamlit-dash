# cot_report.py
import streamlit as st
import pandas as pd
import requests, os, zipfile, io, datetime

def app():
    st.set_page_config(page_title="COT Dashboard", layout="wide")

    # --------------------------------------------------
    # دوال الجلب والحفظ
    # --------------------------------------------------
    CFTC_BASE = "https://www.cftc.gov/files/dea/history/{}_{}.zip"
    DATA_DIR = "data/cot"
    os.makedirs(DATA_DIR, exist_ok=True)

    def local_path(year, report_type):
        return os.path.join(DATA_DIR, f"{report_type}_{year}.parquet")

    def fetch_cot(year, report_type, force=False):
        file_out = local_path(year, report_type)
        if not force and os.path.isfile(file_out):
            return pd.read_parquet(file_out)

        st.info("جاري تحميل ملف ZIP من CFTC ...")
        key = {"Legacy": "fut_fin_txt", "Disaggregated": "fut_disagg_txt"}[report_type]
        url = CFTC_BASE.format(key, year)
        r = requests.get(url, timeout=60)
        r.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            txt_name = [f for f in z.namelist() if f.endswith(".txt")][0]
            with z.open(txt_name) as txt:
                df = pd.read_csv(txt, low_memory=False)

        df.columns = [c.strip() for c in df.columns]
        df["Report_Date_as_YYYY-MM-DD"] = pd.to_datetime(df["Report_Date_as_YYYY-MM-DD"], errors="coerce")
        df = df.dropna(subset=["Report_Date_as_YYYY-MM-DD"])
        df.to_parquet(file_out, index=False)
        return df

    # --------------------------------------------------
    # حسابات COT-Index + ΔNet %OI + Momentum
    # --------------------------------------------------
    WINDOW_INDEX = 156   # 3 سنوات تقريباً

    def enrich_cot(df, long_col, short_col, mom_weeks):
        df = df.copy()
        df["Net_Position"] = df[long_col] - df[short_col]

        # ΔNet %OI الأسبوعي
        df["DeltaNet_OI"] = df["Net_Position"].diff()

        # COT-Index (0-100)
        roll = df["Net_Position"].rolling(window=WINDOW_INDEX, min_periods=1)
        df["COT_Index"] = ((df["Net_Position"] - roll.min()) /
                           (roll.max() - roll.min() + 1e-9)) * 100

        # COT Momentum آخر N أسبوع
        df["COT_Momentum"] = (df["DeltaNet_OI"]
                              .rolling(window=mom_weeks, min_periods=1)
                              .sum())
        return df

    # --------------------------------------------------
    # واجهة Streamlit
    # --------------------------------------------------
    st.title("📊 COT Dashboard (Legacy & Disaggregated)")

    with st.sidebar:
        year = st.selectbox("السنة", list(range(2015, datetime.datetime.now().year + 1)),
                            index=len(list(range(2015, datetime.datetime.now().year + 1))) - 1)
        report_type = st.selectbox("نوع التقرير", ["Legacy", "Disaggregated"])
        mom_weeks = st.sidebar.slider("Momentum Weeks", 1, 12, 4)
        force_reload = st.sidebar.button("🔄 إعادة تحميل")

    df = fetch_cot(year, report_type, force=force_reload)

    market_list = sorted(df["Market_and_Exchange_Names"].dropna().unique())
    keyword = st.sidebar.text_input("بحث سريع (إنجليزي)", "")
    if keyword:
        market_list = [m for m in market_list if keyword.lower() in m.lower()]
    selected = st.sidebar.selectbox("اختر السوق", market_list)

    # أعمدة المراكز حسب نوع التقرير
    if report_type == "Legacy":
        long_col  = "Pct_of_OI_Dealer_Long_All"
        short_col = "Pct_of_OI_Dealer_Short_All"
    else:
        long_col  = "Pct_of_OI_Producer_Merchant_Processor_User_Long_All"
        short_col = "Pct_of_OI_Producer_Merchant_Processor_User_Short_All"

    sub = df[df["Market_and_Exchange_Names"] == selected].copy()
    sub = enrich_cot(sub, long_col, short_col, mom_weeks).sort_values("Report_Date_as_YYYY-MM-DD")

    # --------------------------------------------------
    # العرض
    # --------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📈 COT-Index (0-100)")
        st.line_chart(sub.set_index("Report_Date_as_YYYY-MM-DD")["COT_Index"])

    with col2:
        st.subheader("📊 ΔNet %OI (Weekly Flow)")
        st.line_chart(sub.set_index("Report_Date_as_YYYY-MM-DD")["DeltaNet_OI"])

    with col3:
        st.subheader(f"⚡ COT Momentum ({mom_weeks}W)")
        st.line_chart(sub.set_index("Report_Date_as_YYYY-MM-DD")["COT_Momentum"])

    st.caption("آخر 10 أسابيع")
    st.dataframe(
        sub[["Report_Date_as_YYYY-MM-DD",
             "Net_Position",
             "DeltaNet_OI",
             "COT_Momentum",
             "COT_Index"]]
        .tail(10)
        .sort_values("Report_Date_as_YYYY-MM-DD", ascending=False)
        .style.format("{:.2f}")
    )

    # --------------------------------------------------
    # تنزيل CSV
    # --------------------------------------------------
    csv = sub.to_csv(index=False).encode()
    st.sidebar.download_button(
        label="⬇️ CSV كامل",
        data=csv,
        file_name=f"{selected}_{year}_{report_type}_full.csv",
        mime="text/csv"
    )

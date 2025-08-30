# volatility.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from arch import arch_model


def app():
    # ===============================
    # 🎯 إعدادات الصفحة
    # ===============================
    st.title("📈 شبكة انتقال تقلبات الفوركس (GARCH)")

    # ===============================
    # 🔐 إدخال API Key
    # ===============================
    API_KEY = st.text_input("🔑 أدخل مفتاح Twelve Data API", type="password", value="")

    # ===============================
    # 📌 إعدادات التحليل
    # ===============================
    all_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD", "USD/CAD"]
    selected_pairs = st.multiselect(
        "اختر أزواج الفوركس:",
        options=all_pairs,
        default=["EUR/USD", "GBP/USD", "USD/JPY"]
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        interval = st.selectbox("الـ Timeframe", ["1min", "5min", "15min", "30min", "1h", "4h", "1day"], index=6)
    with col2:
        periods = st.slider("عدد الشموع", min_value=100, max_value=1000, value=500)
    with col3:
        threshold = st.slider("حد الربط في الشبكة", 0.0, 1.0, 0.5)

    # ===============================
    # 🚀 زر بدء التحليل
    # ===============================
    if st.button("ابدأ التحليل"):
        if not API_KEY:
            st.error("⚠️ من فضلك أدخل مفتاح API")
        elif len(selected_pairs) < 2:
            st.error("⚠️ اختر زوجين على الأقل")
        else:
            with st.spinner("📥 جاري جلب البيانات من Twelve Data..."):
                data_dict = {}
                for pair in selected_pairs:
                    url = f"https://api.twelvedata.com/time_series"
                    params = {
                        "symbol": pair,
                        "interval": interval,
                        "outputsize": periods,
                        "apikey": API_KEY,
                        "format": "JSON"
                    }
                    try:
                        r = requests.get(url, params=params).json()
                        if "values" in r:
                            df = pd.DataFrame(r["values"])
                            df["datetime"] = pd.to_datetime(df["datetime"])
                            df.set_index("datetime", inplace=True)
                            df.sort_index(inplace=True)
                            data_dict[pair] = df["close"].astype(float)
                        else:
                            st.warning(f"⚠️ خطأ في جلب {pair}: {r.get('message', 'Unknown')}")
                    except Exception as e:
                        st.error(f"فشل في جلب {pair}: {e}")

                if data_dict:
                    # دمج الأسعار
                    prices = pd.DataFrame(data_dict)
                    returns = prices.pct_change().dropna()

                    # عرض الأسعار
                    st.subheader("📊 أسعار الإغلاق")
                    st.line_chart(prices)

                    # ===============================
                    # 📈 نموذج GARCH - التقلبات الشرطية
                    # ===============================
                    st.subheader("📉 التقلبات الشرطية (GARCH(1,1))")
                    vol_df = pd.DataFrame(index=returns.index)
                    for pair in selected_pairs:
                        try:
                            model = arch_model(returns[pair], vol='GARCH', p=1, q=1, dist='normal')
                            res = model.fit(disp='off')
                            vol_df[pair] = res.conditional_volatility
                        except:
                            st.warning(f"فشل في تقدير GARCH لـ {pair}")
                    
                    st.line_chart(vol_df)

                    # ===============================
                    # 🔗 مصفوفة الارتباط
                    # ===============================
                    corr_matrix = vol_df.corr()

                    st.subheader("🔗 ارتباط التقلبات")
                    fig1, ax1 = plt.subplots(figsize=(8, 6))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
                    ax1.set_title("مصفوفة ارتباط التقلبات الشرطية")
                    st.pyplot(fig1)

                    # ===============================
                    # 🌐 شبكة انتقال التقلبات
                    # ===============================
                    st.subheader("🌐 شبكة انتقال التقلبات (Volatility Spillover)")
                    G = nx.DiGraph()
                    for i in selected_pairs:
                        for j in selected_pairs:
                            if i != j:
                                w = corr_matrix.loc[i, j]
                                if abs(w) > threshold:
                                    G.add_edge(i, j, weight=round(w, 2))

                    if len(G.nodes) == 0:
                        st.info("❌ ما فيش روابط تفوق الحد المحدد. جرب تنقص الـ threshold.")
                    else:
                        pos = nx.spring_layout(G, seed=42)
                        fig2, ax2 = plt.subplots(figsize=(10, 8))
                        nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', ax=ax2)
                        nx.draw_networkx_edges(G, pos, width=2, edge_color='gray', arrows=True, ax=ax2)
                        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax2)
                        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
                        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax2)
                        ax2.set_title("شبكة انتقال التقلبات")
                        ax2.axis('off')
                        st.pyplot(fig2)

                    # ===============================
                    # 📥 تنزيل البيانات
                    # ===============================
                    @st.cache_data
                    def convert_df(df):
                        return df.to_csv().encode('utf-8')

                    csv = convert_df(prices)
                    st.download_button(
                        "📥 تنزيل بيانات الأسعار (CSV)",
                        csv,
                        "forex_prices.csv",
                        "text/csv"
                    )
                else:
                    st.error("❌ ما تقدرش تلقا أي بيانات. تحقق من الـ API Key أو الاتصال.")

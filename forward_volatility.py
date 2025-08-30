# forward_volatility.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model

# --------------------------
# الإعدادات
# --------------------------
TRADING_DAYS = 252

# --------------------------
# تحميل البيانات
# --------------------------
def fetch_price_history(ticker, period="2y"):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval="1d")
        if hist.empty:
            st.error("لم يتم العثور على بيانات. تحقق من الرمز.")
            return None
        return hist[['Close']].rename(columns={'Close': 'price'})
    except Exception as e:
        st.error(f"خطأ في التحميل: {e}")
        return None

# --------------------------
# حساب العوائد
# --------------------------
def prepare_returns(df):
    df = df.copy()
    df['logret'] = np.log(df['price'] / df['price'].shift(1))
    return df.dropna()

# --------------------------
# GARCH(1,1) - توقع Forward Volatility
# --------------------------
def garch_forecast_vol(df, horizon_days):
    """
    يستخدم GARCH(1,1) للتنبؤ بالتقلب في الأفق المحدد (مثلاً 30 يوم)
    """
    returns = df['logret'] * 100
    am = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal', mean='Zero')
    try:
        res = am.fit(disp='off')
        forecasts = res.forecast(horizon=horizon_days, reindex=False)
        var_forecast = forecasts.variance.values.flatten() / (100**2)  # إلى وحدة عشرية
        vol_forecast = np.sqrt(var_forecast) * np.sqrt(TRADING_DAYS)  # سنوي
        return vol_forecast
    except:
        return np.full(horizon_days, 0.20)  # افتراضي

# --------------------------
# التقلب الضمني (IV) من Yahoo (إن وُجد)
# --------------------------
def get_implied_volatility(ticker):
    """
    يحاول جلب التقلب الضمني من الخيارات (إن وُجد)
    ملاحظة: yfinance لا يدعم IV مباشرة، لكن نستخدم تقديراً
    """
    if "USD" in ticker or "EUR" in ticker or "=X" in ticker:
        return None  # الفوركس ما عندهش IV
    try:
        tk = yf.Ticker(ticker)
        opts = tk.options
        if not opts:
            return None
        near_date = opts[0]  # أول تاريخ
        opt = tk.option_chain(near_date)
        if not opt.calls.empty:
            at_the_money = opt.calls.iloc[(opt.calls['strike'] - tk.history(period="1d")['Close'].iloc[-1]).abs().argsort()[:1]]
            if not at_the_money.empty and 'impliedVolatility' in at_the_money.columns:
                iv = at_the_money['impliedVolatility'].iloc[0]
                return iv * 100  # نسبة مئوية
    except:
        pass
    return None

# --------------------------
# واجهة Streamlit
# --------------------------
def app():
    st.markdown("<h1 style='text-align: center; color: #8E44AD;'>🔮 Forward Volatility: التقلب المستقبلي</h1>", unsafe_allow_html=True)
    st.write("توقع التقلب في الأفق القريب (30 يوم) باستخدام GARCH والخيارات")

    # --- إدخالات المستخدم ---
    st.subheader("🔍 اختر الأصل الذي تريد تحليله")

    # تصنيف الأصول
    assets = {
        "📈 المؤشرات": {
            "S&P 500": "SPY",
            "Nasdaq 100": "QQQ",
            "Dow Jones": "DIA",
            "FTSE 100": "UKX",
            "DAX (ألمانيا)": "DAX",
            "CAC 40 (فرنسا)": "FCHI"
        },
        "💵 الفوركس (FX)": {
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X",
            "USD/JPY": "USDJPY=X",
            "AUD/USD": "AUDUSD=X",
            "USD/CAD": "USDCAD=X",
            "NZD/USD": "NZDUSD=X",
            "USD/CHF": "USDCHF=X"
        },
        "🪙 السلع": {
            "الذهب": "GC=F",
            "الفضة": "SI=F",
            "النفط الخام (WTI)": "CL=F",
            "نفط برنت": "BZ=F",
            "النحاس": "HG=F",
            "الذرة": "ZC=F"
        },
        "⚡ العملات الرقمية": {
            "Bitcoin": "BTC-USD",
            "Ethereum": "ETH-USD",
            "Solana": "SOL-USD",
            "Cardano": "ADA-USD",
            "Dogecoin": "DOGE-USD"
        },
        "🏢 أسهم كبرى": {
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Google": "GOOGL",
            "Amazon": "AMZN",
            "Meta": "META",
            "Tesla": "TSLA",
            "NVIDIA": "NVDA",
            "Netflix": "NFLX"
        }
    }

    # قائمة مسطحة
    all_options = []
    ticker_mapping = {}
    for category, items in assets.items():
        for name, ticker in items.items():
            display_name = f"{name} ({ticker}) - {category.replace('📈 ', '').replace('💵 ', '').replace('🪙 ', '').replace('⚡ ', '').replace('🏢 ', '')}"
            all_options.append(display_name)
            ticker_mapping[display_name] = ticker

    # اختيار المستخدم
    selected_asset = st.selectbox(
        "اختر زوجًا أو أصلًا",
        options=all_options,
        index=all_options.index("S&P 500 (SPY) - المؤشرات")  # الافتراضي
    )

    # استخراج الرمز
    ticker = ticker_mapping[selected_asset]
    st.info(f"**الرمز المستخدم**: `{ticker}`")

    # اختيار الفترة
    period = st.selectbox("المدة", ["1y", "2y", "5y"], index=1)

    if st.button("🚀 توقع التقلب المستقبلي"):
        with st.spinner("تحليل التقلب المستقبلي..."):
            df = fetch_price_history(ticker, period=period)
            if df is None:
                return

            df = prepare_returns(df)
            st.success(f"✅ تم تحميل {len(df)} يوم من بيانات `{ticker}`")

            # --- 1. توقع GARCH لـ 30 يوم ---
            st.subheader("📊 توقع التقلب باستخدام GARCH(1,1)")

            horizon = 30
            forward_vol = garch_forecast_vol(df, horizon)

            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=list(range(1, horizon + 1)),
                y=forward_vol * 100,
                mode='lines+markers',
                name='Forward Vol (GARCH)',
                line=dict(color='purple', width=2)
            ))
            fig1.add_hline(y=forward_vol[0]*100, line_dash="dash", line_color="green", annotation_text="اليوم 1")
            fig1.add_hline(y=forward_vol[-1]*100, line_dash="dash", line_color="red", annotation_text="اليوم 30")

            fig1.update_layout(
                title=f"منحنى التقلب المستقبلي (Forward Volatility Curve) - {ticker}",
                xaxis_title="الأيام المستقبلية",
                yaxis_title="التقلب السنوي (%)",
                template="plotly_white"
            )
            st.plotly_chart(fig1, use_container_width=True)

            # --- 2. إحصائيات ---
            st.subheader("📈 إحصائيات التقلب المستقبلي")
            col1, col2, col3 = st.columns(3)
            col1.metric("التقلب المتوقع (اليوم 1)", f"{forward_vol[0]*100:.1f}%")
            col2.metric("التقلب المتوقع (اليوم 30)", f"{forward_vol[-1]*100:.1f}%")
            col3.metric("متوسط 30 يوم", f"{forward_vol.mean()*100:.1f}%")

            # --- 3. التقلب الضمني (IV) إن وُجد ---
            st.subheader("📉 التقلب الضمني (Implied Volatility)")
            iv = get_implied_volatility(ticker.replace("=X", ""))
            if iv:
                st.metric("IV من الخيارات", f"{iv:.1f}%")
                if iv > forward_vol[0]*100:
                    st.info("السوق يتوقع تقلبًا أعلى من النموذج → فرصة بيع خيارات؟")
                else:
                    st.warning("النموذج يتوقع تقلبًا أعلى → فرصة شراء خيارات؟")
            else:
                st.info("لا توجد بيانات للخيارات (مثلاً: الفوركس أو العملات الرقمية)")

            # --- 4. إشارات التداول ---
            st.subheader("🎯 إشارات التداول")
            if forward_vol[-1] > forward_vol[0]:
                st.warning("📈 **اتجاه تصاعدي في التقلب**: احتمال اضطراب قادم → خفّض الحجم أو اشترِ تحوط (Hedge)")
            elif forward_vol[-1] < forward_vol[0]:
                st.success("📉 **اتجاه تنازلي في التقلب**: استقرار قادم → يمكن زيادة الحجم")
            else:
                st.info("🔄 **التقلب مستقر**: استمر بالاستراتيجية الحالية")

            # --- 5. تنزيل ---
            forecast_df = pd.DataFrame({
                "Day": range(1, horizon + 1),
                "Forward_Vol_%": forward_vol * 100
            })

            @st.cache_data
            def convert_df(_df):
                return _df.to_csv(index=False).encode('utf-8')

            csv = convert_df(forecast_df)
            st.download_button(
                "📥 تنزيل توقعات التقلب",
                csv,
                "forward_volatility_forecast.csv",
                "text/csv"
            )

    # --- ملاحظات ---
    with st.expander("📚 مصادر"):
        st.markdown("""
        - **GARCH**: نموذج إحصائي قوي لتوقع التقلب
        - **Forward Volatility Curve**: تستخدم في أسواق الخيارات والفيوتشرز
        - **تطبيقات**:
          - تسعير الخيارات
          - إدارة محفظة ديناميكية
          - تحديد توقيت الدخول/الخروج
        """)
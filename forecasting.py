# forecasting.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
import altair as alt

def app():
    st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🔮 تنبؤ بالذكاء الاصطناعي</h1>", unsafe_allow_html=True)
    st.write("استراتيجية تداول تعتمد على **LSTM + ATR Regime**")

    # إعدادات
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("رمز الزوج", value="EURUSD=X")
    with col2:
        start_date = st.date_input("تاريخ البدء", value=pd.to_datetime("2018-01-01"))

    atr_window = st.slider("نافذة ATR المتوسط", 5, 30, 14)
    epochs_per_task = st.slider("عدد الدورات (Epochs) لكل سنة", 1, 10, 5)

    if st.button("🚀 بدء التحليل"):
        with st.spinner("تحميل البيانات وتحليلها..."):

            # 1. تحميل البيانات
            try:
                df_raw = yf.download(symbol, start=start_date, interval='1d')
                if df_raw.empty:
                    st.error("لم يتم العثور على بيانات. تحقق من الرمز أو التاريخ.")
                    return
            except Exception as e:
                st.error(f"خطأ في تحميل البيانات: {e}")
                return

            st.success(f"✅ تم تحميل {len(df_raw)} يوم من البيانات.")

            # 2. ترتيب حسب التاريخ
            df_raw = df_raw.sort_index()

            # 3. حساب ATR و ATR_MA
            df_raw['ATR'] = df_raw['High'] - df_raw['Low']
            df_raw['ATR_MA'] = df_raw['ATR'].rolling(window=atr_window).mean()
            df_raw.dropna(inplace=True)

            # 4. Regime
            df_raw['Regime'] = (df_raw['ATR'] > df_raw['ATR_MA']).astype(int)

            # 5. تقسيم حسب السنة
            df_raw['Year'] = df_raw.index.year
            tasks = [group for _, group in df_raw.groupby('Year') if len(group) >= 20]

            # 6. نموذج LSTM
            def create_model():
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(20, 1)),
                    tf.keras.layers.LSTM(64),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model

            model = create_model()

            # 7. تحضير البيانات
            def prepare_data(data):
                X, y = [], []
                closes = data['Close'].values
                regimes = data['Regime'].values
                for i in range(20, len(data)):
                    X.append(closes[i-20:i])
                    y.append(regimes[i])
                return np.array(X).reshape(-1, 20, 1), np.array(y)

            # 8. التعلم المستمر
            training_log = []
            for task in tasks:
                year = task['Year'].iloc[0]
                X_task, y_task = prepare_data(task)
                if len(X_task) == 0:
                    continue
                model.fit(X_task, y_task, epochs=epochs_per_task, verbose=0)
                loss, acc = model.evaluate(X_task, y_task, verbose=0)
                training_log.append(f"السنة {year}: الدقة = {acc:.3f}")

            st.subheader("📈 نتائج التدريب")
            for log in training_log:
                st.write(log)

            # 9. التنبؤ على كامل البيانات
            X_all, y_true = prepare_data(df_raw)
            y_pred = (model.predict(X_all) > 0.5).astype(int).flatten()

            # 10. إشارات التداول
            lstm_window = 20
            signals = [None] * lstm_window
            for i in range(lstm_window, len(df_raw)):
                regime = float(df_raw.iloc[i]['Regime'])
                close = float(df_raw.iloc[i]['Close'])
                prev_close = float(df_raw.iloc[i - 1]['Close'])
                if regime == 0 and close > prev_close:
                    signals.append('Buy')
                elif regime == 1:
                    signals.append('Hold')
                else:
                    signals.append('Sell')

            # 11. DataFrame بالإشارات
            df = df_raw.iloc[lstm_window:].copy()
            df['Signal'] = signals[lstm_window:]

            # 12. آخر النتائج
            st.subheader("📊 الإشارات الأخيرة")
            st.dataframe(df[['Close', 'ATR', 'ATR_MA', 'Regime', 'Signal']].tail(10))

            # 13. رسم بياني
            st.subheader("📈 رسم OHLC تفاعلي")
            df_plot = df[['Open', 'High', 'Low', 'Close', 'Signal', 'Regime']].reset_index()

            color_condition = alt.condition(
                alt.datum.Open <= alt.datum.Close,
                alt.value("#06982d"),
                alt.value("#ae1325")
            )

            rule = alt.Chart(df_plot).mark_rule().encode(
                x=alt.X('Date:T', title='التاريخ', axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('Low:Q', title='السعر', scale=alt.Scale(zero=False)),
                y2='High:Q',
                color=color_condition
            ).properties(width=800, height=400)

            bar = alt.Chart(df_plot).mark_bar(width=6).encode(
                x=alt.X('Date:T'),
                y='Open:Q',
                y2='Close:Q',
                color=color_condition
            )

            # Buy & Sell markers
            buy_signals = df_plot[df_plot['Signal'] == 'Buy']
            sell_signals = df_plot[df_plot['Signal'] == 'Sell']

            points_buy = alt.Chart(buy_signals).mark_point(
                shape='triangle', size=100, color='green', filled=True
            ).encode(x='Date:T', y='Low:Q', tooltip=['Date:T', 'Close:Q']) if len(buy_signals) > 0 else None

            points_sell = alt.Chart(sell_signals).mark_point(
                shape='triangle-down', size=100, color='red', filled=True
            ).encode(x='Date:T', y='High:Q', tooltip=['Date:T', 'Close:Q']) if len(sell_signals) > 0 else None

            # مناطق التقلب
            regime_areas = df_plot[df_plot['Regime'] == 1]
            band = None
            if not regime_areas.empty:
                price_min = float(df_plot['Low'].min())
                price_max = float(df_plot['High'].max())
                band = alt.Chart(regime_areas).mark_rect(opacity=0.1, color='orange').encode(
                    x='Date:T',
                    x2=alt.value(0),
                    y=alt.value(price_min),
                    y2=alt.value(price_max)
                )

            # دمج
            chart = rule + bar
            if points_buy:
                chart += points_buy
            if points_sell:
                chart += points_sell
            if band:
                chart += band

            st.altair_chart(chart, use_container_width=True)

            # 14. تحميل CSV
            @st.cache_data
            def convert_df_to_csv(_df):
                return _df[['Close', 'ATR', 'ATR_MA', 'Regime', 'Signal']].to_csv().encode('utf-8')

            csv = convert_df_to_csv(df)
            st.download_button("📥 تنزيل النتائج", csv, "forecasting_signals.csv", "text/csv")

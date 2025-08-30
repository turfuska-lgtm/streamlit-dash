import streamlit as st
from streamlit_option_menu import option_menu
import cot_report
import fundamentals
import volatility
import forecasting
import forward_volatility
import vol_spillover

# دمج CSS
with open("style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar Menu
with st.sidebar:
    selected = option_menu(
        menu_title="📊 Main Menu",
        options=["COT Report", "Fundamentals", "Volatility", "Forecasting", "Forward Vol", "Vol Spillover"],
        icons=["bar-chart", "book", "activity", "graph-up-arrow", "trending-up", "arrows-expand"],
        menu_icon="cast",
        default_index=0,
    )

# ربط الصفحات
if selected == "COT Report":
    cot_report.app()

elif selected == "Fundamentals":
    fundamentals.app()

elif selected == "Volatility":
    volatility.app()

elif selected == "Forecasting":
    forecasting.app()

elif selected == "Forward Vol":
    forward_volatility.app()

elif selected == "Vol Spillover":
    vol_spillover.app()
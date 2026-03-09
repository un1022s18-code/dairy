import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import base64


st.set_page_config(
    page_title="DairyOptima",
    layout="wide"
)

def svg_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

money_icon = svg_to_base64("images/money.svg")

st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)



# =====================================================
# 🎨 Global CSS (화이트 SaaS 스타일)
# =====================================================
# =====================================================
# 🎨 Global CSS (회색 SaaS 스타일)
# =====================================================
st.markdown("""
<style>

/* 전체 배경 */
[data-testid="stAppViewContainer"] {
    background-color: #f3f4f6;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color: #ffffff;
}

/* 컨텐츠 여백 */
.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 🏷 헤더
# =====================================================



st.title("💰 수익 최적화")

st.markdown("""
<div style="color:#6b7280;font-size:15px;font-weight:600;">
스마트 낙농 수익 최적화 의사결정 대시보드
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='margin-top:0px;margin-bottom:30px;'>", unsafe_allow_html=True)
st.subheader("💵 최적 수익 분석")
with st.sidebar:

    

    st.title("📂 메뉴")
    st.markdown('')

    if st.button("📊 전체현황"):
        st.switch_page("ov.py")
        
    if st.button("🔎 개체파악"):
        st.switch_page("pages/11.py")

    if st.button("📈 생산량 예측"):
        st.switch_page("pages/predict.py")

    if st.button("💰 수익 최적화"):
        st.switch_page("pages/stt.py")

    st.divider()
    st.header("⚙️ 설정")
# =====================================================
# 데이터 & 모델 로드
# =====================================================
df = pd.read_csv("final_diary_data.csv")

import os
import pickle

# 현재 파일(11.py 등)의 절대 경로를 가져옵니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

# 상위 폴더(diary/)로 이동하여 모델 파일 경로를 설정합니다.
model_high_path = os.path.join(current_dir, "..", "model_high.pkl")
model_low_path = os.path.join(current_dir, "..", "model_low.pkl")

# model_high 로드
if os.path.exists(model_high_path):
    with open(model_high_path, "rb") as f:
        model_high = pickle.load(f)

# model_low 로드
if os.path.exists(model_low_path):
    with open(model_low_path, "rb") as f:
        model_low = pickle.load(f)

threshold = 9

# =====================================================
# Sidebar
# =====================================================
cattle_list = df["Cattle_ID"].unique().tolist()

default_index = cattle_list.index("CATTLE_003885")

selected_cattle = st.sidebar.selectbox(
    "Cattle ID",
    cattle_list,
    index=default_index
)
milk_price = st.sidebar.number_input(
    "우유 가격 (₩/L)",
    min_value=0,
    value=2000,
    step=10
)

feed_cost = st.sidebar.number_input(
    "사료 가격 (₩/kg)",
    min_value=0,
    value=200,
    step=10
)

# =====================================================
# 데이터 선택
# =====================================================
cow_data = df[df["Cattle_ID"] == selected_cattle]
cow_row = cow_data.iloc[0]

last_week = cow_row["Previous_Week_Avg_Yield"]
model_used = model_low if last_week <= threshold else model_high
trained_features = model_used.booster_.feature_name()

feed_range = np.linspace(6, 50, 200)

pred_milk_list = []
profit_list = []

for feed in feed_range:
    input_dict = {}
    for col in trained_features:
        input_dict[col] = cow_row[col] if col in cow_row.index else 0

    input_dict["Feed_Quantity_kg"] = feed
    input_df = pd.DataFrame([input_dict])[trained_features]

    pred_milk = model_used.predict(input_df)[0]
    profit = (pred_milk * milk_price) - (feed * feed_cost)

    pred_milk_list.append(pred_milk)
    profit_list.append(profit)

pred_milk_array = np.array(pred_milk_list)
profit_array = np.array(profit_list)

max_index = np.argmax(profit_array)
optimal_feed = feed_range[max_index]
optimal_profit = profit_array[max_index]
optimal_milk = pred_milk_array[max_index]

breakeven_idx = np.where(profit_array > 0)[0]
breakeven_feed = feed_range[breakeven_idx[0]] if len(breakeven_idx) > 0 else 0

current_profit = (
    cow_row["Milk_Yield_L"] * milk_price
    - cow_row["Feed_Quantity_kg"] * feed_cost
)

daily_gain = optimal_profit - current_profit
annual_gain = daily_gain * 365




# =====================================================
# ⭐ KPI 카드
# =====================================================


k1, k2, k3, k4 = st.columns(4)

def kpi_card(title, value):
    st.markdown(f"""
    <div style="
        background:#ffffff;
        padding:25px;
        border-radius:18px;
        box-shadow:0 6px 18px rgba(0,0,0,0.08);
        height:140px;
    ">
        <div style="color:#000000;font-size:14px;">{title}</div>
        <div style="font-size:32px;font-weight:700;color:#2E8B57;margin-top:10px;">
            {value}
        </div>
    </div>
    """, unsafe_allow_html=True)

with k1:
    kpi_card("최적 급여량", f"{optimal_feed:.1f} kg")

with k2:
    kpi_card("최대 순이익", f"₩{int(optimal_profit):,}")

with k3:
    kpi_card("예상 생산량", f"{optimal_milk:.1f} L")

with k4:
    kpi_card("연간 수익 개선", f"₩{int(annual_gain):,}")

st.markdown("<br>", unsafe_allow_html=True)

st.divider()
# =====================================================
# 📈 Profit Curve
# =====================================================
st.subheader("📈 급여량별 수익 분석")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=feed_range,
    y=profit_array,
    mode="lines",
    line=dict(width=4, color="#2E8B57"),
    name="순이익"
))

fig.add_trace(go.Scatter(
    x=[cow_row["Feed_Quantity_kg"]],
    y=[current_profit],
    mode="markers",
    marker=dict(size=12, color="red"),
    name="현재 급여"
))

fig.add_trace(go.Scatter(
    x=[optimal_feed],
    y=[optimal_profit],
    mode="markers",
    marker=dict(size=14, color="#F7C429"),
    name="수익 최대점"
))

fig.add_vline(x=breakeven_feed, line_dash="dash")

fig.update_layout(
    template="plotly_white",
    xaxis=dict(range=[5,50], showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#EAECEE"),
    xaxis_title="사료 급여량 (kg)",
    yaxis_title="순이익 (₩)",
    height=450
)

st.plotly_chart(fig, use_container_width=True)


#----------------------------------------------------------
# 📊 Dynamic Sensitivity Analysis
# =========================================================
st.subheader("📊 원유/사료가격 변화에 따른 수익 분석")

price_list = np.arange(milk_price - 200, milk_price + 201, 100)
price_list = [p for p in price_list if p > 0]

rows = []

for price in price_list:

    temp_profit = (pred_milk_array * price) - (feed_range * feed_cost)

    idx = np.argmax(temp_profit)
    max_p = temp_profit[idx]
    opt_feed = feed_range[idx]

    breakeven_idx = np.where(temp_profit > 0)[0]
    breakeven = feed_range[breakeven_idx[0]] if len(breakeven_idx) > 0 else 0

    rows.append([
        price,
        round(opt_feed, 1),
        int(max_p),
        round(breakeven, 1)
    ])

result_df = pd.DataFrame(
    rows,
    columns=[
        "우유 가격 (원)",
        "최적 급여량 (kg)",
        "최대 순이익 (원)",
        "손익분기점 (kg)"
    ]
)

def highlight_current_price(row):
    if row["우유 가격 (원)"] == milk_price:
        return ['background-color: #FFF3A3'] * len(row)   # 노란색
    else:
        return [''] * len(row)

styled_df = result_df.style.apply(highlight_current_price, axis=1)

st.dataframe(styled_df, use_container_width=True)


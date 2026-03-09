import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import base64


# -----------------------
# 페이지 설정
# -----------------------
st.set_page_config(page_title="DairyOptima", layout="wide")

# -----------------------
# 색상 팔레트
# -----------------------
green_palette = [
    "#1C461E",
    "#2E7D32",
    "#4CAF50",
    "#81C784",
    "#A5D6A7",
    "#C8E6C9",
    "#E8F5E9",
]
def svg_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

money_icon = svg_to_base64("images/money.svg")
# -----------------------
# UI 스타일
# -----------------------

# -----------------------
# UI 스타일
# -----------------------
st.markdown("""
<style>

/* 기본 사이드바 네비 숨김 */
[data-testid="stSidebarNav"] {display:none;}

/* 전체 배경 */
[data-testid="stAppViewContainer"] {
    background-color:#f3f4f6;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color:#ffffff;
}

/* 컨텐츠 여백 */
.block-container {
    padding-top:3rem;
    padding-bottom:2rem;
    padding-left:4rem;
    padding-right:4rem;
}

/* KPI 카드 */
[data-testid="stMetric"]{
background:#ffffff;
border-radius:20px;
padding:25px;
height:140px;
box-shadow:0 4px 12px rgba(0,0,0,0.06);
}

/* KPI 제목 */
[data-testid="stMetricLabel"]{
color:#000000;
font-size:14px;
}

/* KPI 값 */
[data-testid="stMetricValue"]{
color:#2E7D32;
font-weight:700;
font-size:26px;
}

/* 입력창 높이 */
div[data-testid="stTextInput"] input{
height:38px;
}

</style>
""", unsafe_allow_html=True)
# -----------------------
# 데이터 로드
# -----------------------
df = pd.read_csv("final_diary_data.csv")
# -----------------------
# Sidebar
# -----------------------
with st.sidebar:

    st.title("📂 메뉴")
    st.markdown('')


    if st.button("📊 전체현황 "):
        st.switch_page("ov.py")
        
    if st.button("🔎 개체파악"):
        st.switch_page("pages/11.py")

    if st.button("📈 생산량 예측"):
        st.switch_page("pages/predict.py")

    if st.button("💰 수익 최적화"):
        st.switch_page("pages/stt.py")

    st.divider()

    st.header("⚙️ 설정")
    cattle_list = df["Cattle_ID"].unique().tolist()

    

    # CSS (한 번만 넣기)
st.markdown("""
<style>
div[data-testid="stTextInput"] input{
height:38px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:

    st.text_input(
        "Cattle ID",
        value="",
        disabled=True
    )

    milk_price = st.number_input(
        "우유 단가 (₩ / L)",
        min_value=0,
        value=1000,
        step=50
    )

    st.text_input(
        "사료 가격 (₩/kg)",
        value="",
        disabled=True
    )




# -----------------------
# Header
# -----------------------
st.title("📊 전체 현황")
st.markdown("""
<div style="color:#6b7280;font-size:15px;font-weight:600;">
스마트 낙농 수익 최적화 의사결정 대시보드
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='margin-top:0px;margin-bottom:30px;'>", unsafe_allow_html=True)

# -----------------------
# 제목 스타일
# -----------------------
title_style = """
<h3 style="
margin-top:0px;
margin-bottom:12px;
font-weight:700;
font-size:26px;">
{}
</h3>
"""



# -----------------------
# KPI 계산
# -----------------------
total_milk = df["Milk_Yield_L"].sum()

df["Feed_Cost"] = df["Feed_Quantity_kg"] * df["Feed_Cost_per_kg"]
total_feed_cost = df["Feed_Cost"].sum()

total_revenue = total_milk * milk_price
housing_score = df["Housing_Score"].mean()

total_prev = df["Previous_Week_Avg_Yield"].sum()
total_pred = df["Milk_Yield_L"].sum()

total_change_rate = ((total_pred-total_prev)/total_prev)*100

prev_revenue = total_prev * milk_price
revenue_change_rate = ((total_revenue-prev_revenue)/prev_revenue)*100

roi = ((total_revenue-total_feed_cost)/total_feed_cost)*100
avg_thi = df["THI"].mean()

# -----------------------
# KPI 카드
# -----------------------
st.subheader("⭐️ 주요 KPI")
col1,col2,col3,col4,col5 = st.columns(5)

col1.metric(" 총 원유 생산량",f"{total_milk:,.1f} L",delta=f"{total_change_rate:.1f}%")
col2.metric(" 총 수익",f"₩{total_revenue:,.0f}",delta=f"{revenue_change_rate:+.1f}%")
col3.metric(" 사료 ROI",f"{roi:.1f}%")
col4.metric(" 평균 THI",f"{avg_thi:.1f}")
col5.metric(" 하우징 스코어",f"{housing_score*100:.0f}점")

st.divider()

# -----------------------
# Row1
# -----------------------
col1,col2 = st.columns([1.5,1])

# 생산량 추이
with col1:

    st.subheader("📈 날짜별 생산량 추이")

    days=np.arange(1,31)
    base=25
    milk=base+np.random.normal(0,1.5,len(days))

    temp_df=pd.DataFrame({
        "날짜":[f"Day {d}" for d in days],
        "생산량":milk
    })

    fig=px.line(
        temp_df,
        x="날짜",
        y="생산량",
        markers=True,
        template="plotly_white"
    )

    fig.update_traces(
        line_color="#2E7D32",
        fill="tozeroy",
        fillcolor="rgba(46,125,50,0.1)"
    )

    fig.update_layout(height=420)

    st.plotly_chart(fig,use_container_width=True)

# -----------------------
# 질병 도넛
# -----------------------
with col2:

    st.subheader("😷 질병 분류 개체 비율")

    disease_counts = df["disease_class"].value_counts().reset_index()
    disease_counts.columns = ["Disease","Count"]

    total_cows = disease_counts["Count"].sum()

    # 비율 계산
    disease_counts["Percent"] = disease_counts["Count"] / total_cows * 100

    # 3% 이하 → Others
    disease_counts["Disease"] = disease_counts.apply(
        lambda x: x["Disease"] if x["Percent"] >= 3 else "Others",
        axis=1
    )
    disease_kor = {
    "Healthy": "건강",
    "Infectious Diseases": "감염성 질병",
    "Metabolic": "대사 질환",
    "Nutritional": "영양 질환",
    "Management-Related": "관리 관련",
    "Others": "기타"
}

    disease_counts["Disease"] = disease_counts["Disease"].map(disease_kor)
    disease_counts = disease_counts.groupby("Disease",as_index=False)["Count"].sum()
    

    # 도넛차트
    fig = px.pie(
        disease_counts,
        values="Count",
        names="Disease",
        hole=0.60,
        color_discrete_sequence=green_palette
    )

    # ⭐ 도넛 안에 label + percent 표시
    fig.update_traces(
        textinfo="label+percent",
        textposition="inside",
        customdata=disease_counts["Count"]
    )

    fig.update_layout(showlegend=False,height=500)

    fig.add_annotation(
        x=0.5,
        y=0.56,
        text="총 사육 수",
        showarrow=False,
        font=dict(size=16,color="#6b7280")
    )

    fig.add_annotation(
        x=0.5,
        y=0.48,
        text=f"<b>{total_cows:,}</b> <span style='font-size:20px'>마리</span>",
        showarrow=False,
        font=dict(size=36,color="#2E7D32")
    )

    st.plotly_chart(fig,use_container_width=True)

# -----------------------
# Row2
# -----------------------
col1,col2 = st.columns([1.5,1])

# TOP10
with col1:

    st.markdown(title_style.format("🏆 개체별 원유 생산량 TOP 10"), unsafe_allow_html=True)

    container = st.container(height=500)

    cow_prod=(df.groupby("Cattle_ID")["Milk_Yield_L"]
              .mean()
              .sort_values(ascending=False)
              .head(10)
              .reset_index())

    medals=["🥇","🥈","🥉"]

    with container:

        for i,row in cow_prod.iterrows():

            rank = medals[i] if i<3 else f"{i+1}"
            bg = "#fff7cc" if i<3 else "#ffffff"

            st.markdown(f"""
            <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:8px 14px;
            border-radius:8px;
            margin-bottom:4px;
            background:{bg};
            border:1px solid #e5e7eb;
            font-size:16px">

            <div style="display:flex;gap:10px">
            <b style="width:24px">{rank}</b>
            🐄 <b>{row['Cattle_ID']}</b>
            </div>

            <div style="font-weight:600">
            {row['Milk_Yield_L']:.1f} L
            </div>

            </div>
            """,unsafe_allow_html=True)

# -----------------------
# 축사 도넛
# -----------------------
with col2:

    st.markdown(title_style.format("🏠 축사별 생산량 비중"), unsafe_allow_html=True)

    barn_total = df.groupby("Barn_Number")["Milk_Yield_L"].sum().reset_index()
    barn_total["Barn_Name"] = barn_total["Barn_Number"].astype(str) + "번 축사"

    total_milk = barn_total["Milk_Yield_L"].sum()

    fig = px.pie(
        barn_total,
        values="Milk_Yield_L",
        names="Barn_Name",
        hole=0.60,
        color_discrete_sequence=green_palette
    )

    fig.update_traces(
        textinfo="label+percent",
        textposition="inside",
        customdata=barn_total["Milk_Yield_L"]
    )

    fig.update_layout(showlegend=False,height=500)

    fig.add_annotation(
        x=0.5,
        y=0.56,
        text="총 원유 생산량",
        showarrow=False,
        font=dict(size=16,color="#6b7280")
    )

    fig.add_annotation(
        x=0.5,
        y=0.48,
        text=f"<b>{total_milk:,.0f}</b> <span style='font-size:25px'>L</span>",
        showarrow=False,
        font=dict(size=38,color="#2E7D32")
    )

    st.plotly_chart(fig,use_container_width=True)
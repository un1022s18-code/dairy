import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --------------------------------------------------
# 페이지 설정
# --------------------------------------------------
st.set_page_config(page_title="DairyOptima", layout="wide")

# --------------------------------------------------
# UI 스타일
# --------------------------------------------------
st.markdown("""
<style>

/* 사이드바 기본 네비 숨김 */
[data-testid="stSidebarNav"] {display:none;}

/* 전체 배경 */
[data-testid="stAppViewContainer"] {
    background-color:#f3f4f6;
}

/* 사이드바 */
[data-testid="stSidebar"] {
    background-color:#ffffff;
}

/* 컨텐츠 여백 (전체 페이지 통일) */
.block-container {
    padding-top:3rem;
    padding-bottom:2rem;
    padding-left:4rem;
    padding-right:4rem;
}

/* KPI 카드 스타일 */
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

/* column spacing */
[data-testid="column"]{
padding:10px;
}

/* 입력창 높이 통일 */
div[data-testid="stTextInput"] input{
height:38px;
}

</style>
""", unsafe_allow_html=True)



# --------------------------------------------------
# 데이터 로드
# --------------------------------------------------
df_raw = pd.read_csv("final_diary_data.csv")

feed_type_list = df_raw["Feed_Type"].unique()

categorical_cols = ["Breed", "Lactation_Stage", "disease_class"]

df = pd.get_dummies(df_raw, columns=categorical_cols)

cattle_list = df_raw["Cattle_ID"].unique().tolist()

# --------------------------------------------------
# 사이드바
# --------------------------------------------------
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
    # CSS
st.markdown("""
<style>
div[data-testid="stTextInput"] input{
height:38px;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:

    default_index = cattle_list.index("CATTLE_003885")

    selected_cattle = st.selectbox(
        "Cattle ID",
        cattle_list,
        index=default_index
    )

    st.text_input(
        "우유 단가 (₩ / L)",
        value="",
        disabled=True
    )

    st.text_input(
        "사료 가격 (₩/kg)",
        value="",
        disabled=True
    )
# --------------------------------------------------
# 모델 로드
# --------------------------------------------------
with open("model_low.pkl", "rb") as f:
    model_low = pickle.load(f)

with open("model_high.pkl", "rb") as f:
    model_high = pickle.load(f)

threshold = 9

# --------------------------------------------------
# Feature 리스트
# --------------------------------------------------
selected_cols = [
'Age_Months','Days_in_Milk','Feed_Quantity_kg','Ambient_Temperature_C',
'Housing_Score','THI',
'Breed_Ankole','Breed_Australian_Friesian_Sahiwal','Breed_Australian_Milking_Zebu',
'Breed_Ayrshire','Breed_Boran','Breed_Brown_Swiss','Breed_Butana','Breed_Danish_Red',
'Breed_Deoni','Breed_Exotic_Local_Cross','Breed_Fleckvieh','Breed_Gangatiri',
'Breed_Gir','Breed_Girolando','Breed_Guernsey','Breed_Hariana','Breed_Holstein-Friesian',
'Breed_Holstein_Zebu_Cross','Breed_Illawarra_Shorthorn','Breed_Jersey',
'Breed_Jersey_Zebu_Cross','Breed_Kankrej','Breed_Kenana','Breed_Krishna_Valley',
'Breed_Milking_Shorthorn','Breed_Montbeliarde','Breed_NDama','Breed_Normande',
'Breed_Norwegian_Red','Breed_Ongole','Breed_Rathi','Breed_Red_Poll_Africa',
'Breed_Red_Sindhi','Breed_Sahiwal','Breed_Simmental','Breed_Tharparkar',
'Breed_Tipo_Carora','Breed_White_Fulani','Breed_Zebu_Cross_Brazil',
'Lactation_Stage_Early','Lactation_Stage_Late','Lactation_Stage_Mid',
'Feed_Nutrition_Protein','Feed_Nutrition_Carbohydrate',
'Previous_Week_Avg_Yield',
'disease_class_Healthy','disease_class_Infectious Diseases',
'disease_class_Management-Related','disease_class_Metabolic',
'disease_class_Nutritional','disease_class_Traumatic'
]

# --------------------------------------------------
# Gate Model
# --------------------------------------------------
def predict_milk(input_df):

    last_week = input_df["Previous_Week_Avg_Yield"].values[0]

    if last_week <= threshold:
        pred = model_low.predict(input_df)[0]
    else:
        pred = model_high.predict(input_df)[0]

    return pred

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📈 생산량 예측")
st.markdown("""
<div style="color:#6b7280;font-size:15px;font-weight:600;">
스마트 낙농 수익 최적화 의사결정 대시보드
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='margin-top:0px;margin-bottom:30px;'>", unsafe_allow_html=True)

# --------------------------------------------------
# 레이아웃
# --------------------------------------------------
left, right = st.columns([1,1.4])


# --------------------------------------------------
# 입력 패널
# --------------------------------------------------
with left:

    st.subheader("📝 입력 조건")

    cow_id = selected_cattle

    cow_raw = df_raw[df_raw["Cattle_ID"] == cow_id].iloc[0]
    cow_data = df[df["Cattle_ID"] == cow_id].iloc[0]

    feed_type = st.selectbox(
        "사료 종류",
        feed_type_list,
        index=list(feed_type_list).index(cow_raw["Feed_Type"])
    )

    feed_amount = st.slider(
        "사료 급여량 (kg)",
        5.0, 40.0,
        float(cow_raw["Feed_Quantity_kg"])
    )

    temperature = st.slider(
        "온도 (°C)",
        -10.0, 40.0,
        float(cow_raw["Ambient_Temperature_C"])
    )

    humidity = st.slider(
        "습도 (%)",
        0.0, 100.0,
        float(cow_raw["Humidity_percent"])
    )

# --------------------------------------------------
# THI 계산
# --------------------------------------------------
thi = (
    1.8 * temperature + 32
    - 0.55 * (1 - humidity/100) * (1.8*temperature - 26)
)

def thi_warning(thi):

    if thi < 68:
        return "Normal"

    elif thi < 72:
        return "Mild Heat Stress"

    elif thi < 78:
        return "Moderate Heat Stress"

    else:
        return "Severe Heat Stress"

stress_text = thi_warning(thi)

# --------------------------------------------------
# 모델 입력 데이터 생성
# --------------------------------------------------
input_df = pd.DataFrame([cow_data])[selected_cols]

input_df["Feed_Quantity_kg"] = feed_amount
input_df["Ambient_Temperature_C"] = temperature
input_df["THI"] = thi

# Feed Type One-hot
input_df["Feed_Nutrition_Protein"] = 0
input_df["Feed_Nutrition_Carbohydrate"] = 0

if feed_type == "Protein":
    input_df["Feed_Nutrition_Protein"] = 1
else:
    input_df["Feed_Nutrition_Carbohydrate"] = 1

# --------------------------------------------------
# 예측
# --------------------------------------------------
pred = predict_milk(input_df)

feed_efficiency = pred / feed_amount

previous_week = cow_data["Previous_Week_Avg_Yield"]

change_rate = ((pred - previous_week) / previous_week) * 100

# --------------------------------------------------
# 결과 패널
# --------------------------------------------------
with right:

    st.markdown("##### ")
    st.markdown("##### ")

    col1, col2 = st.columns(2)

    col1.metric(
        " 예측 우유 생산량",
        f"{pred:.1f} L",
        delta=f"{change_rate:.1f}%"
    )

    col2.metric(
        " 사료 효율",
        f"{feed_efficiency:.2f} L/kg"
    )

    color = "#70a772" if thi < 72 else "#d98953" if thi < 78 else "#d95a53"

    st.markdown(f"""
<div style="background:#ffffff;padding:25px;border-radius:18px;box-shadow:0 6px 18px rgba(0,0,0,0.08);">

<div style="color:#000000;font-size:14px;">
 온습도 스트레스 지수 (THI)
</div>

<div style="font-size:32px;font-weight:700;color:#2E8B57;margin-top:6px;">
{thi:.1f}
</div>

<div style="margin-top:10px;"></div>

<div style="width:100%; background:#e5e7eb;border-radius:10px;height:40px;overflow:hidden;">

<div style="
width:{min(thi,100)}%;
background:{color};
height:40px;
display:flex;
align-items:center;
justify-content:center;
color:white;
font-weight:600;
border-radius:10px;
text-align:center;
">
<span style="position:relative; top:0px;">
{stress_text}
</span>


</div>

</div>

</div>
""", unsafe_allow_html=True)
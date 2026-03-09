import streamlit as st
import pandas as pd
import os
import base64
import numpy as np
import pickle

# =====================================================
# 페이지 설정
# =====================================================
st.set_page_config(page_title="DairyOptima", layout="wide")

# =====================================================
# CSS 스타일
# =====================================================
st.markdown("""
<style>

/* 사이드바 페이지 메뉴 숨김 */
[data-testid="stSidebarNav"] {display:none;}

/* 전체 배경 */
[data-testid="stAppViewContainer"] {
    background-color:#f3f4f6;
}

/* 사이드바 배경 */
[data-testid="stSidebar"] {
    background-color:#ffffff;
}

/* 컨텐츠 여백 (전체 페이지 레이아웃 통일) */
.block-container {
    padding-top:3rem;
    padding-bottom:2rem;
    padding-left:4rem;
    padding-right:4rem;
}

/* 카드 스타일 */
.card{
background:#ffffff;
padding:25px;
border-radius:20px;
box-shadow:0 4px 12px rgba(0,0,0,0.06);
margin-bottom:20px;
}

/* 카드 제목 */
.card_title{
font-size:18px;
font-weight:700;
margin-bottom:15px;
}

/* 태그 스타일 */
.tag{
display:inline-flex;
align-items:center;
height:28px;
padding:0 12px;
border-radius:16px;
background:#e8f5e9;
color:#000;
margin-right:6px;
font-size:12px;
font-weight:600;
border:1px solid #a5d6a7;
gap:4px;
white-space:nowrap;
}

/* 경고 박스 */
.warning{
background:#FDECEA;
padding:15px;
border-radius:12px;
border-left:6px solid #E74C3C;
margin-bottom:10px;
}

/* 안전 박스 */
.safe{
background:#ECF9F1;
padding:15px;
border-radius:12px;
border-left:6px solid #2E7D32;
margin-bottom:10px;
}

/* 품종 관리 박스 */
.warning-box{
background:#fff6ea;
border:1px solid #f0d3b3;
border-radius:14px;
padding:20px;
margin-top:20px;
}

/* AI 제안 박스 */
.ai-box{
background:#fdecea;
border:1px solid #f5c6cb;
border-radius:14px;
padding:20px;
margin-top:20px;
}

.warning-title{
font-weight:700;
font-size:18px;
margin-bottom:10px;
}

.warning-list{
line-height:1.8;
color:#5b4636;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 데이터 불러오기
# =====================================================
df = pd.read_csv("final_diary_data.csv")

with open("model_low.pkl","rb") as f:
    model_low = pickle.load(f)

with open("model_high.pkl","rb") as f:
    model_high = pickle.load(f)

threshold = 9

# =====================================================
# 사이드바
# =====================================================
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
# 소 선택
# =====================================================
cattle_list = df["Cattle_ID"].unique().tolist()
default_index = cattle_list.index("CATTLE_000041")

selected_cattle = st.sidebar.selectbox(
    "Cattle ID",
    cattle_list,
    index=default_index
)

milk_price = st.sidebar.number_input("우유 가격 (₩/L)",0,10000,2000,10)
feed_cost = st.sidebar.number_input("사료 가격 (₩/kg)",0,1000,200,10)

# =====================================================
# 데이터 선택
# =====================================================
cow = df[df["Cattle_ID"] == selected_cattle]
cow_row = cow.iloc[0]

breed = cow_row["Breed"]
age = cow_row["Age_Months"]
thi = cow_row["THI"]
disease = cow_row["disease_class"]

feed = cow_row["Feed_Quantity_kg"]
protein_ratio = cow_row["Feed_Nutrition_Protein"]
carb_ratio = cow_row["Feed_Nutrition_Carbohydrate"]

protein = feed * protein_ratio
carb = feed * carb_ratio

current_yield = cow_row["Previous_Week_Avg_Yield"]

# -----------------------------
# 권장 섭취량 계산
# -----------------------------
feed_ratio_dict = {
    "Concentrates": 0.02,
    "Crop Residues": 0.0125,
    "Dry Fodder": 0.02,
    "Green Fodder": 0.025,
    "Hay": 0.02,
    "Mixed Feed": 0.0325,
    "Pasture Grass": 0.0325,
    "Silage": 0.025
}

protein_ratio_dict = {
    "Early": 0.185,
    "Mid": 0.255,
    "Late": 0.218,
    "Dry": 0.21
}

weight = cow_row["Weight_kg"]
feed_type = cow_row["Feed_Type"]
lact_stage = cow_row["Lactation_Stage"]

# 사료 권장량
feed_rate = feed_ratio_dict.get(feed_type, 0.02)
feed_recommended = weight * feed_rate

# 탄수화물 권장량
carb_recommended = weight * 0.02 * 0.7

# 단백질 권장량
protein_rate = protein_ratio_dict.get(lact_stage, 0.2)
protein_recommended = weight * 0.02 * protein_rate

# =====================================================
# 생산량 예측
# =====================================================
last_week = cow_row["Previous_Week_Avg_Yield"]
model_used = model_low if last_week <= threshold else model_high

trained_features = model_used.booster_.feature_name()

feed_range = np.linspace(6,50,200)

pred_list = []
profit_list = []

for f in feed_range:

    input_dict = {}
    for col in trained_features:
        input_dict[col] = cow_row[col] if col in cow_row.index else 0

    input_dict["Feed_Quantity_kg"] = f
    input_df = pd.DataFrame([input_dict])[trained_features]

    pred = model_used.predict(input_df)[0]
    profit = (pred * milk_price) - (f * feed_cost)

    pred_list.append(pred)
    profit_list.append(profit)

optimal_milk = pred_list[np.argmax(profit_list)]

# =====================================================
# 목표 달성률
# =====================================================
progress = (current_yield / optimal_milk) * 100 if optimal_milk > 0 else 0
color = "#70a772" if progress >= 80 else "#d98953" if progress >= 60 else "#d95a53"

# =====================================================
# 품종 관리 메시지
# =====================================================
breed_warning = {

"Holstein-Friesian":"고온 스트레스와 대사성 질병에 취약 → THI와 건강상태 확인 필요",

"Fleckvieh":"체격이 커 난산 위험 존재 → 분만 전후 BCS 2.5~3.5 유지",

"Brown_Swiss":"착유 속도가 느린 품종 → 착유 간격과 착유 시간 배분 관리",

"Simmental":"비만 시 번식 효율 급격히 감소 → 사료량 관리 필요",

"Danish_Red":"높은 영양 요구 → 고품질 목초와 단백질 공급 중요",

"Norwegian_Red":"기계 착유 효율 향상을 위해 유방 형질 점검 필요",

"Ayrshire":"성격이 예민 → 큰소리나 거친 핸들링 주의",

"Montbeliarde":"체격이 커 체상태 변화 관리 중요 → BCS 2.5~3.5 유지",

"Milking_Shorthorn":"방목 적응 품종 → 방목 시간 감소 시 스트레스 주의",

"Holstein_Zebu_Cross":"세대별 성격과 생산성 차이 큼 → 개체별 맞춤 관리 필요",

"Jersey":"저칼슘혈증 발생 위험 높음 → 분만 전후 칼슘 관리 필수",

"Normande":"단백질 함량 높은 우유 생산 → 단백질 섭취량 유지",

"Exotic_Local_Cross":"질병 면역 반응 개체별 차이 큼 → 백신 스케줄 철저 관리",

"Australian_Friesian_Sahiwal":"예민한 성격 → 착유 시 차분한 환경 필요",

"Guernsey":"영양 불균형 시 번식 효율 감소 → 사료 영양 설계 중요",

"Girolando":"분만 초기 영양 부족 시 체손실 큼 → 분만 초기 영양 관리 중요",

"Jersey_Zebu_Cross":"예민한 성격 → 착유실 적응 훈련 필요",

"Illawarra_Shorthorn":"활동량 높음 → 충분한 영양 공급 필요",

"Zebu_Cross_Brazil":"기계 착유 시 유방 자극 부족 가능 → 착유 전 마사지 필요",

"Tharparkar":"사료 질 변화에 취약 → 급격한 사료 변화 금지",

"Sahiwal":"모성 본능 강함 → 송아지 분리 시 스트레스 관리 필요",

"Australian_Milking_Zebu":"더위에는 강하지만 환기와 그늘 시설 필요",

"Tipo_Carora":"발굽이 습기에 약함 → 우사 바닥 건조 상태 유지",

"Red_Sindhi":"유량은 적고 사료 효율 높음 → 건강 유지 중심 관리",

"Gir":"서열 다툼 발생 가능 → 개체 간 부상 여부 확인"

}

warning_text = breed_warning.get(breed,"품종 관리 정보 없음")

ai_warning = []

if thi >= 72:
    ai_warning.append("🌡 THI 상승 → 냉각팬 가동 필요")

if protein < 3:
    ai_warning.append("🥩 단백질 부족 → 단백질 사료 보충")

if progress < 80:
    ai_warning.append("📉 생산량 목표 미달 → 건강 점검 필요")

if progress >= 80 and thi < 72:
    ai_warning.append("✅ 생산 상태 양호")

if current_yield <= 9:
    production_group="저산유군"
    production_message="질병 관리 중심 사양 필요"
else:
    production_group="고산유군"
    production_message="환경 관리 중심 생산성 유지"

# =====================================================
# 상태
# =====================================================
thi_color="#ffcc80" if thi>=72 else "#a5d6a7"
disease_color="#ef9a9a" if disease!="Healthy" else "#a5d6a7"

# =====================================================
# UI
# =====================================================
st.title("🔎 개체 파악")

st.markdown("""
<div style="color:#6b7280;font-size:15px;font-weight:600;">
스마트 낙농 수익 최적화 의사결정 대시보드
</div>
""", unsafe_allow_html=True)
st.markdown("<hr style='margin-top:0px;margin-bottom:30px;'>", unsafe_allow_html=True)

col1,col2 = st.columns([1,1])

# =====================================================
# 왼쪽 카드
# =====================================================
with col1:
    
    image_folder = "images"
    img_path = os.path.join(image_folder, f"{breed}.png")

    # 이미지가 없으면 기본 이미지 사용
    if not os.path.exists(img_path):
        img_path = os.path.join(image_folder, "qwer.png")

    with open(img_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(f"""
<div class="card">

<div class="card_title">🐮 개체 정보</div>
<img src="data:image/png;base64,{encoded}"
style="
width:100%;
height:220px;
object-fit:cover;
border-radius:15px;
margin-top:10px;
">

<div style="margin-top:15px; 
            display:flex; 
            gap:10px; 
            align-items:center;
            flex-wrap:wrap;">

<span class="tag">{breed}</span>
<span class="tag">{int(age)} 개월</span>
<span class="tag">비유 중기</span>

<span class="tag"
style="background:{thi_color};border-color:{thi_color};white-space:nowrap;">
🌡 THI {thi:.1f}
</span>

<span class="tag"
style="background:{disease_color};border-color:{disease_color};white-space:nowrap;">
🦠 {disease}
</span>

</div>
</div>
""", unsafe_allow_html=True)

    # 품종 관리
    st.markdown(f"""
<div class="warning-box">

<div class="warning-title">📋 품종 관리 주의사항</div>

<ul class="warning-list">
<li><b style="font-weight:700;">{breed}</b> : {warning_text}</li>
<li><b>{production_group}</b> : {production_message}</li>
</ul>

</div>
""",unsafe_allow_html=True)

    # AI 관리
ai_html = f"""
<div class="ai-box">

<div class="warning-title">📌 AI 관리 제안</div>

<ul class="warning-list">
"""

for w in ai_warning:
    ai_html += f"<li>{w}</li>"

ai_html += """
</ul>
</div>
"""

st.markdown(ai_html, unsafe_allow_html=True)

# =====================================================
# 오른쪽 카드
# =====================================================
with col2:

    st.markdown(f"""
<div class="card">

<div class="card_title">🥛 목표 생산 대비 원유 생산량</div>

<div style="font-size:38px;font-weight:700;color:#2E8B57">
{current_yield:.1f} / {optimal_milk:.1f} <span style="font-size:29px;">L</span>
</div>

<div style="width:100%;background:#e5e7eb;border-radius:12px;height:44px;margin-top:10px">

<div style="
width:{min(progress,100)}%;
background:{color};
height:44px;
line-height:44px;
text-align:center;
color:white;
border-radius:12px;
font-weight:600">
{progress:.1f}% 달성
</div>

</div>

<div style="height:92px;"></div>



<div class="card_title">🍽️ 사료 섭취 상태</div>

사료 섭취량 {feed:.1f} kg

<div style="width:100%;background:#e5e7eb;height:20px;border-radius:10px">
<div style="width:{min(feed/feed_recommended*100,100)}%;background:#70a772;height:20px;border-radius:10px"></div>
</div>

<br>

탄수화물 섭취량 {carb:.2f} kg

<div style="width:100%;background:#e5e7eb;height:20px;border-radius:10px">
<div style="width:{min(carb/carb_recommended*100,100)}%;background:#8bc34a;height:20px;border-radius:10px"></div>
</div>

<br>

단백질 섭취량 {protein:.2f} kg

<div style="width:100%;background:#e5e7eb;height:20px;border-radius:10px">
<div style="width:{min(protein/protein_recommended*100,100)}%;background:#4caf50;height:20px;border-radius:10px"></div>
</div>

</div>
""", unsafe_allow_html=True)








import streamlit as st
import pandas as pd
import pickle


# -----------------------------
# 페이지 설정 (한 번만!)
# -----------------------------
st.set_page_config(
    page_title="DairyOptima",
    layout="wide"
)
st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

[data-testid="stSidebarNav"] {display: none;}

/* KPI 카드 스타일 */
[data-testid="stMetric"] {
    background-color: #ffffff;   /* 연한 회색 카드 */
    border-radius: 20px;
    padding: 25px;
    border: none;
    height: 140px;
}

/* KPI 제목 */
[data-testid="stMetricLabel"] {
    color: #6b7280;
    font-size: 14px;
}

/* KPI 숫자 */
[data-testid="stMetricValue"] {
    color: #2E7D32;   /* 초록색 숫자 */
    font-weight: 700;
    font-size: 26px;
}

/* KPI 카드 간격 */
[data-testid="column"] {
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------
# -----------------------------
# UI 스타일 (Green Dashboard)
# -----------------------------


# -----------------------------
# 사이드바
# -----------------------------
with st.sidebar:

    st.title("🐄 DairyOptima")

    st.markdown("### 📂 메뉴")

    if st.button("📊 Overview"):
        st.switch_page("ov.py")

    if st.button("📈 생산량 예측"):
        st.switch_page("pages/predict.py")

    if st.button("💰 수익 최적화"):
        st.switch_page("pages/stt.py")

    if st.button("🔎 개체파악"):
        st.switch_page("pages/cattle2.py")

    st.divider()

# ---------------------------
# 데이터 불러오기
# ---------------------------

df_raw = pd.read_csv("final_diary_data.csv")

breed_list = df_raw["Breed"].unique()
lactation_list = df_raw["Lactation_Stage"].unique()
disease_list = df_raw["disease_class"].unique()
feed_type_list = df_raw["Feed_Type"].unique()

categorical_cols = ["Breed","Lactation_Stage","disease_class"]

df = pd.get_dummies(df_raw, columns=categorical_cols)

# ---------------------------
# 모델 불러오기
# ---------------------------

with open("model_low.pkl", "rb") as f:
    model_low = pickle.load(f)

with open("model_high.pkl", "rb") as f:
    model_high = pickle.load(f)

threshold = 9

# ---------------------------
# Feature 리스트
# ---------------------------

selected_cols = ['Age_Months','Days_in_Milk','Feed_Quantity_kg','Ambient_Temperature_C','Housing_Score','THI',
 'Breed_Ankole','Breed_Australian_Friesian_Sahiwal','Breed_Australian_Milking_Zebu','Breed_Ayrshire',
 'Breed_Boran','Breed_Brown_Swiss','Breed_Butana','Breed_Danish_Red','Breed_Deoni','Breed_Exotic_Local_Cross',
 'Breed_Fleckvieh','Breed_Gangatiri','Breed_Gir','Breed_Girolando','Breed_Guernsey','Breed_Hariana',
 'Breed_Holstein-Friesian','Breed_Holstein_Zebu_Cross','Breed_Illawarra_Shorthorn','Breed_Jersey',
 'Breed_Jersey_Zebu_Cross','Breed_Kankrej','Breed_Kenana','Breed_Krishna_Valley','Breed_Milking_Shorthorn',
 'Breed_Montbeliarde','Breed_NDama','Breed_Normande','Breed_Norwegian_Red','Breed_Ongole','Breed_Rathi',
 'Breed_Red_Poll_Africa','Breed_Red_Sindhi','Breed_Sahiwal','Breed_Simmental','Breed_Tharparkar',
 'Breed_Tipo_Carora','Breed_White_Fulani','Breed_Zebu_Cross_Brazil',
 'Lactation_Stage_Early','Lactation_Stage_Late','Lactation_Stage_Mid',
 'Feed_Nutrition_Protein','Feed_Nutrition_Carbohydrate',
 'Previous_Week_Avg_Yield',
 'disease_class_Healthy','disease_class_Infectious Diseases',
 'disease_class_Management-Related','disease_class_Metabolic',
 'disease_class_Nutritional','disease_class_Traumatic'
]

# ---------------------------
# Gate Model
# ---------------------------

def predict_milk(input_df):

    last_week = input_df["Previous_Week_Avg_Yield"].values[0]

    if last_week <= threshold:
        pred = model_low.predict(input_df)[0]
    else:
        pred = model_high.predict(input_df)[0]

    return pred


# ---------------------------
# Header
# ---------------------------

st.title("🐄 DairyOptima")
st.caption("낙농 경영 최적화 대시보드")

# ---------------------------
# 레이아웃
# ---------------------------

left, right = st.columns([1,1.4])

# ---------------------------
# Cattle 선택
# ---------------------------

cow_list = df_raw["Cattle_ID"].unique()


# =====================================================================================================
# --------------------------------------
# 1단 왼쪽 - 품종 사진, 품종, 개월 수 정보
# --------------------------------------

with left:
    st.markdown("### 🐄 개체 선택")
    selected_cow = st.selectbox("분석할 개체 ID를 선택하세요", cow_list, label_visibility="collapsed")
    
    # 1. 데이터 매칭 (변수명을 cow_data로 통일하여 밑줄 에러 해결)
    input_df = df[df_raw["Cattle_ID"] == selected_cow].head(1)
    cow_data = df_raw[df_raw["Cattle_ID"] == selected_cow].iloc[0] 

    # 2. 품종 추출 로직 (에러 났던 컴프리헨션 대신 안전한 next() 사용)
    breed_cols = [col for col in selected_cols if col.startswith('Breed_')]
    active_breed_col_name = next((col for col in breed_cols if input_df[col].iloc[0] == 1), None)

    if active_breed_col_name:
        # 'Breed_Holstein-Friesian' -> 'Holstein-Friesian'
        breed_display_name = active_breed_col_name.replace("Breed_", "")
        img_filename = f"{breed_display_name}.png"
    else:
        breed_display_name = "알 수 없음"
        img_filename = "default.png"

    # 3. 경로 설정 (루트 폴더의 이미지를 찾음)
    import os 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    img_path = os.path.join(root_dir, img_filename)

    # 4. 화면 표시 (사진 출력)
    
    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.info(f"등록된 사진이 없습니다. (파일명: {img_filename})")
        st.markdown(
            f"""
            <div style="width: 100%; height: 250px; background-color: #f8f9fa; 
                        border-radius: 15px; display: flex; align-items: center; 
                        justify-content: center; border: 1px dashed #dee2e6;">
                <span style="color: #adb5bd;">{breed_display_name} 사진 준비 중</span>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # 5. 품종명 및 개월수 하단 박스 배치
    st.markdown("<br>", unsafe_allow_html=True)
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(
            f"""
            <div style="background-color: white; padding: 10px; border-radius: 10px; 
                        text-align: center; border: 1px solid #ddd; font-weight: bold; color: #333;">
                {cow_data['Breed']}
            </div>
            """, unsafe_allow_html=True
        )
        
    with info_col2:
        st.markdown(
            f"""
            <div style="background-color: white; padding: 10px; border-radius: 10px; 
                        text-align: center; border: 1px solid #ddd; font-weight: bold; color: #333;">
                {int(cow_data['Age_Months'])} 개월
            </div>
            """, unsafe_allow_html=True
        )


# --------------------------------------
# 1단 오른쪽 - 비유 단계, 분만 후 경과 일수
# --------------------------------------
with right:
    st.markdown("### 분만 후 경과 일수")

    # 1. 비유 주기 차트
    import plotly.graph_objects as go

    def draw_lactation_stage_chart(days_in_milk):

        # -----------------------------
        # 단계 정의
        # -----------------------------
        stages = ["초기", "중기", "후기", "건기"]

        colors = [
            "#006400",   # 초기
            "#32CD32",   # 중기
            "#8FBC8F",   # 후기
            "#BDBDBD",   # 건기
            "rgba(0,0,0,0)"  # 하단 여백
        ]

        # 270도 영역 + 90도 여백
        values = [1,1,1,1,4]

        # -----------------------------
        # 현재 단계 계산
        # -----------------------------
        if days_in_milk <= 70:
            stage_index = 0
        elif days_in_milk <= 200:
            stage_index = 1
        elif days_in_milk <= 305:
            stage_index = 2
        else:
            stage_index = 3

        # -----------------------------
        # 라벨 생성 (현재 단계 Bold)
        # -----------------------------
        labels = []

        for i, s in enumerate(stages):
            if i == stage_index:
                labels.append(f"<span style='color:yellow;'><b>{s}</b></span>")
            else:
                labels.append(s)

        labels.append("")  # 여백

        # -----------------------------
        # 차트 생성
        # -----------------------------
        fig = go.Figure()

        fig.add_trace(go.Pie(
            values=values,
            hole=0.75,
            marker=dict(colors=colors),
            rotation=270,   # ⭐ 하단 중앙 여백 핵심
            direction="clockwise",
            sort=False,
            text=labels,
            textinfo="text",
            textposition="inside",
            insidetextfont=dict(size=20, color="white"),
            hoverinfo="skip",
            showlegend=False
        ))

        # -----------------------------
        # 중앙 텍스트
        # -----------------------------
        fig.add_annotation(
            text=f"<b>{int(days_in_milk)} 일차</b>",
            x=0.5,
            y=0.6,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=42, color="black")
        )

        # -----------------------------
        # 레이아웃
        # -----------------------------
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        return fig

    # ---------------------------
    # 대시보드에 배치
    # ---------------------------
    st.markdown("<br>", unsafe_allow_html=True)  
    
    days_val = cow_data['Days_in_Milk']  # 실제 데이터프레임에서 추출한 숫자 값
    
    # 따옴표 없이 변수명(days_val)만 전달합니다.
    fig = draw_lactation_stage_chart(days_val)

    # 2. 스트림릿에 Plotly 차트 출력
    st.plotly_chart(fig, use_container_width=True)
    
    
    
# ---------------------------
# 품종별 관리 가이드
# ---------------------------
st.markdown("<br>", unsafe_allow_html=True)

# 품종별 관리 팁 데이터 (이미 정의하셨다면 이 변수명을 그대로 사용하세요)
breed_management_tips = {
    "Holstein-Friesian": "해당 품종은 고온 스트레스와 대사성 질병에 취약합니다. THI와 건강상태를 확인하세요.",
    "Fleckvieh": "해당 품종은 체격이 커 적정 체중 유지가 핵심입니다. 분만 전후 BCS 점수를 적정치(2.5~3.5)로 유지하세요.",
    "Brown_Swiss": "해당 품종은 착유 속도가 느리므로 착유 간격에 유의하세요.",
    "Simmental": "해당 품종은 비만 시 번식 효율이 떨어지므로 사료량 배분에 유의하세요.",
    "Danish_Red": "해당 품종은 높은 영양소를 요구하므로 고품질 목초와 단백질 공급이 중요합니다.",
    "Norwegian_Red": "해당 품종은 성격이 예민해 스트레스에 취약합니다. 큰소리나 거친 핸들링에 유의하세요.",
    "Ayrshire": "해당 품종은 성격이 예민해 스트레스에 취약합니다. 큰소리나 거친 핸들링에 유의하세요.",
    "Montbeliarde": "해당 품종은 체격이 커 적정 체중 유지가 핵심입니다. 분만 전후 BCS 점수를 적정치(2.5~3.5)로 유지하세요.",
    "Milking_Shorthorn": "해당 품종은 방목 위주 품종입니다. 방목 시간이 줄어들 시 스트레스를 받을 수 있으니 유의하세요.",
    "Holstein_Zebu_Cross": "해당 품종은 세대별로 성격과 생산성 차이가 크므로 개체별 맞춤 관리가 필요합니다.",
    "Jersey": "해당 품종은 저칼슘혈증 빈도가 높으므로 분만 전후 칼슘 대사 관리가 중요합니다.",
    "Normande": "해당 품종이 생산하는 우유는 단백질 함량이 높아 치즈용으로 많이 쓰입니다. 단백질 섭취량을 높게 유지하세요.",
    "Exotic_Local_Cross": "해당 품종은 지역 질병에 노출될 경우 면역 반응이 제각각이므로 백신 스케줄을 엄격히 준수하세요.",
    "Australian_Friesian_Sahiwal": "해당 품종은 성격이 예민해 스트레스에 취약합니다. 큰소리나 거친 핸들링에 유의하세요.",
    "Guernsey": "해당 품종은 영양 불균형 시 번식 효율이 떨어지므로 사료 영양 설계에 유의하세요.",
    "Girolando": "해당 품종은 분만 초기 관리가 핵심입니다. 영양 공급이 부족하지 않도록 주의하세요.",
    "Jersey_Zebu_Cross": "해당 품종은 성격이 예민해 스트레스에 취약합니다. 큰소리나 거친 핸들링에 유의하세요.",
    "Illawarra_Shorthorn": "해당 품종은 높은 활동량에 따른 영양공급이 중요합니다. 적절한 사료량을 유지하세요.",
    "Zebu_Cross_Brazil": "해당 품종은 기계 착유 시 유방 자극이 부족하면 유량이 나오지 않을 수 있으니 마사지를 병행하세요.",
    "Tharparkar": "해당 품종은 사료 질 변화에 취약합니다. 갑작스런 사료 변화는 지양하세요.",
    "Sahiwal": "해당 품종은 모성 본능이 강해 송아지와 분리 시 스트레스를 크게 받으므로 착유 시 심리적 안정을 유도하세요.",
    "Australian_Milking_Zebu": "해당 품종은 모성 본능이 강해 송아지와 분리 시 스트레스를 크게 받으므로 착유 시 심리적 안정을 유도하세요.",
    "Tipo_Carora": "해당 품종은 발굽이 습기에 약할 수 있으므로 습도 조절에 유의하세요.",
    "Red_Sindhi": "해당 개체는 무리한 증량 사료 급여보다는 건강 유지 위주로 관리하는 것이 경제적입니다.",
    "Gir": "해당 개체는 호전적입니다. 개체별 서열 다툼으로 인한 부상이 없는지 확인하세요."
}

# 1. 딕셔너리에서 현재 품종에 맞는 팁 가져오기
# breed_display_name은 앞서 사진 경로를 만들 때 사용한 변수입니다 (예: "Holstein-Friesian")
current_tip = breed_management_tips.get(breed_display_name, "해당 품종에 대한 특이사항이 없습니다. 일반적인 비유 단계별 관리에 유의하세요.")

# 2. 화면에 출력
# 품종명과 현재 비유 단계를 조합하여 강조 표시합니다.
st.success(f"**📌 {breed_display_name} 관리 가이드**")
st.info(current_tip)


# ---------------------------
# 2단 왼쪽 - 목표생산량, 현재생산량, 전주 평균 생산량
# ---------------------------

target_yield = 15  # 목표 생산량 (임시)

current_yield = cow_data["Previous_Week_Avg_Yield"]  # 현재 생산량 (임시 동일 사용)
last_week_yield = cow_data["Previous_Week_Avg_Yield"]

disease = cow_data["disease_class"]
thi = cow_data["THI"]

feed = cow_data["Feed_Quantity_kg"]
protein_ratio = cow_data["Feed_Nutrition_Protein"]
carb_ratio = cow_data["Feed_Nutrition_Carbohydrate"]

protein_intake = feed * protein_ratio
carb_intake = feed * carb_ratio

# -----------------------------
# 2단 오른쪽 - 질병 분류, THI, 사료 섭취량
# -----------------------------

# 질병 색상
if disease == "Healthy":
    disease_color = "#66C56C"
else:
    disease_color = "#E74C3C"

# THI 색상
if thi < 72:
    thi_color = "#E74C3C"
elif thi < 78:
    thi_color = "#F4D03F"
else:
    thi_color = "#66C56C"


# -----------------------------
# 카드 CSS
# -----------------------------

st.markdown("""
<style>

.card {
    background-color:#f5f5f5;
    border-radius:15px;
    padding:20px;
    text-align:center;
    font-weight:600;
    border:4px solid #777;
}

.card_green {
    border-radius:15px;
    padding:20px;
    text-align:center;
    font-weight:700;
    border:4px solid #3FA34D;
}

.card_large {
    background-color:#f5f5f5;
    border-radius:15px;
    padding:25px;
    text-align:center;
    font-weight:600;
    border:4px solid #777;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# 레이아웃
# -----------------------------

left, center, right = st.columns([1.1,1,1.3])

# -----------------------------
# 왼쪽 KPI 시각화
# -----------------------------

with left:

    st.markdown(
        f'<div class="card">목표 생산량: {target_yield} L</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="card">현재 생산량: {current_yield:.1f} L</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="card">전주 평균 생산량: {last_week_yield:.1f} L</div>',
        unsafe_allow_html=True
    )

# -----------------------------
# 가운데 목표치 달성률 시각화
# -----------------------------

with center:

    st.markdown("### 목표치 달성률")
    import plotly.graph_objects as go
    def draw_goal_achievement_chart(current, target):

        # -----------------------------
        # 달성률 계산
        # -----------------------------
        ratio = current / target
        ratio = max(0, min(ratio, 1))   # 0~1 사이 제한
        percent = ratio * 100

        # -----------------------------
        # 270도 차트 + 90도 여백
        # -----------------------------
        achieved = ratio
        remaining = 1 - ratio

        values = [
            achieved,   # 달성
            remaining,  # 미달성
            0.3333      # 90도 여백
        ]

        colors = [
            "#2E7D32",          # 초록 (달성)
            "#D9D9D9",          # 연회색 (미달성)
            "rgba(0,0,0,0)"     # 투명 (여백)
        ]

        fig = go.Figure()

        fig.add_trace(go.Pie(
            values=values,
            hole=0.75,
            marker=dict(colors=colors),
            rotation=225,              # ⭐ 하단 중앙 여백
            direction="clockwise",
            sort=False,
            textinfo="none",
            hoverinfo="skip",
            showlegend=False
        ))

        # -----------------------------
        # 중앙 퍼센트 표시
        # -----------------------------
        fig.add_annotation(
            text=f"<b>{percent:.0f}%</b>",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=30, color="black")
        )

        # -----------------------------
        # 0% / 100% 라벨
        # -----------------------------
        # fig.add_annotation(
        #     text="0%",
        #     x=0.08,
        #     y=0.18,
        #     showarrow=False,
        #     font=dict(size=16)
        # )

        # fig.add_annotation(
        #     text="100%",
        #     x=0.92,
        #     y=0.18,
        #     showarrow=False,
        #     font=dict(size=16)
        # )

        # -----------------------------
        # 레이아웃
        # -----------------------------
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        return fig
    
    st.plotly_chart(
    draw_goal_achievement_chart(current_yield, target_yield),
    use_container_width=True
)
    
    
# -----------------------------
# 오른쪽 KPI 시각화 내용
# -----------------------------

with right:

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f'<div style="background:{disease_color}; padding:18px; border-radius:12px; text-align:center; font-weight:700;">질병 분류: {disease}</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f'<div style="background:{thi_color}; padding:18px; border-radius:12px; text-align:center; font-weight:700;">THI: {thi:.0f}점</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f'<div class="card_large">사료 섭취량: {feed:.1f} kg</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    with c3:
        st.markdown(
            f'<div class="card">단백질 섭취량<br>{protein_intake:.2f} kg</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f'<div class="card">탄수화물 섭취량<br>{carb_intake:.2f} kg</div>',
            unsafe_allow_html=True
        )

# ---------------------------
# 3단 CCTV 움짤
# ---------------------------
st.markdown("---")
st.markdown("### 📹 CCTV 모니터링")

st.caption("현재 축사 내부 실시간 모니터링 영상")

st.image("__CCTV_.gif", use_container_width=True)

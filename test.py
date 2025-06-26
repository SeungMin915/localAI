# HEART Insight AI - Streamlit 기반 미래 모빌리티 트렌드 분석 솔루션

import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
import json
from datetime import datetime

# 데이터베이스 연결
def get_connection():
    conn = sqlite3.connect("heart_insight_ai.db")
    return conn

# 트렌드 데이터 로드 함수
def load_trend_data():
    conn = get_connection()
    query = "SELECT * FROM trends_data ORDER BY publish_date DESC LIMIT 100"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# AI 분석 결과 로드 함수
def load_ai_analysis():
    conn = get_connection()
    query = """
        SELECT a.*, t.title, t.category, t.publish_date 
        FROM ai_analysis_results a
        JOIN trends_data t ON a.trend_id = t.id
        ORDER BY a.analysis_date DESC
        LIMIT 100
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 보고서 JSON -> 시각화용 DataFrame 변환
def forecast_json_to_df(json_str):
    try:
        data = json.loads(json_str)
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

# Streamlit 앱 시작
st.set_page_config(page_title="HEART Insight AI", layout="wide")
st.title("🚘 미래 모빌리티 트렌드 분석 대시보드")

# 탭 설정
tabs = st.tabs(["트렌드 데이터", "AI 분석 결과", "시계열 예측", "보고서 다운로드"])

# 트렌드 데이터 탭
with tabs[0]:
    st.subheader("📄 최근 수집된 모빌리티 트렌드 데이터")
    trend_df = load_trend_data()
    st.dataframe(trend_df[['publish_date', 'category', 'title', 'source', 'keywords']], use_container_width=True)

# AI 분석 결과 탭
with tabs[1]:
    st.subheader("🧠 AI 분석을 통한 보험 시사점")
    analysis_df = load_ai_analysis()
    selected_category = st.selectbox("카테고리 필터", ["전체"] + sorted(analysis_df['category'].unique().tolist()))
    if selected_category != "전체":
        analysis_df = analysis_df[analysis_df['category'] == selected_category]
    st.dataframe(analysis_df[['analysis_date', 'identified_trend', 'predicted_impact', 'risk_level', 'opportunity_level']], use_container_width=True)

# 시계열 예측 탭
with tabs[2]:
    st.subheader("📈 트렌드별 예측 시각화")
    options = analysis_df.dropna(subset=['forecast_data'])[['identified_trend', 'forecast_data']]
    if len(options) > 0:
        selected_trend = st.selectbox("트렌드 선택", options['identified_trend'].tolist())
        selected_json = options[options['identified_trend'] == selected_trend]['forecast_data'].values[0]
        forecast_df = forecast_json_to_df(selected_json)
        if not forecast_df.empty:
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            fig = px.line(forecast_df, x='date', y='value', title=f"'{selected_trend}'의 향후 예측")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("예측 데이터가 없습니다.")
    else:
        st.warning("예측 가능한 데이터가 없습니다.")

# 보고서 다운로드 탭
with tabs[3]:
    st.subheader("📄 보고서 다운로드 (PDF/Word) - 준비 중")
    st.info("보고서 자동 생성 기능은 추후 제공될 예정입니다.")

# 하단 주석
st.caption("HEART Insight AI - 현대해상을 위한 미래 모빌리티 트렌드 분석 솔루션 v0.1")

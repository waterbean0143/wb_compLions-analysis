import streamlit as st
import pandas as pd
import os
import requests
import tempfile
import re
import json
import difflib
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_upstage import UpstageGroundednessCheck

# 환경변수 로드
load_dotenv()

# 사용자 인증 정보
users = {
    "admin": {"password": "admin", "name": "관리자"},
    "test": {"password": "test", "name": "테스트 사용자"},
}

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """UTF-8-sig 인코딩된 CSV를 읽어옵니다."""
    return pd.read_csv(path, dtype=str, encoding='utf-8-sig')

@st.cache_resource
def download_to_temp(url: str) -> str:
    r = requests.get(url)
    r.raise_for_status()
    suffix = '.pdf' if url.lower().endswith('pdf') else '.docx'
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'wb') as f:
        f.write(r.content)
    return tmp_path


def normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())

def find_matching_majors(query: str, majors: list[str]) -> list[str]:
    ni = normalize(query)
    return [m for m in majors if normalize(m).startswith(ni) or ni in normalize(m)]


def display_major_detail(df: pd.DataFrame, major_label: str):
    """
    주어진 major_label에 대해 타이밍, 책임자, 실무자, 지원부서, 시스템 정보를 표시합니다.
    """
    sub = df[df['major'] == major_label]
    cols = ['timing', 'owner', 'worker', 'support', 'system']
    available = [c for c in cols if c in sub.columns]
    if not sub.empty and available:
        st.table(sub[available].rename(columns={
            'timing': '시기',
            'owner': '책임자',
            'worker': '실무자',
            'support': '협조 및 지원 부서',
            'system': '적용 시스템'
        }))
    else:
        st.write("세부 정보가 없습니다.")


def main():
    st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")

    # --- 로그인 로직 ---
    if not st.session_state.get('logged_in', False):
        st.header("📋 SI 프로세스 챗봇 로그인")
        username = st.text_input("아이디", key="login_user")
        password = st.text_input("비밀번호", type="password", key="login_pw")
        if st.button("로그인", key="login_button"):
            if username in users and password == users[username]["password"]:
                st.session_state['logged_in'] = True
                st.session_state['logged_in_user'] = username
                st.experimental_rerun()
            else:
                st.error("로그인 정보가 올바르지 않습니다.")
        st.stop()

    # 로그인 완료 후 환영 이름
    logged_in_user = st.session_state['logged_in_user']
    display_name = users[logged_in_user]['name']

    # 단계별 상세정보 CSV 로드 및 컬럼명 변환
    df = load_csv("SI_FULL_PROCESS_HIERARCHY.CSV")
    rename_map = {
        '주요 단계': 'step_name',
        '주요 활동': 'major',
        '시기': 'timing',
        '책임자': 'owner',
        '실무자': 'worker',
        '협조 및 지원 부서': 'support',
        '적용 시스템': 'system',
        '문서 URL': 'document_url'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if 'document_url' not in df.columns:
        df['document_url'] = ''

    # configs: 단계명과 문서 URL 목록
    configs = []
    for step in df['step_name'].dropna().unique():
        urls = df.loc[df['step_name'] == step, 'document_url'].dropna().unique()
        configs.append({'step_name': step, 'document_url': urls[0] if len(urls) else ''})

    # 대표 Q&A JSON 로드
    try:
        with open("vdc_a_대표질문.json", encoding='utf-8') as f:
            rep_qna = json.load(f)
    except FileNotFoundError:
        rep_qna = []

    # --- 사이드바 ---
    st.sidebar.header(f"[접속자] {display_name}님, 환영합니다!")
    if st.sidebar.button("새로운 대화 주제", key="clear_button"):
        st.session_state.clear()
        st.experimental_rerun()

    model_name = st.sidebar.selectbox("언어 모델 선택", ["o3-mini"], key="model_select")
    st.session_state['answer_mode'] = st.sidebar.selectbox(
        "답변 모드 선택", ["빠른 답변", "정확한 답변"], index=0, key="answer_mode_select"
    )
    st.session_state['reasoning_effort'] = st.sidebar.selectbox(
        "추론 수준 선택", ["low", "medium", "high"], index=1, key="reasoning_effort_select"
    )

    st.sidebar.header("피드백")
    feedback = st.sidebar.text_area("전반적인 사용후기를 입력해주세요:", key="feedback_text")
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        if st.sidebar.button("피드백 제출", key="feedback_submit"):
            # 피드백 저장 로직 추가 필요
            st.sidebar.success("피드백이 제출되었습니다.")
    with col2:
        if st.sidebar.button("로그아웃", key="logout_button"):
            st.session_state.clear()
            st.experimental_rerun()

    st.sidebar.write("🐧 저작자: @AI이행봇")

    # --- 메인 탭 구성 ---
    intro_tab, overview_tab, qa_tab = st.tabs(["소개", "절차 개요", "Q&A"])

    with intro_tab:
        st.markdown("## SI 전체 프로세스 챗봇에 오신 것을 환영합니다.")
        st.write("좌측에서 단계 선택 후 개요·Q&A 이용")

    with overview_tab:
        step = st.sidebar.selectbox("프로세스 단계 선택", [c['step_name'] for c in configs], key="step_select")
        st.header(f"{step} 단계 상세 ({display_name}님)")
        df_step = df[df['step_name'] == step]
        for major_label in df_step['major'].dropna().unique():
            with st.expander(major_label):
                display_major_detail(df_step, major_label)

    with qa_tab:
        st.header("Q&A")
        query = st.text_input("질문 입력")
        if query:
            # 대표 Q&A 또는 문서 기반 Q&A 로직
            pass

if __name__ == "__main__":
    main()

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
from rank_bm25 import BM25Okapi

# 환경변수 로드
load_dotenv()

# 사용자 인증 정보
users = {
    "admin": {"password": "admin", "name": "관리자"},
    "test": {"password": "test", "name": "테스트 사용자"},
}

# 로그인 및 권한 확인 함수 정의 (이제 호출 이전에 정의됩니다)
def check_password():
    def password_entered():
        if (
            st.session_state["username"] in users
            and st.session_state["password"] == users[st.session_state["username"]]["password"]
        ):
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = users[st.session_state["username"]]["name"]
            st.session_state["is_admin"] = st.session_state["username"] == "admin"
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.markdown("---")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("😕 알 수 없는 사용자이거나 비밀번호가 틀립니다.")
        return False
    else:
        return True

# CSV 로드 유틸리티
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, encoding='utf-8-sig')

# 프로젝트 프로세스 CSV 경로
CSV_PATH = os.path.join(os.path.dirname(__file__), "SI_FULL_PROCESS_HIERARCHY.csv")

# 메인 앱

def main():
    # 페이지 구성
    st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")

    # 로그인 처리
    if not check_password():
        return

    # 로그인 성공 후 UI
    display_name = st.session_state.get("logged_in_user", "Unknown")
    st.sidebar.header(f"[접속자] {display_name}님, 환영합니다!")

    # 로그아웃 버튼
    if st.sidebar.button("로그아웃"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

    # 탭 생성
    tab_intro, tab_overview, tab_qa = st.tabs(["소개", "절차 개요", "Q&A"])

    # 소개 탭
    with tab_intro:
        st.header("📋 SI 프로세스 챗봇")
        st.write("사이드바에서 로그인 후, 절차 개요 탭에서 프로세스를 확인하거나 Q&A 탭에서 질문하세요.")

    # 절차 개요 탭
    with tab_overview:
        st.header("절차 개요")
        if not os.path.isfile(CSV_PATH):
            st.error(f"CSV 파일을 찾을 수 없습니다: {CSV_PATH}")
            return
        df = load_csv(CSV_PATH)
        # 주요 단계별로 보여주기
        for step in df['step_name'].unique():
            st.subheader(step)
            sub = df[df['step_name'] == step][['major','timing','owner','worker','support','system']]
            sub = sub.rename(columns={
                'major':'주요 활동','timing':'시기','owner':'책임자',
                'worker':'실무자','support':'협조 및 지원 부서','system':'적용 시스템'
            })
            st.table(sub)

    # Q&A 탭
    with tab_qa:
        st.header("Q&A")
        query = st.text_input("질문 입력")
        if st.button("질문하기") and query:
            # 간단 예: echo
            st.markdown(f"**답변:** {query} 에 대한 답변을 구현하세요.")


if __name__ == "__main__":
    main()

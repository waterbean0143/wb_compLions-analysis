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

# 1) 페이지 설정은 여기 딱 한 번만!
st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")

load_dotenv()

# --- CSV 로딩 헬퍼 ---
@st.cache_data
def load_hierarchy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding='utf-8-sig')
    # 컬럼명 통일
    rename_map = {
        '주요 단계':'step_name','주요 활동':'major','시기':'timing',
        '책임자':'owner','실무자':'worker','협조 및 지원 부서':'support',
        '적용 시스템':'system'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    return df

# --- 사용자 인증 (기존 코드) ---
users = {
    "admin": {"password": "admin", "name": "관리자"},
    "test": {"password": "test", "name": "테스트 사용자"},
}
def check_password():
    # ... (기존 로그인 로직 그대로)
    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.stop()
    elif not st.session_state["password_correct"]:
        st.error("😕 알 수 없는 사용자이거나 비밀번호가 틀립니다.")
        st.stop()
    else:
        return True

# 로그인 처리
if not check_password():
    st.stop()

# 로그인 성공한 사용자 이름
display_name = st.session_state.get("logged_in_user", "Unknown")

# 사이드바
st.sidebar.header(f"[접속자] {display_name}님, 환영합니다!")
if st.sidebar.button("새로운 대화 주제"):
    st.session_state.clear()
    st.experimental_rerun()

# --- 탭 생성: Overview vs Q&A ---
overview_tab, qa_tab = st.tabs(["Overview","Q&A"])

# 2) Overview 탭: CSV 기반 접이식 테이블
with overview_tab:
    st.header("절차 개요")
    csv_path = os.path.join(os.path.dirname(__file__), "SI_FULL_PROCESS_HIERARCHY.csv")
    df = load_hierarchy(csv_path)
    # "step_name"별 그룹
    for step, grp in df.groupby("step_name"):
        with st.expander(step, expanded=False):
            # 테이블 컬럼만 뽑아서 표시
            st.table(
                grp[["major","timing","owner","worker","support","system"]]
                .reset_index(drop=True)
            )

# 3) Q&A 탭: 기존 SIBOT Q&A 로직 안으로 이동
with qa_tab:
    st.header("Q&A")
    # CSV 로드가 아닌 Q&A 문서 로직
    # 예시: 간단 RetrievalQA
    query = st.text_input("질문 입력", key="qna_input")
    if st.button("질문하기", key="qna_button") and query:
        # 여기에 기존 rep_qna, BM25, LLM chain 등 삽입
        st.write("답변 생성 중…")
        # 예시 응답
        st.write("여기에 문서 기반 답변이 출력됩니다.")

    # 2) RetrievalQA on process docs
    prompt = PromptTemplate(
        input_variables=["context","question"],
        template="""
        당신은 SI 프로세스 문서 기반 질문에 답하는 어시스턴트입니다.

        [문서]
        {context}

        [질문]
        {question}

        [답변 형식]
        💡 핵심 요약
        📋 절차 또는 판단 주체
        """
    )
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=process_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt":prompt},
        return_source_documents=True
    )
    res = qa.invoke({"query":query})
    st.markdown(f"**💡 핵심 요약**\n{res['result']}")
    with st.expander("📎 문서 근거"):
        for i,doc in enumerate(res["source_documents"], start=1):
            st.write(f"[{i}] {doc.metadata.get('source_name','unknown')}")
            st.code(doc.page_content[:200] + "…")

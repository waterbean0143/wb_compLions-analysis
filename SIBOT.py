import streamlit as st
import requests
import tempfile
import os
import pandas as pd
import re
from datetime import datetime, timezone, timedelta
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from io import BytesIO
from kiwipiepy import Kiwi
from langgraph.graph import END, StateGraph
from langchain_upstage import UpstageGroundednessCheck
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.document_transformers import LongContextReorder
from sklearn.metrics.pairwise import cosine_similarity
from typing import TypedDict, Dict, List, Tuple
import uuid
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
import threading
import openai

# 글로벌 변수 선언
global memory
global si_qna_vectordbs
global si_qna_docs
global for_show_si_process_vectordbs
global si_process_docs
global si_process_vectordbs
global similar_cases_db
global fcpa_retrievers

# 페이지 설정
st.set_page_config(page_title="AX SI 방법론 이행봇", page_icon="🤖")

# -----------------------------
# 1) 사용자 정보 및 로그인
# -----------------------------
users = {
    "10154371": {"password": "10154371", "name": "배수빈"},
    "10154372": {"password": "10154372", "name": "김도완"},
    "10156350": {"password": "10156350", "name": "박영준"},
}

if 'logged_in' not in st.session_state:
    st.sidebar.title("🔒 로그인")
    uid = st.sidebar.text_input("ID", key="login_id")
    pwd = st.sidebar.text_input("PW", type="password", key="login_pw")
    if st.sidebar.button("로그인"):
        if uid in users and users[uid]['password'] == pwd:
            st.session_state['logged_in'] = True
            st.session_state['user_id'] = uid
            st.sidebar.success(f"환영합니다, {users[uid]['name']}님!")
        else:
            st.sidebar.error("ID 또는 비밀번호가 올바르지 않습니다.")
    st.stop()

# -----------------------------
# 2) UI: 답변 모드 및 탭
# -----------------------------
st.sidebar.title("⚙️ 설정")
answer_mode = st.sidebar.radio("답변 모드 선택", ['빠른 답변', '정확한 답변'], index=0)

tabs = st.tabs(["Q&A", "Feedback", "사례관리"])
qa_tab, fb_tab, case_tab = tabs

# -----------------------------
# 3) 외부 파일 경로 및 로드 설정
# -----------------------------
# CSV: SI 프로세스 목록
csv_file_id = "1gzu8FnjAR2x99M-xQiaNaQevIbFo9rXl"
csv_download_url = f"https://drive.google.com/uc?export=download&id={csv_file_id}"
si_full_process = "SI_FULL_PROCESS_HIERARCHY.csv"

# QnA JSON: 절차별 주요 질의응답
QNA_JSON_PATH = "si_qna.json"

@st.cache_data
def load_data():
    # CSV 다운로드
    gdown.download(csv_download_url, si_full_process, quiet=True)
    df = pd.read_csv(si_full_process, encoding="utf-8-sig")
    # QnA JSON 로드
    try:
        with open(QNA_JSON_PATH, 'r', encoding='utf-8') as f:
            si_qna = json.load(f)
    except FileNotFoundError:
        si_qna = {}
    return df, si_qna

df, si_qna = load_data()

# si_process: 회사 내부 SI 절차 종류 리스트\ nsi_process = df['주요 단계'].unique().tolist()

# -----------------------------
# 4) Q&A 탭
# -----------------------------
with qa_tab:
    st.header("AX SI 방법론 이행봇")
    st.subheader("📋 전체 SI 프로세스 목록")
    st.dataframe(df)

    # 절차 선택
    selected_stage = st.selectbox("📂 절차 단계 선택", si_process)
    st.markdown(f"**선택된 절차:** {selected_stage}")

    # 관련 PDF 링크 (필요 시 추가)
    procedure_pdf_urls = {
        "사전영업": ["https://drive.google.com/uc?export=download&id=PRE_SALES_PDF_ID"],
        "VDC-A 발의": ["https://drive.google.com/uc?export=download&id=VDC_A_PDF_ID"],
    }
    urls = procedure_pdf_urls.get(selected_stage, [])
    if urls:
        st.write("관련 문서:")
        for url in urls:
            st.markdown(f"- [PDF 문서]({url})")
    else:
        st.info("해당 절차에 등록된 문서가 없습니다.")

    # 사전 정의된 Q&A 예시 표시
    if selected_stage in si_qna:
        st.write("🔍 사전 정의된 Q&A 예시:")
        for qa_item in si_qna[selected_stage]:
            st.markdown(f"- **Q:** {qa_item['question']}  
                      **A:** {qa_item['answer']}")

    # CSV를 Document 리스트로 변환
    docs = []
    for _, row in df[df['주요 단계'] == selected_stage].iterrows():
        content = "\n".join(f"{col} : {row[col]}" for col in df.columns)
        metadata = row.to_dict()
        docs.append(Document(page_content=content, metadata=metadata))

    # RetrievalQA 초기화 (캐시)
    @st.cache_resource
    def init_qa(docs):
        emb = OpenAIEmbeddings()
        vs = FAISS.from_documents(docs, emb)
        return RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            chain_type="stuff",
            retriever=vs.as_retriever()
        )

    qa_chain = init_qa(docs)

    # 사용자 질문
    query = st.text_input("💬 질문을 입력하세요", key="proc_query")
    if query:
        with st.spinner("답변 생성 중…"):
            answer = qa_chain.run(query)
        st.markdown("**답변:**")
        st.write(answer)

# -----------------------------
# 5) Feedback 탭
# -----------------------------
with fb_tab:
    st.header("📝 Feedback")
    st.write("Feedback 기능은 추후 구현 예정입니다.")

# -----------------------------
# 6) 사례관리 탭
# -----------------------------
with case_tab:
    st.header("📂 사례관리 (SI Q&A)")
    st.write("사례관리 기능은 추후 구현 예정입니다.")

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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
import threading
import openai

from langchain.chains import RetrievalQA 

# ─────────────────────────────────────────────────────
# 1) 페이지 설정 및 Secrets 로드
# ─────────────────────────────────────────────────────
st.set_page_config(page_title="AX SI 방법론 이행봇", page_icon="🤖", layout="wide")
os.environ["OPENAI_API_KEY"]      = st.secrets["openai"]["api_key"]
os.environ["UPSTAGE_API_KEY"]     = st.secrets["upstage"]["api_key"]
os.environ["LANGCHAIN_API_KEY"]   = st.secrets["langchain"]["api_key"]
os.environ["LANGCHAIN_ENDPOINT"]  = st.secrets["langchain"]["endpoint"]
os.environ["LANGCHAIN_PROJECT"]   = st.secrets["langchain"]["project"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["langchain"]["tracing_v2"]
os.environ["LANGSMITH_API_KEY"]   = st.secrets.get("langsmith", {}).get("api_key", "")

# ─────────────────────────────────────────────────────
# 2) 글로벌 설정
# ─────────────────────────────────────────────────────
proc_docs = []
proc_vectordbs = {}
qna_vectordbs = {}
case_docs = []
for_show_proc_vectordbs = {}
selected_for_show_proc_vectordbs = {}
proc_retrievers = {}
qna_retrievers = {}


executor = ThreadPoolExecutor(max_workers=5)
bm25_weight = 0.3
faiss_weight = 0.7
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────────────
# 3) 로그인
# ─────────────────────────────────────────────────────
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
            st.session_state['user_id']   = uid
            st.sidebar.success(f"환영합니다, {users[uid]['name']}님!")
        else:
            st.sidebar.error("ID 또는 비밀번호가 올바르지 않습니다.")
    st.stop()

# ─────────────────────────────────────────────────────
# 4) UI 설정
# ─────────────────────────────────────────────────────
st.sidebar.title("⚙️ 설정")
answer_mode = st.sidebar.radio("답변 모드 선택", ['빠른 답변', '정확한 답변'], index=0)

tabs = st.tabs(["Q&A", "Feedback", "사례관리"])
qa_tab, fb_tab, case_tab = tabs

# ─────────────────────────────────────────────────────
# 5) PDF 매핑
# ─────────────────────────────────────────────────────
PROCESS_PDF_URLS = {
    "제안/계약": "https://drive.google.com/uc?export=download&id=1TNOhmUds7hMpwz3NO4QD-mO-J1sUJoEa",
}
QNA_PDF_URLS = {
    "제안/계약": "https://drive.google.com/uc?export=download&id=17M1mnMZVl29EahbSVqzcyZEX8LYsx5ER",
}

# ─────────────────────────────────────────────────────
# 6) 전처리 함수
# ─────────────────────────────────────────────────────
kiwi = Kiwi()
def preprocess(text: str) -> str:
    toks = kiwi.analyze(text)[0][0]
    return ' '.join(t.form for t in toks if t.tag.startswith(('N','V','MA')))

# ─────────────────────────────────────────────────────
# 7) 문서 로드 & 벡터DB 생성 (캐시 활용)
# ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600*24)
def cached_load_all_docs() -> Tuple[Dict[str, List[Document]], Dict[str, List[Document]]]:
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    proc_map: Dict[str, List[Document]] = {}
    qna_map: Dict[str, List[Document]] = {}

    # 프로세스 문서 로드
    for step, url in PROCESS_PDF_URLS.items():
        docs = splitter.split_documents(PyMuPDFLoader(url).load())
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata['step'] = step
        proc_map[step] = docs

    # QnA 문서 로드
    for step, url in QNA_PDF_URLS.items():
        docs = splitter.split_documents(PyMuPDFLoader(url).load())
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata['step'] = step
        qna_map[step] = docs

    return proc_map, qna_map

@st.cache_resource(ttl=3600*24)
def cached_build_vectordbs(
    proc_docs_map: Dict[str, List[Document]],
    qna_docs_map:  Dict[str, List[Document]]
) -> Tuple[Dict[str, FAISS], Dict[str, FAISS]]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    proc_vdb = {
        step: FAISS.from_documents(docs, emb)
        for step, docs in proc_docs_map.items()
    }
    qna_vdb = {
        step: FAISS.from_documents(docs, emb)
        for step, docs in qna_docs_map.items()
    }
    return proc_vdb, qna_vdb

# 캐시 워밍업 (앱 실행 시 한 번만)
with st.spinner("초기 데이터 준비 중…"):
    proc_docs_map, qna_docs_map = cached_load_all_docs()
    proc_vectordbs, qna_vectordbs = cached_build_vectordbs(proc_docs_map, qna_docs_map)

# ─────────────────────────────────────────────────────
# 8) 탭 UI 정의
# ─────────────────────────────────────────────────────
tabs = st.tabs(["Q&A", "Feedback", "사례관리"])
qa_tab, fb_tab, case_tab = tabs

# ─────────────────────────────────────────────────────
# 9) Q&A 탭 로직
# ─────────────────────────────────────────────────────
with qa_tab:
    st.header("AX SI 방법론 이행봇")
    st.subheader("📂 절차 선택 및 질의응답")

    # 단계 선택
    step = st.selectbox(
        "📂 절차 단계 선택",
        options=list(PROCESS_PDF_URLS.keys())
    )

    # PDF 링크
    st.markdown(f"- [프로세스 PDF]({PROCESS_PDF_URLS[step]})")
    st.markdown(f"- [Q&A PDF]({QNA_PDF_URLS[step]})")

    # 리트리버 구성
    proc_retr = proc_vectordbs[step].as_retriever()
    qna_retr  = qna_vectordbs[step].as_retriever()
    ensemble = EnsembleRetriever(
        retrievers=[qna_retr, proc_retr],
        weights=[bm25_weight, faiss_weight]
    )

    # RetrievalQA 체인 준비
    qa_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=qa_llm,
        chain_type="stuff",
        retriever=ensemble,
        return_source_documents=False
    )

    # 질문 입력 및 답변
    query = st.text_input("💬 질문을 입력하세요", key="proc_query")
    if st.button("질문 요청"):
        if not query.strip():
            st.warning("먼저 질문을 입력해주세요.")
        else:
            with st.spinner("답변 생성 중…"):
                answer = qa_chain.run(query)
            st.markdown("**답변:**")
            st.write(answer)

# ─────────────────────────────────────────────────────
# 10) Feedback 탭
# ─────────────────────────────────────────────────────
with fb_tab:
    st.header("📝 Feedback")
    st.write("Feedback 기능은 추후 구현 예정입니다.")

# ─────────────────────────────────────────────────────
# 11) 사례관리 탭
# ─────────────────────────────────────────────────────
with case_tab:
    st.header("📂 사례관리")
    st.write("사례관리 기능은 추후 구현 예정입니다.")

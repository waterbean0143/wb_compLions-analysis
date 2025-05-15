# ─────────────────────────────────────────────────────
# 0) 라이브러리 및 선언 영역
# ─────────────────────────────────────────────────────

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
# 0-1) PDF 첫페이지 인덱스 자동추출 유틸 (제안/계약 전용)
# ─────────────────────────────────────────────────────
def download_and_load(url: str) -> List[Document]:
    resp = requests.get(url)  
    resp.raise_for_status()  
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf: 
        tf.write(resp.content) 
        tmp_path = tf.name  
    try:  
        docs = PyMuPDFLoader(tmp_path).load() 
    except Exception: 
        docs = []  
    finally:  
        try:  
            os.remove(tmp_path) 
        except OSError:  #추가
            pass  #추가
    return docs  #추가

def extract_index_chunks(url: str) -> List[Document]:  
    raw_docs = download_and_load(url)  
    if not raw_docs: 
        return [] 
    first_page = raw_docs[0].page_content 
    lines = first_page.splitlines()  
    # '##' 헤딩 다음 줄부터 인덱스 시작
    start = 0  
    for i, line in enumerate(lines): 
        if line.strip().startswith("##"): 
            start = i + 1  
            break  
    pattern = re.compile(r"^(\d+)\.\s*(.+)$")  
    index_docs: List[Document] = []  
    for line in lines[start:]: 
        m = pattern.match(line.strip()) 
        if not m: 
            break  
        num, title = m.groups()  
        meta = {"step": int(num), "title": title}  
        text = f"{num}. {title}"  
        index_docs.append(Document(page_content=text, metadata=meta))  
    return index_docs  #추가

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
# 4) PDF 매핑
# ─────────────────────────────────────────────────────
PROCESS_PDF_URLS = {
    "제안/계약": "https://drive.google.com/uc?export=download&id=1TNOhmUds7hMpwz3NO4QD-mO-J1sUJoEa",
    "착수/계획": "https://drive.google.com/uc?export=download&id=16j9ypXkWD7oi477ylSXWhVVe7jLtRuI7",
}
QNA_PDF_URLS = {
    "제안/계약": "https://drive.google.com/uc?export=download&id=17M1mnMZVl29EahbSVqzcyZEX8LYsx5ER",
}

# ─────────────────────────────────────────────────────
# 5) UI 설정
# ─────────────────────────────────────────────────────
st.sidebar.title("⚙️ 설정")
answer_mode = st.sidebar.radio("답변 모드 선택", ['빠른 답변', '정확한 답변'], index=0)

selected_steps = st.sidebar.multiselect(
    "⚙️ 절차 단계 선택",
    options=list(PROCESS_PDF_URLS.keys()),
    default=list(PROCESS_PDF_URLS.keys())[:1]
)
if not selected_steps:
    st.sidebar.warning("하나 이상의 절차 단계를 선택하세요.")
    st.stop()
    
tabs = st.tabs([" Q&A ", " (추가예정) "])
qa_tab, else_tab = tabs


# ─────────────────────────────────────────────────────
# 6) 전처리 함수
# ─────────────────────────────────────────────────────
kiwi = Kiwi()
def preprocess(text: str) -> str:
    toks = kiwi.analyze(text)[0][0]
    return ' '.join(t.form for t in toks if t.tag.startswith(('N','V','MA')))

# ─────────────────────────────────────────────────────
# 7) 문서 로드 & 벡터DB 생성 (별도 저장)
# ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600*24)
def load_all_docs() -> Tuple[Dict[str, List[Document]], Dict[str, List[Document]]]:
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    proc_map: Dict[str, List[Document]] = {}
    qna_map:  Dict[str, List[Document]] = {}

    def download_and_load(url: str, step: str) -> List[Document]:
        # 1) PDF 바이너리 다운로드
        resp = requests.get(url)
        resp.raise_for_status()
        # 2) 임시 파일에 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(resp.content)
            tf_path = tf.name
        # 3) PyMuPDFLoader로 로드
        docs = PyMuPDFLoader(tf_path).load()
        # 4) 로드 후 임시 파일 삭제
        try:
            os.remove(tf_path)
        except OSError:
            pass
        return docs

    # 1) 프로세스 문서
    for step, url in PROCESS_PDF_URLS.items():
        raw_docs = download_and_load(url, step)
        chunks = splitter.split_documents(raw_docs)
        for d in chunks:
            d.page_content = preprocess(d.page_content)
            d.metadata['step'] = step
        proc_map[step] = chunks

    # 2) QnA 문서
    for step, url in QNA_PDF_URLS.items():
        raw_docs = download_and_load(url, step)
        chunks = splitter.split_documents(raw_docs)
        for d in chunks:
            d.page_content = preprocess(d.page_content)
            d.metadata['step'] = step
        qna_map[step] = chunks

    return proc_map, qna_map


def build_vectordbs(
    proc_docs_map: Dict[str, List[Document]],
    qna_docs_map:  Dict[str, List[Document]]
) -> Tuple[Dict[str, FAISS], Dict[str, FAISS]]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    proc_vdb: Dict[str, FAISS] = {
        step: FAISS.from_documents(docs, emb)
        for step, docs in proc_docs_map.items()
    }
    qna_vdb: Dict[str, FAISS] = {
        step: FAISS.from_documents(docs, emb)
        for step, docs in qna_docs_map.items()
    }
    return proc_vdb, qna_vdb

index_docs = extract_index_chunks(PROCESS_PDF_URLS["제안/계약"])  #추가
st.write("🔍 제안/계약 INDEX:", [d.metadata for d in index_docs])  #추가

# 앱 시작 시 한 번만 실행
with st.spinner("📦 데이터 로드 중…"):
    proc_docs_map, qna_docs_map   = load_all_docs()
    proc_vectordbs, qna_vectordbs = build_vectordbs(proc_docs_map, qna_docs_map)

# ─────────────────────────────────────────────────────
# 8) Q&A 탭
# ─────────────────────────────────────────────────────
# Q&A 탭
with qa_tab:
    # (1) 단계 선택 확인
    st.write("▶️ 선택된 단계:", selected_steps)

    # (2) retriever 리스트 생성
    retrievers = []
    weights    = []
    for step in selected_steps:
        # proc + qna 두 개를 묶거나, BM25/FAISS 가중치를 달리할 수도 있고…
        retrievers.append(proc_vectordbs[step].as_retriever())
        retrievers.append(qna_vectordbs.get(step, []).as_retriever())
        weights.extend([faiss_weight, bm25_weight])

    ensemble = EnsembleRetriever(retrievers=retrievers, weights=weights)

    # (3) 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        chain_type="stuff",
        retriever=ensemble
    )

    # (4) 질문 & 답변
    query = st.text_input("💬 질문을 입력하세요")
    if st.button("질문 요청") and query:
        with st.spinner("⏳ 답변 생성 중…"):
            answer = qa_chain.run(query)
        st.markdown("**답변:**")
        st.write(answer)

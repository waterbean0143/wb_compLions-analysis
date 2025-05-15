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
        except OSError:
            pass
    return docs

def extract_index_chunks(url: str) -> List[Document]:
    raw_docs = download_and_load(url)
    if not raw_docs:
        return []
    first_page = raw_docs[0].page_content
    lines = first_page.splitlines()
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
    return index_docs
    
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
            st.sidebar.success(f"환영합니다, {users[uid]['name']}님!")
            st.experimental_rerun()    # ← 로그인 성공 직후 스크립트를 강제 재실행
        else:
            st.sidebar.error("ID 또는 비밀번호가 올바르지 않습니다.")
    st.stop()  # 로그인 안 된 경우에만 스탑

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

tabs = st.tabs(["Q&A", "추가예정"])
qa_tab, _ = tabs

# ─────────────────────────────────────────────────────
# 6) 전처리 함수
# ─────────────────────────────────────────────────────
kiwi = Kiwi()
def preprocess(text: str) -> str:
    toks = kiwi.analyze(text)[0][0]
    return ' '.join(t.form for t in toks if t.tag.startswith(('N','V','MA')))

# ─────────────────────────────────────────────────────
# 7) 문서 로드 & vectordb 생성
# ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600*24)
def load_all_docs() -> Tuple[Dict[str, List[Document]], Dict[str, List[Document]]]:
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    proc_map, qna_map = {}, {}
    def dl(url):
        docs = download_and_load(url)
        return splitter.split_documents(docs)
    for step, url in PROCESS_PDF_URLS.items():
        docs = dl(url)
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata['step'] = step
        proc_map[step] = docs
    for step, url in QNA_PDF_URLS.items():
        docs = dl(url)
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata['step'] = step
        qna_map[step] = docs
    return proc_map, qna_map

def build_vectordbs(
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

def build_index_retrievers() -> Dict[str, any]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    idx_retrs = {}
    for step, url in PROCESS_PDF_URLS.items():
        idx_docs = extract_index_chunks(url)
        if idx_docs:
            idx_retrs[step] = FAISS.from_documents(idx_docs, emb).as_retriever()
    return idx_retrs

def build_substep_vectordbs(
    proc_docs_map: Dict[str, List[Document]]
) -> Dict[str, Dict[str, FAISS]]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    sub_vdbs: Dict[str, Dict[str, FAISS]] = {}
    for step, docs in proc_docs_map.items():
        idx_docs = extract_index_chunks(PROCESS_PDF_URLS[step])
        sub_db: Dict[str, FAISS] = {}
        for idx in idx_docs:
            title = idx.metadata["title"]
            subset = [d for d in docs if title in d.page_content]
            if subset:
                sub_db[title] = FAISS.from_documents(subset, emb)
        sub_vdbs[step] = sub_db
    return sub_vdbs

# 앱 시작 시 한 번만 로드·벡터화·인덱스 생성
with st.spinner("📦 데이터 로드 중…"):
    proc_docs_map, qna_docs_map   = load_all_docs()
    proc_vectordbs, qna_vectordbs = build_vectordbs(proc_docs_map, qna_docs_map)
    index_retrievers             = build_index_retrievers()
    substep_vectordbs            = build_substep_vectordbs(proc_docs_map)

# ─────────────────────────────────────────────────────
# 8) Q&A 탭
# ─────────────────────────────────────────────────────
with qa_tab:
    st.header("AX SI 방법론 이행봇")

    # 1) 절차 단계 선택
    step = st.selectbox("📂 절차 단계 선택", list(PROCESS_PDF_URLS.keys()))
    if not step:
        st.info("먼저 절차 단계를 선택하세요.")
        st.stop()

    # 2) INDEX 개요
    st.subheader(f"[{step}] 프로세스 개요")
    idx_docs = extract_index_chunks(PROCESS_PDF_URLS[step])
    with st.expander("목록 펼치기", expanded=False):
        for d in idx_docs:
            st.markdown(f"- {d.page_content}")

    # 3) 질문 입력
    query = st.text_input("💬 질문을 입력하세요", key=f"query_{step}")
    if st.button("질문 요청", key=f"btn_{step}"):
        if not query.strip():
            st.warning("질문을 입력해주세요.")
        else:
            # 4) INDEX retriever로 substep 추론
            idx_retr = index_retrievers.get(step)
            top_meta = idx_retr.get_relevant_documents(query)[0].metadata
            sub_title = top_meta["title"]
            st.info(f"📌 이 질문은 ‘{sub_title}’ 단계입니다.")

            # 5) substep vectordb 사용
            retriever = substep_vectordbs.get(step, {}).get(sub_title)
            if retriever is None:
                # fallback to 전체 단계
                retriever = proc_vectordbs[step].as_retriever()

            # 6) 답변 생성
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0,
                               openai_api_key=os.environ["OPENAI_API_KEY"]),
                chain_type="stuff",
                retriever=retriever
            )
            with st.spinner("답변 생성 중…"):
                answer = qa_chain.run(query)
            st.subheader("💡 답변")
            st.write(answer)

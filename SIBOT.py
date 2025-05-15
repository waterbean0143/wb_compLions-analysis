# ─────────────────────────────────────────────────────
# 0) 라이브러리 및 선언 영역
# ─────────────────────────────────────────────────────
import streamlit as st
import requests
import tempfile
import os
import re
from datetime import datetime, timezone, timedelta
from typing import TypedDict, Dict, List, Tuple
from io import BytesIO
import uuid
import time
import threading
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from kiwipiepy import Kiwi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
#from langchain_text_splitters.regex import RegexTextSplitter

# ─────────────────────────────────────────────────────
# 0-1) PDF 첫페이지 인덱스 자동추출 유틸
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
# 0-2) GraphState 정의 & 질문 유형 분류
# ─────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    step_name: str
    sub_title: str
    question_type: str
    context: str
    response: str
    attempts: int

def classify_question_type(q: str) -> str:
    q_lower = q.lower()                    # q_lower 정의
    if any(k in q_lower for k in ["정의", "이란"]):
        return "정의 요청"
    if any(k in q_lower for k in ["어떻게", "절차", "방법"]):
        return "수행 절차 안내"
    if any(k in q_lower for k in ["산출물", "문서", "준비"]):
        return "산출물·문서 요구 사항"
    if any(k in q_lower for k in ["누가", "책임", "역할"]):
        return "책임·역할 분담"
    # q_lower를 사용해서 숫자+일 패턴도 감지
    if re.search(r"\d+일", q_lower) or any(k in q_lower for k in ["언제", "기한", "마감"]):
        return "일정·마일스톤 확인"
    return "일반 질문"

# ─────────────────────────────────────────────────────
# 0-3) 질문유형별 Persona + SystemPrompt
# ─────────────────────────────────────────────────────
QUESTION_TYPE_SYSTEM: Dict[str,str] = {
    "정의 요청": """당신은 대기업 KT의 SI 이행론 전문 PM입니다.
지금 질문은 세부절차 “{sub_title}”의 정의를 묻고 있습니다.
What(무엇인지)과 Why(의의)를 간결히 설명하세요.
""",
    "수행 절차 안내": """당신은 대기업 KT의 SI 이행론 전문 PM입니다.
지금 질문은 세부절차 “{sub_title}”의 수행 방법을 묻고 있습니다.
What/Why/How 순으로 단계별 설명하세요.
""",
    "산출물·문서 요구 사항": """당신은 대기업 KT의 SI 이행론 전문 PM입니다.
지금 질문은 세부절차 “{sub_title}”의 필수 산출물을 묻고 있습니다.
준비해야 할 문서·양식을 목록으로 제시하세요.
""",
    "책임·역할 분담": """당신은 대기업 KT의 SI 이행론 전문 PM입니다.
지금 질문은 세부절차 “{sub_title}”의 책임 주체를 묻고 있습니다.
RACI 형식으로 역할과 책임을 정리하세요.
""",
    "일정·마일스톤 확인": """당신은 대기업 KT의 SI 이행론 전문 PM입니다.
지금 질문은 세부절차 “{sub_title}”의 일정·마감 기한을 묻고 있습니다.
시작일·종료일·N일 이내 요건을 표로 정리하세요.
""",
    "일반 질문": """당신은 대기업 KT의 SI 이행론 전문 PM입니다.
지금 질문은 세부절차 “{sub_title}”에 대한 일반 문의입니다.
관련된 What/Why/How 또는 체크리스트를 간결히 답변하세요.
"""
}

# ─────────────────────────────────────────────────────
# 0-4) STEP별 시스템 메시지 정의 (start/end 동적 반영)
# ─────────────────────────────────────────────────────
STEP_SYSTEM_PROMPTS = {
    "제안/계약": """당신은 AX SI 방법론의 ‘제안/계약’ 단계 전문가입니다.
‘{start_title}’부터 ‘{end_title}’까지의 프로세스를 깊이 이해하고 있으며,
절차별로 무엇을, 왜, 어떻게 해야 하는지를 명확하게 설명하세요.

답변 전 반드시 확인할 사항:
1. 제안 범위(공공/민간)와 고객 특성이 질문에 명확히 드러나 있나요?
2. 관련 이해관계자(영업대표·BD·PM 등)의 역할이 분명합니까?
3. RFI/RFP 일정과 제출 기한이 명시되어 있나요?
4. VDC-A/B/C 심의 요건(사전규격, 심의 소요 기간 등)이 충분히 주어졌습니까?
5. 리스크 검토 항목(가격·기술·하도급·법무 등)이 언급되어 있나요?

불명확한 점이 있으면, 추가 정보를 요청하세요.
""",
    "착수/계획": """당신은 AX SI 방법론의 ‘착수/계획’ 단계 전문가입니다.
‘{start_title}’부터 ‘{end_title}’까지의 준비사항을 체계적으로 파악하고,
PMS 구축, 조직·역할 정의, 관리정책 수립 등의 절차를 설명하세요.

답변 전 반드시 확인할 사항:
1. 수주 결정 일자와 착수계 제출일이 명확히 주어졌나요?
2. PMO·전사QA 등 주요 조직 구성원이 언급되어 있습니까?
3. 관리·작업 환경 구축 범위(PMS·인력관리·보안 등)가 충분히 제공되었나요?
4. 하도급 승인 여부 및 하도급 계약 조건이 명시되어 있나요?

모호한 점이 있으면, 추가 정보를 요청하세요.
"""
}

# ─────────────────────────────────────────────────────
# 0-5) STEP별 ChatPromptTemplate 생성
# ─────────────────────────────────────────────────────
STEP_PROMPTS = {
    step: ChatPromptTemplate.from_messages([
        ("system", STEP_SYSTEM_PROMPTS[step]),
        ("user", """
질문: {question}

관련 절차 요약:
{context}

이 단계는 '{start_title}'부터 '{end_title}'까지 진행됩니다.
오직 제공된 문서를 참고하여, 단계별로 무엇을, 왜, 어떻게 해야 하는지 설명해주세요.
""")
    ])
    for step in STEP_SYSTEM_PROMPTS
}

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
executor = ThreadPoolExecutor(max_workers=5)
bm25_weight, faiss_weight = 0.3, 0.7
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
            st.experimental_rerun()
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
# 7) 문서 split & VectorDB 빌드
# ─────────────────────────────────────────────────────
@st.cache_data(ttl=24*3600)
def load_all_docs() -> Tuple[Dict[str, List[Document]], Dict[str, List[Document]]]:
    from langchain.text_splitter import CharacterTextSplitter
    body_splitter  = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    proc_map, qna_map = {}, {}

    def dl_and_split(url, is_qna=False):
        docs = download_and_load(url)
        out_chunks = []
        for d in docs:
            text = d.page_content
            if d.metadata.get("page", 0) == 0 and not is_qna:
                # 첫 페이지: 인덱스용 자동 추출
                idxs = index_splitter.split_text(text)
                for idx in idxs:
                    out_chunks.append(Document(page_content=idx, metadata=d.metadata))
            else:
                # 나머지 페이지: 길이 기반 분할
                for chunk in body_splitter.split_text(text):
                    out_chunks.append(Document(page_content=chunk, metadata=d.metadata))
        return out_chunks

    # PROCESS PDF
    for step, url in PROCESS_PDF_URLS.items():
        proc_map[step] = dl_and_split(url, is_qna=False)

    # Q&A PDF
    for step, url in QNA_PDF_URLS.items():
        qna_map[step] = dl_and_split(url, is_qna=True)

    return proc_map, qna_map

@st.cache_resource(ttl=3600*24)
def build_vectordbs(
    _proc_map: Dict[str, List[Document]],
    _qna_map:  Dict[str, List[Document]]
) -> Tuple[Dict[str, FAISS], Dict[str, FAISS]]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    proc_vdb = {step: FAISS.from_documents(docs, emb)
                for step, docs in _proc_map.items()}
    qna_vdb  = {step: FAISS.from_documents(docs, emb)
                for step, docs in _qna_map.items()}
    return proc_vdb, qna_vdb

@st.cache_resource(ttl=3600*24)
def build_global_qna_vectordb(
    _qna_map: Dict[str, List[Document]]
) -> FAISS:
    all_qna = []
    for docs in _qna_map.values():
        all_qna.extend(docs)
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    return FAISS.from_documents(all_qna, emb)

@st.cache_resource(ttl=3600*24)
def build_index_retrievers() -> Dict[str, any]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    idx_retrs = {}
    for step, url in PROCESS_PDF_URLS.items():
        idx_docs = extract_index_chunks(url)
        if idx_docs:
            idx_retrs[step] = FAISS.from_documents(idx_docs, emb).as_retriever()
    return idx_retrs

@st.cache_resource(ttl=3600*24)
def build_substep_vectordbs(
    _proc_docs_map: Dict[str, List[Document]]
) -> Dict[str, Dict[str, FAISS]]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    sub_vdbs: Dict[str, Dict[str, FAISS]] = {}
    for step, docs in _proc_docs_map.items():
        idx_docs = extract_index_chunks(PROCESS_PDF_URLS[step])
        sub_db: Dict[str, FAISS] = {}
        for idx in idx_docs:
            title = idx.metadata["title"]
            subset = [d for d in docs if title in d.page_content]
            if subset:
                sub_db[title] = FAISS.from_documents(subset, emb)
        sub_vdbs[step] = sub_db
    return sub_vdbs

# 앱 시작 시 한 번만 로드·벡터화
with st.spinner("📦 데이터 로드 중…"):
    proc_docs_map, qna_docs_map   = load_all_docs()
    proc_vectordbs, qna_vectordbs = build_vectordbs(proc_docs_map, qna_docs_map)
    global_qna_vectordb           = build_global_qna_vectordb(qna_docs_map)
    index_retrievers              = build_index_retrievers()
    substep_vectordbs             = build_substep_vectordbs(proc_docs_map)


# ─────────────────────────────────────────────────────
# 8) Q&A 탭
# ─────────────────────────────────────────────────────
with qa_tab:
    st.header("AX SI 방법론 이행봇")

    # 1) 절차 단계 선택
    step = st.selectbox("📂 절차 단계를 하나 선택해 주세요", list(PROCESS_PDF_URLS.keys()))
    if step is None:
        st.info("모든 단계를 선택 후, 질문을 입력하고 '질문 요청' 버튼을 눌러주세요.")
        st.stop()

    # 2) SUBSTEP 선택
    idx_docs    = extract_index_chunks(PROCESS_PDF_URLS[step])
    sub_choices = [d.metadata["title"] for d in idx_docs]
    substep     = st.selectbox("⚙️ 세부 절차를 하나 선택해 주세요", sub_choices)
    if not substep:
        st.info("모든 단계를 선택 후, 질문을 입력하고 '질문 요청' 버튼을 눌러주세요.")
        st.stop()

    # 3) 질문 유형 선택
    qtype = st.selectbox("❓ 질문 유형을 하나 선택해 주세요", ["정의 요청", "일반 질문"])
    if not qtype:
        st.info("모든 단계를 선택 후, 질문을 입력하고 '질문 요청' 버튼을 눌러주세요.")
        st.stop()

    # 4) 절차 개요
    st.subheader(f"[{step}] 프로세스 개요")
    with st.expander("목록 펼치기", expanded=False):
        for d in idx_docs:
            st.markdown(f"- {d.page_content}")

    # 5) 질문 입력
    query = st.text_input("💬 질문을 입력하세요", key=f"query_{step}")

    # 6) 질문 요청 버튼 — 여기서만 모든 분석/API 호출 수행
    if st.button("질문 요청", key=f"btn_{step}"):
        if not query.strip():
            st.warning("질문을 입력해주세요.")
            st.stop()

        # (디버그) 전체 청크 확인
        with st.expander("🔍 전체 청크 확인", expanded=False):
            docs = proc_docs_map[step]
            st.write(f"• 총 청크 개수: {len(docs)}")
            for i, d in enumerate(docs[:3]):
                st.markdown(
                    f"**Chunk {i+1} (메타: {d.metadata}):**\n```\n{d.page_content[:200]}...\n```"
                )

        # 7) 질문 유형·SUBSTEP 매핑 출력
        st.info(f"📌 사용자의 질문은 ‘{substep}’ 단계의 “{qtype}” 입니다.")

        # 8) Q&A 사례 (<0.5 threshold) — 글로벌 retriever 사용
        docs_and_scores = global_qna_vectordb.similarity_search_with_score(query, k=3)
        with st.expander("🔍 Q&A 유사도 Top 3", expanded=False):
            for doc, score in docs_and_scores:
                st.write(f"- **{score:.3f}**: {doc.page_content.splitlines()[0]}…")
        # 첫 케이스가 충분히 유사하면 바로 리턴
        top_doc, top_score = docs_and_scores[0]
        if top_score >= 0.5:
            st.subheader("💡 사례 응답")
            st.write(top_doc.page_content)
            st.stop()

        # 9) SUBSTEP 벡터DB로 RetrievalQA
        retriever = substep_vectordbs[step].get(substep) or proc_vectordbs[step].as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0,
                           openai_api_key=os.environ["OPENAI_API_KEY"]),
            chain_type="stuff",
            retriever=retriever,
        )
        with st.spinner("답변 생성 중…"):
            answer = qa_chain.run(query)

        st.subheader("💡 답변")
        st.write(answer)

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

INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """당신은 AX SI 방법론 이행봇의 질문 의도 분류기입니다.
아래 6가지 유형 중 하나로 이 질문의 **의도**를 분류하세요.
- 정의 요청: 절차의 개념과 목적을 물음  
- 수행 절차 안내: 절차를 단계별로 물음  
- 산출물·문서 요구 사항: 준비해야 할 산출물·문서 물음  
- 책임·역할 분담: 누가 무엇을 담당하는지 물음  
- 일정·마일스톤 확인: 기한·마감·N일 이내 처리 여부 물음  
- 일반 질문: 위에 해당하지 않는 기타 문의  

질문: “{question}”

**출력 형식** (한 줄):
질문유형: <위 6가지 중 하나>"""
    )
])

def classify_with_llm(question: str) -> str:
    chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")),
        prompt=INTENT_CLASSIFICATION_PROMPT
    )
    output = chain.predict(question=question)
    # 예시: "질문유형: 정의 요청"
    return output.split(":")[-1].strip()


# ─────────────────────────────────────────────────────
# 0-1) 질문 유형 리스트 및 Persona + 지침 선택
# ─────────────────────────────────────────────────────
QUESTION_TYPES = [
    "자유 질의",
    "정의 요청",
    "수행 절차 안내",
    "산출물·문서 요구 사항",
    "책임·역할 분담",
    "일정·마일스톤 확인",
]

def select_persona_prompt(question_type: str) -> str:
    base_prompt = """\
당신은 대기업이자 사기업인 KT의 SI프로젝트 내부 절차 안내 담당자입니다.
질문자는 기본적으로 KT 직원이며, 공직자가 아닌 민간 기업의 직원입니다.
KT는 정부 기관이 아니며, 직원들은 공무원이 아닙니다.
"""

    if question_type == "자유 질의":
        return base_prompt + """\
주어진 모든 SI 절차 문서, Q&A, 용어집 등을 기반으로
어떤 질문에도 가장 관련 있는 정보를 찾아 답변하세요.
– 답변 근거: <제공된 문서>에 기반합니다.
– 상세 설명: 필요 시 간단한 예시나 추가 배경을 덧붙여도 됩니다.
– 범용성: KT 직원 외에도 이해할 수 있게 쉽게 서술합니다.
"""

    elif question_type == "단순 질의응답":
        return base_prompt + """\
주어진 SI 내부 절차에 대해 간단명료하게 답변해야 합니다. 다음 지침을 따라 주세요:

1. 답변 근거: <제공된 문서>에 기반하여 답변하세요.  
2. 질문 이해: 질문의 핵심을 정확히 파악하세요.  
3. 관련 절차 확인: 질문과 관련된 절차 및 세부절차를 명시하세요.  
4. 명확한 답변: 간결하고 명확하게 응답하세요.  
5. 추가 설명: 필요한 경우 간단한 부연 설명을 제공합니다.  
6. 한계 명시: 답변의 한계나 예외 사항이 있다면 언급하세요.  
7. 정보 부족 시 추가 정보 요청: 추가 정보가 필요하다면 질문을 통해 요청하세요.

답변은 SI 내부절차 담당자가 아닌 KT 직원도 이해할 수 있도록 쉽게 설명해 주세요.
"""

    ## 단계별 적용 예정
    # elif question_type == "정의 요청":
    #     ...
    # elif question_type == "수행 절차 안내":
    #     ...
    # (이하 생략)

    # 기본 페르소나 리턴
    return base_prompt

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

WORDPOOL_PDF_URLS = {
    "SI_용어집": "https://drive.google.com/uc?export=download&id=1aD4QYP1OBXRP7PbXYrlXHn5LlLyFzDtx"
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
def load_all_docs() -> Tuple[
    Dict[str, List[Document]],  # proc_map
    Dict[str, List[Document]],  # qna_map
    Dict[str, List[Document]]   # wordpool_map
]:
    # 첫 페이지 전용: 빈 줄(2번 연속 개행) 또는 “.␣”을 경계로 분할
    first_page_splitter = CharacterTextSplitter(
        separator=r"\n{2,}|\.(?:\s|$)",
        chunk_size=800,
        chunk_overlap=0,
        is_separator_regex=True,
    )
    # 나머지 페이지 전용: 고정 길이 청크
    body_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    proc_map: Dict[str, List[Document]] = {}
    qna_map:  Dict[str, List[Document]] = {}
    wordpool_map: Dict[str, List[Document]] = {}

    def dl_and_chunk(url: str) -> List[Document]:
        pages = download_and_load(url)
        if not pages:
            return []
        # 첫 페이지만 regex 스타일로 분할
        first, *rest = pages
        first_texts = first_page_splitter.split_text(first.page_content)
        first_docs = [
            Document(page_content=text, metadata={**first.metadata})
            for text in first_texts if text.strip()
        ]
        # 나머지는 고정 길이 분할
        rest_docs = body_splitter.split_documents(rest) if rest else []
        return first_docs + rest_docs

    # STEP 문서 로드
    for step, url in PROCESS_PDF_URLS.items():
        docs = dl_and_chunk(url)
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata["step"] = step
        proc_map[step] = docs

    # Q&A 문서 로드
    for step, url in QNA_PDF_URLS.items():
        docs = dl_and_chunk(url)
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata["step"] = step
        qna_map[step] = docs

    # 용어집(Wordpool) 문서 로드
    for name, url in WORDPOOL_PDF_URLS.items():
        docs = dl_and_chunk(url)
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata["step"] = name
        wordpool_map[name] = docs

    return proc_map, qna_map, wordpool_map
    
    def download_and_split(url: str) -> List[Document]:
        # PDF 다운로드
        resp = requests.get(url)
        resp.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(resp.content)
            tmp_path = tf.name

        try:
            pages = PyMuPDFLoader(tmp_path).load()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        docs: List[Document] = []
        if pages:
            # 첫 페이지는 인덱스(chunk_size=1)로, 나머지는 body_splitter
            first = pages[0]
            idx_chunks = index_splitter.split_text(first.page_content)
            for txt in idx_chunks:
                docs.append(Document(page_content=txt, metadata=first.metadata.copy()))
            # 2페이지부터
            rest = pages[1:]
            if rest:
                rest_docs = body_splitter.split_documents(rest)
                docs.extend(rest_docs)
        return docs

    # STEP 문서 로드
    for step, url in PROCESS_PDF_URLS.items():
        docs = download_and_split(url)
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata["step"] = step
        proc_map[step] = docs

    # Q&A 문서 로드
    for step, url in QNA_PDF_URLS.items():
        docs = download_and_split(url)
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata["step"] = step
        qna_map[step] = docs

    return proc_map, qna_map


@st.cache_resource(ttl=3600 * 24)
def build_vectordbs(
    _proc_docs_map: Dict[str, List[Document]],
    _qna_docs_map: Dict[str, List[Document]]
) -> Tuple[Dict[str, FAISS], Dict[str, FAISS]]:
    emb = OpenAIEmbeddings(model="text-embedding-ada-002",
                           openai_api_key=os.environ["OPENAI_API_KEY"])
    proc_vdb = {step: FAISS.from_documents(docs, emb)
                for step, docs in _proc_docs_map.items()}
    qna_vdb = {step: FAISS.from_documents(docs, emb)
               for step, docs in _qna_docs_map.items()}
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
def build_index_vectordbs() -> Dict[str, FAISS]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    idx_vdbs: Dict[str, FAISS] = {}
    for step, url in PROCESS_PDF_URLS.items():
        idx_docs = extract_index_chunks(url)
        if idx_docs:
            idx_vdbs[step] = FAISS.from_documents(idx_docs, emb)
    return idx_vdbs
    

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


with st.spinner("📦 데이터 로드 중…"):
    # 1) PDF 로드
    proc_docs_map, qna_docs_map, wordpool_map = load_all_docs()

    # 2) 프로세스·QnA 벡터 DB 생성
    proc_vectordbs, qna_vectordbs = build_vectordbs(proc_docs_map, qna_docs_map)

    # 3) 전체 QnA 합본 벡터 DB (자유 질의용)
    global_qna_vectordb           = build_global_qna_vectordb(qna_docs_map)

    # 4) STEP별 Substep 인덱스용 FAISS 생성
    index_vectordbs               = build_index_vectordbs()

    # 5) 세부절차별 벡터 DB 생성
    substep_vectordbs             = build_substep_vectordbs(proc_docs_map)

with qa_tab:
    st.header("AX SI 방법론 이행봇 - Q&A")

    # 1) STEP 선택
    step = st.selectbox("📂 절차 단계를 선택해 주세요",
                        list(PROCESS_PDF_URLS.keys()), key="sel_step")

    # 2) 질문 유형 선택
    qtype = st.selectbox("❓ 질문 유형을 선택해 주세요",
                         QUESTION_TYPES, key="sel_qtype")

    # 3) 질문 입력
    query = st.text_input("💬 질문을 입력하세요", key="input_query")

    # 4) 질문 요청
    if st.button("질문 요청", key="btn_query"):
        if not query:
            st.warning("❗️ 질문을 입력한 후 버튼을 눌러 주세요.")
            st.stop()

        # 5) Substep 자동 추론 (Index FAISS)
        idx_scores = index_vectordbs[step].similarity_search_with_score(query, k=3)
        top_idx_doc, _ = idx_scores[0]
        substep_option = top_idx_doc.page_content  # ex. "2. VDC-A 심의 준비"
        st.info(f"📌 사용자의 질문은 ‘{step}’ 단계의 “{substep_option}”에 대한 “{qtype}”입니다.")

        # 6) 절차(전체 STEP) / QnA Top-3 검색
        proc_db     = proc_vectordbs[step]
        proc_scores = proc_db.similarity_search_with_score(query, k=3)
        qna_db      = qna_vectordbs[step]
        qna_scores  = qna_db.similarity_search_with_score(query, k=3)

        # 7) 경로 분기 (QnA 최우선 유사도 >=0.7 ?)
        top_qna_score = qna_scores[0][1]
        if top_qna_score >= 0.7:
            # QnA 경로
            top_doc = qna_scores[0][0]
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(select_persona_prompt(qtype)),
                HumanMessagePromptTemplate.from_template(
                    """세부절차: {substep}
QnA 청크:
{chunk}

사용자 질문: {question}

위 정보를 바탕으로 문장형으로 답변해 주세요."""
                )
            ])
            chain  = LLMChain(
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                prompt=prompt
            )
            answer = chain.predict(
                substep=substep_option,
                chunk=top_doc.page_content,
                question=query
            )
        else:
            # 절차 경로
            top_doc = proc_scores[0][0]
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(select_persona_prompt(qtype)),
                HumanMessagePromptTemplate.from_template(
                    """세부절차: {substep}
절차 문서 청크:
{chunk}

사용자 질문: {question}

위 정보를 바탕으로 문장형으로 답변해 주세요."""
                )
            ])
            chain  = LLMChain(
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                prompt=prompt
            )
            answer = chain.predict(
                substep=substep_option,
                chunk=top_doc.page_content,
                question=query
            )

        # 8) 본문 응답
        st.markdown(f"### {substep_option}")
        st.write(answer)

        # 9) Expander: TOP3 - 절차 CHUNK
        with st.expander("1) TOP3 - 절차 CHUNK"):
            for i, (doc, score) in enumerate(proc_scores, start=1):
                st.write(f"**{i}. Score {score:.2f}** — {doc.page_content[:200]}…")
                st.write("---")

        # 10) Expander: TOP3 - QnA CHUNK
        with st.expander("2) TOP3 - QNA CHUNK"):
            for i, (doc, score) in enumerate(qna_scores, start=1):
                snippet = doc.page_content.replace("\n", " ")
                st.write(f"**{i}. Score {score:.2f}** — {snippet[:200]}…")


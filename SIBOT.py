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

from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain import LLMChain

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
    # (이미 작성하신 키워드 기반 분류)
    q_lower = q.lower()
    if any(k in q_lower for k in ["정의", "이란"]):
        return "정의 요청"
    if any(k in q_lower for k in ["어떻게", "절차", "방법"]):
        return "수행 절차 안내"
    if any(k in q_lower for k in ["산출물", "문서", "준비"]):
        return "산출물·문서 요구 사항"
    if any(k in q_lower for k in ["누가", "책임", "역할"]):
        return "책임·역할 분담"
    if any(k in q_lower for k in ["언제", "기한", "마감"]) or re.search(r"\d+일", q_lower):
        return "일정·마일스톤 확인"
    return "일반 질문"

def classify_with_llm(question: str) -> str:
    """
    LLM을 이용해 질문 유형("정의 요청" 등)으로 분류합니다.
    """
    system = SystemMessage(content=(
        "당신은 AX SI 방법론 이행봇의 질문 유형 분류기입니다. "
        "아래 질문을 다음 유형 중 하나로 분류하고, 오직 분류명만 응답하세요:\n"
        "- 정의 요청\n"
        "- 수행 절차 안내\n"
        "- 산출물·문서 요구 사항\n"
        "- 책임·역할 분담\n"
        "- 일정·마일스톤 확인\n"
        "- 일반 질문\n"
    ))
    user = HumanMessage(content=f"질문: {question}")
    chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"]),
        prompt=[system, user]
    )
    # LLM이 내놓는 문자열 양쪽 공백·개행을 제거하고 리턴
    return chain.run().strip()

# ─────────────────────────────────────────────────────
# 0-3) Base persona 및 질문유형별 시스템 메시지 정의
# ─────────────────────────────────────────────────────
BASE_PERSONA = """\
당신은 대기업 KT의 SI(Project Management) 전문 PM입니다.
KT 내부 프로세스, 조직·역할, 산출물 요건까지 정확히 파악하고 있으며,
질문자는 KT 직원이므로 너무 법률·학술적 용어 대신 실무에 바로 쓸 수 있는 언어로 답변하세요.
"""

QUESTION_TYPE_SYSTEM: Dict[str, str] = {
    "정의 요청": """\
주어진 세부절차의 개념과 목적을 명확히 정의해주세요.
– 무엇(What): 이 절차가 무엇인지
– 왜(Why): 이 절차가 필요한 이유
– 언제(When): 이 절차가 실시되는 시점
""",
    "일반 질문": """\
주어진 절차 컨텍스트를 참고하여, 사용자 질문의 의도에 맞게 간결하게 답변해주세요.
""",
    # 나중에 “수행 절차 안내” 등 더 늘리셔도 됩니다.
}

# 사용자 메시지 템플릿 (모든 유형 공통)
USER_TMPL = HumanMessagePromptTemplate.from_template(
    "질문: {question}\n\n절차 요약:\n{context}\n\n간결하게 답변해주세요."
)

def make_prompt_for_type(question_type: str) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(BASE_PERSONA),
        SystemMessagePromptTemplate.from_template(QUESTION_TYPE_SYSTEM[question_type]),
        USER_TMPL,
    ])

# ─────────────────────────────────────────────────────
# 0-4) 질문 의도·맥락 분류용 시스템 메시지
# ─────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────
# 0-2') classify_with_llm 를 LLMChain 버전으로 대체
# ─────────────────────────────────────────────────────
def classify_with_llm(question: str) -> str:
    # 1) 분류하도록 지시할 시스템 메시지
    system_prompt = SystemMessage(
        content=(
            "아래 질문을 다음 유형 중 하나로 분류하세요:\n"
            "- 정의 요청\n"
            "- 수행 절차 안내\n"
            "- 산출물·문서 요구 사항\n"
            "- 책임·역할 분담\n"
            "- 일정·마일스톤 확인\n"
            "- 일반 질문\n"
            "질문만 받고, 꼭 해당 분류 이름만 한 줄로 출력해주세요."
        )
    )
    # 2) 실제 사용자 질문
    user_prompt = HumanMessage(content=f"질문: {question}")
    # 3) 체인 생성 및 실행
    chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0,
                       openai_api_key=os.environ["OPENAI_API_KEY"]),
        prompt=[system_prompt, user_prompt]
    )
    classification = chain.run()
    return classification.strip()

# ─────────────────────────────────────────────────────
# 0-5) 질문유형별 Persona + SystemPrompt
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
# 0-6) STEP별 시스템 메시지 정의 (start/end 동적 반영)
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
# 0-7) STEP별 ChatPromptTemplate 생성
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
    # unpack all three
    proc_docs_map, qna_docs_map, wordpool_map = load_all_docs()

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
    if not step:
        st.info("모든 단계를 선택 후 질문해주세요.")
        st.stop()

    # 2) SUBSTEP 선택
    idx_docs    = extract_index_chunks(PROCESS_PDF_URLS[step])
    sub_choices = [d.metadata["title"] for d in idx_docs]
    substep     = st.selectbox("⚙️ 세부 절차를 하나 선택해 주세요", sub_choices)
    if not substep:
        st.info("세부 절차를 선택해주세요.")
        st.stop()

    # 3) 분류 방식 선택 (NEW)
    classify_method = st.radio(
        "🔍 질문 유형 분류 방식",
        ("키워드 기반", "LLM 기반", "비교 보기")
    )

    # 4) 절차 개요
    st.subheader(f"[{step}] 프로세스 개요")
    with st.expander("목록 펼치기", expanded=False):
        for d in idx_docs:
            st.markdown(f"- {d.page_content}")

    # 5) 질문 입력
    query = st.text_input("💬 질문을 입력하세요", key=f"query_{step}")
    if st.button("질문 요청", key=f"btn_{step}"):
        if not query.strip():
            st.warning("질문을 입력해주세요.")
            st.stop()

        # (디버그) 전체 청크 확인
        with st.expander("🔍 전체 청크 확인", expanded=False):
            docs = proc_docs_map[step]
            st.write(f"• 총 청크 개수: {len(docs)}")
            for i, d in enumerate(docs[:3]):
                st.markdown(f"**Chunk {i+1}**: `{d.page_content[:100]}…`")

        # 6) 질문 유형 분류
        if classify_method == "키워드 기반":
            qtype = classify_question_type(query)
        elif classify_method == "LLM 기반":
            qtype = classify_with_llm(query)
        else:  # 비교 보기
            kw = classify_question_type(query)
            llm = classify_with_llm(query)
            st.info(f"🔎 **키워드 기반**: {kw}  
                    🔍 **LLM 기반**: {llm}")
            # 대표 하나만 뽑고 싶으면, 예를 들어 KW 우선:
            qtype = kw

        st.info(f"📌 사용자의 질문은 ‘{substep}’ 단계의 “{qtype}” 입니다.")

        # 7) 글로벌 Q&A 매핑 (Top-3, threshold=0.5)
        docs_and_scores = global_qna_vectordb.similarity_search_with_score(query, k=3)
        with st.expander("🔍 Q&A 유사도 Top 3", expanded=False):
            for doc, score in docs_and_scores:
                first_line = doc.page_content.splitlines()[0]
                st.write(f"- **{score:.3f}** {first_line}…")
        top_doc, top_score = docs_and_scores[0]
        if top_score >= 0.5:
            st.subheader("💡 사례 응답")
            st.write(top_doc.page_content)
            st.stop()

        # 8) SUBSTEP RetrievalQA
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

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
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from kiwipiepy import Kiwi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain import LLMChain

# ─────────────────────────────────────────────────────
# 1) 유틸리티 함수 정의
# ─────────────────────────────────────────────────────
def download_and_load(url: str) -> List[Document]:
    resp = requests.get(url)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(resp.content)
        tmp_path = tf.name
    try:
        pages = PyMuPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)
    return pages


def extract_index_chunks(url: str) -> List[Document]:
    pages = download_and_load(url)
    if not pages: return []
    lines = pages[0].page_content.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("##"): start = i+1; break
    pattern = re.compile(r"^(\d+)\.\s*(.+)$")
    out = []
    for line in lines[start:]:
        m = pattern.match(line.strip())
        if not m: break
        num, title = m.groups()
        out.append(Document(page_content=f"{num}. {title}", metadata={"step": num, "title": title}))
    return out

# ─────────────────────────────────────────────────────
# 2) 질문 유도 및 분류 정의
# ─────────────────────────────────────────────────────
def classify_question_type(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["정의","이란"]): return "정의 요청"
    if any(k in ql for k in ["어떻게","절차","방법"]): return "수행 절차 안내"
    if any(k in ql for k in ["산출물","문서","준비"]): return "산출물·문서 요구 사항"
    if any(k in ql for k in ["누가","책임","역할"]): return "책임·역할 분담"
    if re.search(r"\d+일", ql) or any(k in ql for k in ["언제","기한","마감"]):
        return "일정·마일스톤 확인"
    return "일반 질문"

INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """
당신은 AX SI 방법론 이행봇의 질문 의도 분류기입니다.
아래 6가지 유형 중 하나로 분류:
- 정의 요청
- 수행 절차 안내
- 산출물·문서 요구 사항
- 책임·역할 분담
- 일정·마일스톤 확인
- 일반 질문

질문: “{question}”

출력형식: 질문유형: <위 6가지 중 하나>"""
    )
])

def classify_with_llm(question: str) -> str:
    chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        prompt=INTENT_CLASSIFICATION_PROMPT
    )
    out = chain.predict(question=question)
    return out.split(':')[-1].strip()

# ─────────────────────────────────────────────────────
# 3) 프롬프트 템플릿 정의
# ─────────────────────────────────────────────────────
BASE_PERSONA = """
당신은 대기업 KT의 SI(Project Management) 전문 PM입니다.
KT 내부 프로세스와 산출물 요건을 정확히 파악하고,
실무에 바로 적용 가능한 언어로 답변하세요.
"""

QUESTION_TYPE_SYSTEM: Dict[str,str] = {
    "정의 요청": """
주어진 세부절차 "{substep}"의 개념과 목적을 명확히 정의하세요.
– 무엇(What): 이 절차가 무엇인지
– 왜(Why): 왜 필요한지
– 언제(When): 언제 실시되는지
""",
    "수행 절차 안내": """
주어진 세부절차 "{substep}"를 단계별로 안내하세요.
– 선행·후속 절차 언급
– 각 단계(What→Why→How)
""",
    "산출물·문서 요구 사항": """
"{substep}"에서 준비해야 할 산출물·문서를 리스트업하세요.
– 산출물명
– 형식·템플릿
– 제출 시점
""",
    "책임·역할 분담": """
"{substep}"에서 각 이해관계자(영업대표·PM·BD 등)의 역할을 정리하세요.
– 누가(Who)
– 어떤 활동(What)
– 언제(When)
""",
    "일정·마일스톤 확인": """
"{substep}"의 기한·마일스톤을 표로 정리하세요.
– 시작일
– 종료일
– N일 이내 요건
""",
    "일반 질문": """
주어진 절차 컨텍스트를 참고하여, 질문 의도에 맞게 간결히 답변하세요.
""",
}

# ─────────────────────────────────────────────────────
# 4) 문서 로드 및 벡터 DB 생성
# ─────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_all_docs() -> Tuple[Dict[str,List[Document]],Dict[str,List[Document]]]:
    splitter_first = CharacterTextSplitter(separator=r"\n{2,}|\.(?:\s|$)", is_separator_regex=True, chunk_size=800, chunk_overlap=0)
    splitter_body = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    proc_map, qna_map = {}, {}
    for name, url in PROCESS_PDF_URLS.items():
        pages = download_and_load(url)
        docs = []
        if pages:
            first, *rest = pages
            for txt in splitter_first.split_text(first.page_content):
                docs.append(Document(page_content=txt, metadata={**first.metadata}))
            docs += splitter_body.split_documents(rest)
        proc_map[name] = docs
    for name, url in QNA_PDF_URLS.items():
        pages = download_and_load(url)
        docs = []
        if pages:
            first, *rest = pages
            for txt in splitter_first.split_text(first.page_content):
                docs.append(Document(page_content=txt, metadata={**first.metadata}))
            docs += splitter_body.split_documents(rest)
        qna_map[name] = docs
    return proc_map, qna_map

@st.cache_resource(ttl=86400)
def build_vectordbs(proc_docs_map, qna_docs_map):
    emb = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY"))
    return (
        {k: FAISS.from_documents(v, emb) for k,v in proc_docs_map.items()},
        {k: FAISS.from_documents(v, emb) for k,v in qna_docs_map.items()}
    )

# ─────────────────────────────────────────────────────
# 5) 앱 시작 및 전역 구축
# ─────────────────────────────────────────────────────
st.set_page_config(page_title="AX SI 방법론 이행봇", layout="wide")
PROCESS_PDF_URLS = {...}
QNA_PDF_URLS = {...}
proc_docs_map, qna_docs_map = load_all_docs()
proc_vdbs, qna_vdbs = build_vectordbs(proc_docs_map, qna_docs_map)
global_qna_vdb = FAISS.from_documents(sum(qna_docs_map.values(), []), OpenAIEmbeddings(...))
substep_vdbs = build_substep_vectordbs(proc_docs_map)

# ─────────────────────────────────────────────────────
# 6) Q&A 탭
# ─────────────────────────────────────────────────────
with st.tabs(["Q&A"])[0]:
    step = st.selectbox("📂 절차 단계", list(PROCESS_PDF_URLS.keys()))
    idxs = extract_index_chunks(PROCESS_PDF_URLS[step])
    substep = st.selectbox("⚙️ 세부 절차", [d.metadata['title'] for d in idxs])
    classify_method = st.radio("🔍 분류 방식", ("키워드 기반","LLM 기반","비교 보기"))
    query = st.text_input("💬 질문을 입력하세요")
    if st.button("질문 요청"):
        # determine question type
        if classify_method == "키워드 기반":
            qtype = classify_question_type(query)
        elif classify_method == "LLM 기반":
            qtype = classify_with_llm(query)
        else:
            kw = classify_question_type(query)
            llm = classify_with_llm(query)
            st.write(f"키워드 기반: {kw}, LLM 기반: {llm}")
            qtype = kw
        st.info(f"📌 사용자의 질문은 ‘{substep}’ 단계의 “{qtype}”입니다.")

        # Q&A 벡터 매핑
        qscores = global_qna_vdb.similarity_search_with_score(query, k=3)
        top_doc, score = qscores[0]
        if score >= 0.5:
            st.subheader("💡 qna 응답")
            st.write(top_doc.page_content)
            return
        
        # 프로세스 기반
        retr = substep_vdbs[step].get(substep) or proc_vdbs[step].as_retriever()
        refs = retr.get_relevant_documents(query)
        st.expander("🔍 참조된 프로세스 내용", expanded=False).write([d.page_content for d in refs[:3]])
        st.subheader("💡 프로세스 응답")
        answer = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
            chain_type="stuff",
            retriever=retr
        ).run(query)
        st.write(answer)

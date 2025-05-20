# ─────────────────────────────────────────────────────
# 0) 라이브러리 및 선언 영역 (RegexSplitter 제거)
# ─────────────────────────────────────────────────────
import streamlit as st
import requests, tempfile, os, re
from datetime import datetime, timezone, timedelta
from typing import TypedDict, Dict, List, Tuple
from io import BytesIO
import uuid, time, threading, asyncio, multiprocessing
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
    ql = q.lower()
    if any(k in ql for k in ["정의", "이란"]):
        return "정의 요청"
    if any(k in ql for k in ["어떻게", "절차", "방법"]):
        return "수행 절차 안내"
    if any(k in ql for k in ["산출물", "문서", "준비"]):
        return "산출물·문서 요구 사항"
    if any(k in ql for k in ["누가", "책임", "역할"]):
        return "책임·역할 분담"
    if re.search(r"\d+일", ql) or any(k in ql for k in ["언제", "기한", "마감"]):
        return "일정·마일스톤 확인"
    return "일반 질문"

# ─────────────────────────────────────────────────────
# 0-3) 질문유형별 Persona + SystemPrompt
# ─────────────────────────────────────────────────────
BASE_PERSONA = """\
당신은 대기업 KT의 SI(Project Management) 전문 PM입니다.
KT 내부 프로세스, 조직·역할, 산출물 요건까지 정확히 파악하고 있으며,
질문자는 KT 직원이므로 실무에서 바로 쓸 수 있는 언어로 답변하세요.
"""
QUESTION_TYPE_SYSTEM: Dict[str,str] = {
    "정의 요청": """\
{base_persona}

지금 질문은 세부절차 “{sub_title}”의 **정의**를 묻고 있습니다.
– **What**: 이 절차가 무엇인지
– **Why**: 이 절차가 필요한 이유
– **When**: 이 절차가 실시되는 시점
""",
    "수행 절차 안내": """\
{base_persona}

지금 질문은 세부절차 “{sub_title}”의 **수행 절차**를 묻고 있습니다.
– 선행절차와 후속절차 간략 언급
– 각 단계(**What→Why→How**) 순서대로 설명
""",
    "산출물·문서 요구 사항": """\
{base_persona}

지금 질문은 세부절차 “{sub_title}”의 **필수 산출물·문서**를 묻고 있습니다.
– 산출물명
– 형식·템플릿
– 제출 시점
""",
    "책임·역할 분담": """\
{base_persona}

지금 질문은 세부절차 “{sub_title}”의 **책임 및 역할 분담**을 묻고 있습니다.
– **Who**: 누가
– **What**: 어떤 활동
– **When**: 언제
(가능하면 RACI 표 형식)
""",
    "일정·마일스톤 확인": """\
{base_persona}

지금 질문은 세부절차 “{sub_title}”의 **일정·마감 기한**을 묻고 있습니다.
– 시작일, 종료일, N일 이내 요건을 표로 정리
""",
    "일반 질문": """\
{base_persona}

지금 질문은 세부절차 “{sub_title}”에 대한 **일반 문의**입니다.
관련된 **What / Why / How** 또는 체크리스트 형태로 간결히 답변하세요.
"""
}

from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate

USER_TMPL = HumanMessagePromptTemplate.from_template(
    "질문: {question}\n\n절차 요약:\n{context}\n\n간결하게 답변해주세요."
)

def make_prompt_for_type(question_type: str, sub_title: str) -> ChatPromptTemplate:
    sys_text = QUESTION_TYPE_SYSTEM[question_type].format(
        base_persona=BASE_PERSONA,
        sub_title=sub_title
    )
    sys_msg = SystemMessagePromptTemplate.from_template(sys_text)
    return ChatPromptTemplate.from_messages([
        sys_msg,
        USER_TMPL
    ])

# ─────────────────────────────────────────────────────
# 7) 문서 로드 & vectordb 생성 (wordpool 맵 추가)
# ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600*24)
def load_all_docs() -> Tuple[
    Dict[str, List[Document]],  # proc_map
    Dict[str, List[Document]],  # qna_map
    Dict[str, List[Document]]   # wordpool_map
]:
    first_page_splitter = CharacterTextSplitter(
        separator=r"\n{2,}|\.(?:\s|$)",
        chunk_size=800,
        chunk_overlap=0,
        is_separator_regex=True,
    )
    body_splitter = CharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )

    proc_map, qna_map, wordpool_map = {}, {}, {}

    def dl_and_chunk(url: str) -> List[Document]:
        pages = download_and_load(url)
        if not pages: return []
        first, *rest = pages
        first_texts = first_page_splitter.split_text(first.page_content)
        first_docs = [
            Document(page_content=t, metadata={**first.metadata})
            for t in first_texts if t.strip()
        ]
        rest_docs = body_splitter.split_documents(rest) if rest else []
        return first_docs + rest_docs

    # STEP PDF
    for step, url in PROCESS_PDF_URLS.items():
        docs = dl_and_chunk(url)
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata["step"] = step
        proc_map[step] = docs

    # Q&A PDF
    for step, url in QNA_PDF_URLS.items():
        docs = dl_and_chunk(url)
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata["step"] = step
        qna_map[step] = docs

    # WORDPOOL PDF
    for name, url in WORDPOOL_PDF_URLS.items():
        docs = dl_and_chunk(url)
        for d in docs:
            d.page_content = preprocess(d.page_content)
            d.metadata["step"] = name
        wordpool_map[name] = docs

    return proc_map, qna_map, wordpool_map

@st.cache_resource(ttl=3600*24)
def build_vectordbs(
    proc_docs_map: Dict[str, List[Document]],
    qna_docs_map: Dict[str, List[Document]],
    wordpool_map: Dict[str, List[Document]]
) -> Tuple[Dict[str, FAISS], Dict[str, FAISS], FAISS]:
    emb = OpenAIEmbeddings(model="text-embedding-ada-002",
                           openai_api_key=os.environ["OPENAI_API_KEY"])
    proc_vdb  = {s: FAISS.from_documents(d, emb) for s,d in proc_docs_map.items()}
    qna_vdb   = {s: FAISS.from_documents(d, emb) for s,d in qna_docs_map.items()}
    # global wordpool
    all_wp = sum(wordpool_map.values(), [])
    wp_vdb = FAISS.from_documents(all_wp, emb)
    return proc_vdb, qna_vdb, wp_vdb

# …(build_global_qna_vectordb, build_index_retrievers, build_substep_vectordbs 동일)…

# 언팩
with st.spinner("📦 데이터 로드 중…"):
    proc_docs_map, qna_docs_map, wordpool_map = load_all_docs()
    proc_vectordbs, qna_vectordbs, wp_vectordb = build_vectordbs(proc_docs_map, qna_docs_map, wordpool_map)
    global_qna_vectordb     = build_global_qna_vectordb(qna_docs_map)
    index_retrievers        = build_index_retrievers()
    substep_vectordbs       = build_substep_vectordbs(proc_docs_map)

# ─────────────────────────────────────────────────────
# 8) Q&A 탭 (Dynamic Prompt + LangSmith 콜백 예시)
# ─────────────────────────────────────────────────────
with qa_tab:
    # … (UI: step, substep, qtype 선택부 동일) …

    # 7) 사용자 매핑
    st.info(f"📌 사용자의 질문은 ‘{substep}’ 단계의 “{qtype}” 입니다.")

    # 8) Q&A PDF 매핑 (Top-3, threshold=0.5)
    docs_and_scores = global_qna_vectordb.similarity_search_with_score(query, k=3)
    with st.expander("🔍 Q&A 유사도 Top 3", expanded=False):
        for doc, score in docs_and_scores:
            st.write(f"- **{score:.3f}**: {doc.page_content.splitlines()[0]}…")
    top_doc, top_score = docs_and_scores[0]
    if top_score >= 0.5:
        st.subheader("💡 사례 응답")
        st.write(top_doc.page_content)
        st.stop()

    # 9) RetrievalQA + Dynamic Prompt
    prompt = make_prompt_for_type(qtype, substep)
    tracer = LangSmithTracer(project_name="SI-Process-Bot")
    handler = LangSmithCallbackHandler(tracer)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"]),
        chain_type="stuff",
        retriever=substep_vectordbs[step].get(substep) or proc_vectordbs[step].as_retriever(),
        callbacks=[handler],
        verbose=True,
        chain_type_kwargs={"prompt": prompt},
    )
    with st.spinner("답변 생성 중…"):
        answer = qa_chain.run({"query": query})

    st.subheader("💡 답변")
    st.write(answer)

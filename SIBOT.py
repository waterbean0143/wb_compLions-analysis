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
# 0) 문서 URL 정의 (반드시 함수 정의 위에 위치)
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
# 0) Streamlit 페이지 설정 — 반드시 첫 번째 Streamlit 호출
# ─────────────────────────────────────────────────────
import streamlit as st
st.set_page_config(
    page_title="AX SI 방법론 이행봇",
    layout="wide",
)

# ─────────────────────────────────────────────────────
# 4) 문서 로드 및 벡터 DB 생성
# ─────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_all_docs() -> Tuple[
        Dict[str, List[Document]],
        Dict[str, List[Document]],
        Dict[str, List[Document]]
    ]:
    splitter_first = CharacterTextSplitter(
        separator=r"\n{2,}|\.(?:\s|$)",
        is_separator_regex=True,
        chunk_size=800,
        chunk_overlap=0,
    )
    splitter_body = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    # 1) 프로세스 문서
    proc_map: Dict[str, List[Document]] = {}
    for name, url in PROCESS_PDF_URLS.items():
        pages = download_and_load(url)
        docs: List[Document] = []
        if pages:
            first, *rest = pages
            for txt in splitter_first.split_text(first.page_content):
                docs.append(Document(page_content=txt, metadata={**first.metadata}))
            docs += splitter_body.split_documents(rest)
        proc_map[name] = docs

    # 2) 대표 QnA 문서
    qna_map: Dict[str, List[Document]] = {}
    for name, url in QNA_PDF_URLS.items():
        pages = download_and_load(url)
        docs: List[Document] = []
        if pages:
            first, *rest = pages
            for txt in splitter_first.split_text(first.page_content):
                docs.append(Document(page_content=txt, metadata={**first.metadata}))
            docs += splitter_body.split_documents(rest)
        qna_map[name] = docs

    # 3) SI 용어집 워드풀
    wordpool_map: Dict[str, List[Document]] = {}
    for name, url in WORDPOOL_PDF_URLS.items():
        pages = download_and_load(url)
        full_text = "\n".join(p.page_content for p in pages) if pages else ""
        docs: List[Document] = []
        for line in full_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                word, rest = [s.strip() for s in line.split(":", 1)]
                definitions = [d.strip() for d in rest.split(",") if d.strip()]
                content = f"{word}: {', '.join(definitions)}"
                docs.append(Document(
                    page_content=content,
                    metadata={"source": name, "wp_word": word, "wp_definitions": definitions}
                ))
            else:
                docs.append(Document(page_content=line, metadata={"source": name}))
        wordpool_map[name] = docs

    return proc_map, qna_map, wordpool_map


@st.cache_resource(ttl=86400)
def build_vectordbs(
    proc_docs_map: Dict[str, List[Document]],
    qna_docs_map:  Dict[str, List[Document]],
    wordpool_docs_map: Dict[str, List[Document]]
) -> Tuple[
        Dict[str, FAISS],
        Dict[str, FAISS],
        Dict[str, FAISS]
    ]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    proc_vdbs   = {k: FAISS.from_documents(v, emb) for k, v in proc_docs_map.items()}
    qna_vdbs    = {k: FAISS.from_documents(v, emb) for k, v in qna_docs_map.items()}
    wp_vdbs     = {k: FAISS.from_documents(v, emb) for k, v in wordpool_docs_map.items()}
    return proc_vdbs, qna_vdbs, wp_vdbs

# ─────────────────────────────────────────────────────
# 5) 앱 시작 및 전역 구축
# ─────────────────────────────────────────────────────

# 문서 맵 및 벡터 DB 생성 호출
proc_docs_map, qna_docs_map, wordpool_docs_map = load_all_docs()
proc_vdbs, qna_vdbs, wp_vdbs           = build_vectordbs(
    proc_docs_map,
    qna_docs_map,
    wordpool_docs_map
)


# ─────────────────────────────────────────────────────
# 6) Q&A 탭 (프로세스 Top-3 & QnA Top-3 → 우선순위 결정 → 원문 응답)
# ─────────────────────────────────────────────────────
with qa_tab:
    st.header("AX SI 방법론 이행봇 - Q&A")

    # 1) 절차 단계 선택
    step = st.selectbox(
        "📂 절차 단계를 선택하세요",
        list(PROCESS_PDF_URLS.keys()),
        key="step_select"
    )

    # 2) 세부 절차(소단계) 선택
    idx_docs = extract_index_chunks(PROCESS_PDF_URLS[step])
    substep = st.selectbox(
        "⚙️ 세부 절차를 선택하세요",
        [d.metadata["title"] for d in idx_docs],
        key="substep_select"
    )

    # 3) 질문 입력
    query = st.text_input("💬 질문을 입력하세요", key=f"query_{step}")
    if not query:
        st.stop()

    # 4) 질문 요청 버튼
    if st.button("질문 요청", key=f"btn_{step}"):
        # 4-1) 질문 유형 분류
        qtype = classify_question_type(query)
        st.info(f"📌 질문 유형: ‘{qtype}’")

        # 4-2) 프로세스 문서 Top-3 검색
        proc_scores = proc_vdbs[step].similarity_search_with_score(query, k=3)
        st.subheader("🔍 프로세스 유사도 Top-3")
        for i, (doc, score) in enumerate(proc_scores, start=1):
            meta = doc.metadata
            location = f"page {meta.get('page','?')}, sentence {meta.get('sentence_id','?')}"
            snippet  = doc.page_content.replace("\n", " ")[:200] + "…"
            st.write(f"{i}. Score {score:.2f} — {snippet}")
            st.write(f"   • 위치: {location}")

        # 4-3) QnA 문서 Top-3 검색
        qna_scores = qna_vdbs[step].similarity_search_with_score(query, k=3)
        st.subheader("🔍 사례 응답 유사도 Top-3")
        for i, (doc, score) in enumerate(qna_scores, start=1):
            lines  = doc.page_content.splitlines()
            q_line = next((l for l in lines if l.startswith("[질문]")), lines[0])
            a_line = next((l for l in lines if l.startswith("[[[답변]")), lines[-1])
            st.write(f"{i}. Score {score:.2f} — {q_line}")
            st.write(f"   {a_line}")

        # 5) 최우선 QnA 결과 활용 여부 결정
        top_doc, top_score = qna_scores[0]
        if top_score >= 0.5:
            st.subheader("💡 QnA 우선 응답")
            answer_text = next(
                (l for l in top_doc.page_content.splitlines() if l.startswith("[[[답변]")),
                ""
            )
            st.write(answer_text.replace("[[[답변]", "").rstrip("]"))

        else:
            st.subheader("💡 프로세스 응답")
            retriever = substep_vdbs[step].get(substep) or proc_vdbs[step].as_retriever()
            prompt    = make_prompt_for_type(qtype)
            qa_chain  = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=os.environ["OPENAI_API_KEY"]
                ),
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt},
            )
            answer = qa_chain.run({"query": query})
            st.write(answer)

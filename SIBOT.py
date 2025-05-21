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

from kiwipiepy import Kiwi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.chains import LLMChain

# ─────────────────────────────────────────────────────
# 0-1) PDF 다운로드 및 인덱스 추출
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
    raw = download_and_load(url)
    if not raw: return []
    lines = raw[0].page_content.splitlines()
    start = next((i+1 for i,l in enumerate(lines) if l.strip().startswith("##")), 0)
    pattern = re.compile(r"^(\d+)\.\s*(.+)$")
    idxs: List[Document] = []
    for line in lines[start:]:
        m = pattern.match(line.strip())
        if not m: break
        num, title = m.groups()
        idxs.append(Document(page_content=f"{num}. {title}", metadata={"step":int(num),"title":title}))
    return idxs

# ─────────────────────────────────────────────────────
# 0-2) 질문 유형 분류 및 Persona
# ─────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str; step_name: str; sub_title: str
    question_type: str; context: str; response: str; attempts: int

INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """당신은 질문 의도 분류기입니다. 아래 6가지 유형 중 하나로 분류하세요.
- 정의 요청, 수행 절차 안내, 산출물·문서 요구 사항, 책임·역할 분담, 일정·마일스톤 확인, 일반 질문
질문: “{question}”
출력: 질문유형: <위 6가지 중 하나>"""
    )
])
def classify_with_llm(question: str) -> str:
    out = LLMChain(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                   prompt=INTENT_CLASSIFICATION_PROMPT).predict(question=question)
    return out.split(":")[-1].strip()

QUESTION_TYPES = [
    "자유 질의","정의 요청","수행 절차 안내",
    "산출물·문서 요구 사항","책임·역할 분담","일정·마일스톤 확인",
]
def select_persona_prompt(qtype: str) -> str:
    base = """당신은 KT SI 프로젝트 내부 절차 안내 담당자입니다.
질문자는 KT 직원입니다."""
    if qtype == "자유 질의":
        return base + """
주어진 절차 문서, Q&A, 용어집을 기반으로 답변하세요.
– 답변 근거: <제공된 문서>"""
    else:
        return base

# ─────────────────────────────────────────────────────
# 1) 페이지 설정 및 Secrets
# ─────────────────────────────────────────────────────
st.set_page_config(page_title="AX SI 방법론 이행봇", layout="wide")
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

# ─────────────────────────────────────────────────────
# 2) 전역 설정
# ─────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'NanumGothic'
executor = ThreadPoolExecutor(max_workers=5)

# ─────────────────────────────────────────────────────
# 3) 로그인
# ─────────────────────────────────────────────────────
users = {"10154371":"10154371","10154372":"10154372","10156350":"10156350"}
if 'logged_in' not in st.session_state:
    st.sidebar.title("🔒 로그인")
    uid = st.sidebar.text_input("ID"); pwd = st.sidebar.text_input("PW", type="password")
    if st.sidebar.button("로그인"):
        if uid in users and users[uid]==pwd:
            st.session_state['logged_in']=True; st.experimental_rerun()
        else: st.sidebar.error("로그인 실패")
    st.stop()

# ─────────────────────────────────────────────────────
# 4) PDF URL 매핑
# ─────────────────────────────────────────────────────
PROCESS_PDF_URLS = {
    "제안/계약":"https://drive.google.com/uc?export=download&id=1TNOhmUds7hMpwz3NO4QD-mO-J1sUJoEa",
    "착수/계획":"https://drive.google.com/uc?export=download&id=16j9ypXkWD7oi477ylSXWhVVe7jLtRuI7",
}
QNA_PDF_URLS = {
    "제안/계약":"https://drive.google.com/uc?export=download&id=17M1mnMZVl29EahbSVqzcyZEX8LYsx5ER",
}
WORDPOOL_PDF_URLS = {
    "SI_용어집":"https://drive.google.com/uc?export=download&id=1aD4QYP1OBXRP7PbXYrlXHn5LlLyFzDtx"
}

# ─────────────────────────────────────────────────────
# 5) UI & 탭 정의
# ─────────────────────────────────────────────────────
st.sidebar.title("⚙️ 설정")
answer_mode = st.sidebar.radio("답변 모드", ["빠른 답변","정확한 답변"])
tabs = st.tabs(["Q&A","추가예정"])
qa_tab, _ = tabs

# ─────────────────────────────────────────────────────
# 6) 전처리 (키워드 추출용)
# ─────────────────────────────────────────────────────
kiwi = Kiwi()
def preprocess(text: str) -> str:
    toks = kiwi.analyze(text)[0][0]
    return ' '.join(t.form for t in toks if t.tag.startswith(('N','V','MA')))

# ─────────────────────────────────────────────────────
# 7) 문서 로드 & VectorDB 생성
# ─────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_all_docs() -> Tuple[
    Dict[str,List[Document]],
    Dict[str,List[Document]],
    Dict[str,List[Document]],
    Dict[str,List[Document]]
]:
    # 텍스트 분할기
    split_first = CharacterTextSplitter(separator=r"\n{2,}|\.(?:\s|$)",
                                        is_separator_regex=True,chunk_size=800,chunk_overlap=0)
    split_body  = CharacterTextSplitter(chunk_size=800,chunk_overlap=100)

    proc_map, qna_map, wp_map = {}, {}, {}
    orig_proc, orig_qna, orig_wp = {}, {}, {}
    # 7-1) 프로세스 문서
    for name,url in PROCESS_PDF_URLS.items():
        pages = download_and_load(url)
        orig_proc[name]=pages
        docs: List[Document]=[]
        if pages:
            first,*rest=pages
            for txt in split_first.split_text(first.page_content):
                docs.append(Document(page_content=txt,metadata={**first.metadata}))
            docs += split_body.split_documents(rest)
        proc_map[name]=docs
        
    # 7-2) QnA 문서 (블록 단위)
    for name, url in QNA_PDF_URLS.items():
        pages = download_and_load(url)
        orig_qna[name] = pages
        # PDF 전체를 하나의 문자열로
        full_text = "\n".join(p.page_content for p in pages)
        # “[질문”이 나오는 곳에서 split
        raw_qnas = [blk for blk in re.split(r'(?=\[질문\s*\d+\s*[:\]])', full_text) if blk.strip()]
        docs = []
        for blk in raw_qnas:
            lines = blk.splitlines()
            # 태그 (예: “[질문 1 : VDC-A 대상 사업]”)
            tag = next((l for l in lines if l.startswith("[질문")), "")
            # 질문 본문 (태그 제외)
            question_context = "\n".join(
                l for l in lines
                if not l.startswith("[[") and not l.startswith("[[[")
            ).strip()
            # 답변 본문
            answer_context = "\n".join(
                l for l in lines if l.startswith("[[[답변]") or l.startswith("[[답변]")
            ).strip()
            docs.append(Document(
                page_content=blk,
                metadata={
                    "tag": tag,
                    "question_context": question_context,
                    "answer_context": answer_context,
                    **pages[0].metadata
                }
            ))
        qna_map[name] = docs
     
    # 7-3) 워드풀 (생략 가능)
    for name,url in WORDPOOL_PDF_URLS.items():
        orig_wp[name]=download_and_load(url)
        wp_map[name]=[]
    # 7-4) original_pages 통합
    original_pages={}
    for k,v in orig_proc.items(): original_pages[f"proc:{k}"]=v
    for k,v in orig_qna.items(): original_pages[f"qna:{k}"]=v
    for k,v in orig_wp.items(): original_pages[f"wp:{k}"]=v

    return proc_map,qna_map,wp_map,original_pages

@st.cache_resource(ttl=86400)
def build_vectordbs(
    proc_docs_map: Dict[str,List[Document]],
    qna_docs_map:  Dict[str,List[Document]]
) -> Tuple[Dict[str,FAISS],Dict[str,FAISS]]:
    emb = OpenAIEmbeddings(model="text-embedding-ada-002",
                           openai_api_key=os.environ["OPENAI_API_KEY"])
    p_vdb={s:FAISS.from_documents(d,emb) for s,d in proc_docs_map.items()}
    q_vdb={s:FAISS.from_documents(d,emb) for s,d in qna_docs_map.items()}
    return p_vdb,q_vdb

@st.cache_resource(ttl=86400)
def build_global_qna_vectordb(qna_map: Dict[str,List[Document]]) -> FAISS:
    all_docs=[d for docs in qna_map.values() for d in docs]
    emb=OpenAIEmbeddings(model="text-embedding-ada-002",
                         openai_api_key=os.environ["OPENAI_API_KEY"])
    return FAISS.from_documents(all_docs,emb)

@st.cache_resource(ttl=86400)
def build_qna_vectordbs(qna_docs_map: Dict[str, List[Document]]) -> Dict[str, FAISS]:
    emb = OpenAIEmbeddings(model="text-embedding-ada-002",
                           openai_api_key=os.getenv("OPENAI_API_KEY"))
    return {
        step: FAISS.from_documents(docs, emb)
        for step, docs in qna_docs_map.items()
    }

@st.cache_resource(ttl=86400)
def build_index_vectordbs() -> Dict[str,FAISS]:
    emb=OpenAIEmbeddings(model="text-embedding-ada-002",
                         openai_api_key=os.environ["OPENAI_API_KEY"])
    idxs={}
    for step,url in PROCESS_PDF_URLS.items():
        docs=extract_index_chunks(url)
        if docs: idxs[step]=FAISS.from_documents(docs,emb)
    return idxs

@st.cache_resource(ttl=86400)
def build_substep_vectordbs(proc_map: Dict[str,List[Document]]) -> Dict[str,Dict[str,FAISS]]:
    emb=OpenAIEmbeddings(model="text-embedding-ada-002",
                         openai_api_key=os.environ["OPENAI_API_KEY"])
    sub_vdbs={}
    for step,docs in proc_map.items():
        idxs=extract_index_chunks(PROCESS_PDF_URLS[step])
        m={}
        for idx in idxs:
            title=idx.metadata["title"]
            subset=[d for d in docs if title in d.page_content]
            if subset: m[title]=FAISS.from_documents(subset,emb)
        sub_vdbs[step]=m
    return sub_vdbs

@st.cache_resource(ttl=86400)
def build_qna_substep_vectordbs(
    proc_map: Dict[str,List[Document]],
    qna_map:  Dict[str,List[Document]]
) -> Dict[str,Dict[str,FAISS]]:
    emb=OpenAIEmbeddings(model="text-embedding-ada-002",
                         openai_api_key=os.environ["OPENAI_API_KEY"])
    res={}
    for step,docs in qna_map.items():
        grp={}
        for d in docs:
            tag=d.metadata.get("tag","")
            sub=re.sub(r"^\[질문\s*\d+\]\s*","",tag)
            grp.setdefault(sub,[]).append(d)
        res[step]={sub:FAISS.from_documents(lst,emb) for sub,lst in grp.items() if lst}
    return res

@st.cache_resource(ttl=86400)
def build_bm25(proc_map: Dict[str,List[Document]]):
    return {s:BM25Retriever.from_documents(docs) for s,docs in proc_map.items()}

@st.cache_resource(ttl=86400)
def build_ensemble(p_vdbs, bm25s):
    ers={}
    for s in p_vdbs:
        ers[s]=EnsembleRetriever(
            retrievers=[p_vdbs[s].as_retriever(), bm25s[s]],
            weights=[0.7,0.3]
        )
    return ers

# ─────────────────────────────────────────────────────
# 8) 데이터 로드 & 벡터 DB 빌드
# ─────────────────────────────────────────────────────
with st.spinner("📦 데이터 로드 중…"):
    # 문서 로드 직후
    proc_docs_map, qna_docs_map, wp_map, original_pages = load_all_docs()

    # vectordb 빌드
    index_vectordbs       = build_index_vectordbs()
    substep_vectordbs     = build_substep_vectordbs(proc_docs_map)
    qna_vectordbs         = build_qna_vectordbs(qna_docs_map)
    qna_substep_vectordbs = build_qna_substep_vectordbs(qna_docs_map)

# ─────────────────────────────────────────────────────
# 9) Q&A 탭 (STEP→SUBSTEP 추론→TOP3 절차→TOP3 QnA→답변)
# ─────────────────────────────────────────────────────
with qa_tab:
    st.header("AX SI 방법론 이행봇 - Q&A")

    # 1) STEP 선택
    step = st.selectbox(
        "📂 절차 단계를 선택해 주세요",
        list(PROCESS_PDF_URLS.keys()),
        key="sel_step"
    )

    # 1-1) 전체 INDEX(서브절차) 목록
    idx_docs = extract_index_chunks(PROCESS_PDF_URLS[step])
    with st.expander("🔖 전체 세부절차 목록", expanded=False):
        for doc in idx_docs:
            st.write(f"- {doc.metadata['title']}")

    # 2) 질문 유형 선택
    qtype = st.selectbox(
        "❓ 질문 유형을 선택해 주세요",
        QUESTION_TYPES,
        key="sel_qtype"
    )

    # 3) 질문 입력
    query = st.text_input("💬 질문을 입력하세요", key="input_query")

    # 4) 질문 요청
    if st.button("질문 요청", key="btn_query"):
        if not query.strip():
            st.warning("❗️ 질문을 입력한 후 버튼을 눌러 주세요.")
            st.stop()

        # 5) Substep 자동 추론
        idx_scores     = index_vectordbs[step].similarity_search_with_score(query, k=1)
        substep_option = idx_scores[0][0].page_content
        st.info(f"📌 사용자의 질문은 '{step}' 단계의 \"{substep_option}\"에 대한 \"{qtype}\"입니다.")

        # 6) TOP3 - 절차 서브스텝
        substep_scores = index_vectordbs[step].similarity_search_with_score(query, k=3)
        with st.expander("1) TOP3 - 절차 서브스텝", expanded=False):
            for i, (sub_doc, dist) in enumerate(substep_scores, start=1):
                sub = sub_doc.page_content
                similarity = 1.0 - dist
                st.markdown(f"**[TOP_{i}]. {sub} — 유사도 {similarity:.2f}**")
                # 세부 청크가 있으면 간단히 snippet, 없으면 원본문서 블록
                vdb = substep_vectordbs[step].get(sub)
                if vdb:
                    chunk_scores = vdb.similarity_search_with_score(query, k=3)
                    for j, (c_doc, c_dist) in enumerate(chunk_scores, start=1):
                        snippet = c_doc.page_content.replace("\n", " ")[:200] + "…"
                        st.write(f"  {j}. {snippet} (유사도 {1-c_dist:.2f})")
                else:
                    # 원본문서에서 ##sub 부터 다음 ##까지 추출
                    pages = original_pages[f"proc:{step}"][1:]
                    page_text = next((p.page_content for p in pages if f"##{sub}" in p.page_content), "")
                    m = re.search(rf"(##{re.escape(sub)}[\s\S]*?)(?=^##\d+\.)",
                                  page_text, flags=re.MULTILINE)
                    block = m.group(1).strip() if m else page_text.strip()
                    st.text(block)
                st.write("---")

        # 7) TOP3 - QnA 청크
        qna_sub_map     = qna_substep_vectordbs.get(step, {})
        default_qna_vdb = qna_vectordbs.get(step)
        qna_vdb_for_sub = qna_sub_map.get(substep_option, default_qna_vdb)
        qna_scores      = qna_vdb_for_sub.similarity_search_with_score(query, k=3) if qna_vdb_for_sub else []
        with st.expander("2) TOP3 - QnA 청크", expanded=False):
            if not qna_scores:
                st.write("⚠️ 해당 서브스텝에 대한 Q&A가 없습니다.")
            for i, (doc, score) in enumerate(qna_scores, start=1):
                tag  = doc.metadata.get("tag", "")
                qc   = doc.metadata.get("question_context", "")
                ac   = doc.metadata.get("answer_context", "")
                st.markdown(f"**[TOP_{i}]. {tag} — Score {score:.2f}**")
                if qc: st.write(f"'{qc}'")
                if ac: st.write(f"[[[답변] '{ac}'")
                st.write("---")

        # 8) 답변 생성 (QnA 점수 ≥ 0.7 우선)
        if qna_scores and qna_scores[0][1] >= 0.7:
            top_doc, _ = qna_scores[0]
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(select_persona_prompt(qtype)),
                HumanMessagePromptTemplate.from_template(
                    """세부절차: {substep}
QnA 질문: {tag}
질문 내용:
{question_context}

답변 내용:
{answer_context}

사용자 질문: {question}

위 정보를 바탕으로 문장형으로 답변해 주세요."""
                )
            ])
            answer = LLMChain(
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                prompt=prompt
            ).predict(
                substep=substep_option,
                tag=top_doc.metadata["tag"],
                question_context=top_doc.metadata["question_context"],
                answer_context=top_doc.metadata["answer_context"],
                question=query
            )
        else:
            proc_vdb    = substep_vectordbs[step].get(substep_option)
            proc_scores = proc_vdb.similarity_search_with_score(query, k=1) if proc_vdb else []
            top_doc, _  = proc_scores[0] if proc_scores else (Document(page_content=""), 0)
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
            answer = LLMChain(
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                prompt=prompt
            ).predict(
                substep=substep_option,
                chunk=top_doc.page_content,
                question=query
            )

        # 9) 본문 응답
        st.markdown(f"## {substep_option}")
        st.write(answer)

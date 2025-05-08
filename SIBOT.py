import streamlit as st
st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")
import os
import requests
import tempfile
import re
import json
import difflib
from dotenv import load_dotenv
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from rank_bm25 import BM25Okapi

# ── Load environment & secrets ─────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🚨 OPENAI_API_KEY not found. Add it to Streamlit Secrets or .env.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── Constants for Google‑Drive IDs ──────────────────────────────────────────────
_PROCESS_DOC_ID = "1TNOhmUds7hMpwz3NO4QD-mO-J1sUJoEa"
_QNA_DOC_ID     = "17M1mnMZVl29EahbSVqzcyZEX8LYsx5ER"

# ── 1) Load & split Process document ────────────────────────────────────────────
@st.cache_resource
def load_and_split_process_docs():
    url = f"https://drive.google.com/uc?export=download&id={_PROCESS_DOC_ID}"
    resp = requests.get(url); resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(resp.content); tmp.flush()
        docs = PyMuPDFLoader(tmp.name).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)

# ── 2) Load & split Q&A document ───────────────────────────────────────────────
@st.cache_resource
def load_and_split_qna_docs():
    url = f"https://drive.google.com/uc?export=download&id={_QNA_DOC_ID}"
    resp = requests.get(url); resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(resp.content); tmp.flush()
        docs = PyMuPDFLoader(tmp.name).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)

# ── Load docs once ──────────────────────────────────────────────────────────────
process_docs = load_and_split_process_docs()
qna_docs     = load_and_split_qna_docs()

# ── Build vectorstores ─────────────────────────────────────────────────────────
embeddings         = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
process_vs         = FAISS.from_documents(process_docs, embeddings)
qna_vs             = FAISS.from_documents(qna_docs, embeddings)
process_retriever  = process_vs.as_retriever(search_kwargs={"k": 4})
qna_retriever      = qna_vs.as_retriever(search_kwargs={"k": 4})

# ── Streamlit page config ──────────────────────────────────────────────────────
st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")

# ── Load CSV hierarchy for overview (not used) ─────────────────────────────────
@st.cache_data
def load_hierarchy(csv_path):
    df = pd.read_csv(csv_path, dtype=str, encoding='utf-8-sig')
    rename = {
        '주요 단계':'step_name','주요 활동':'major','시기':'timing',
        '책임자':'owner','실무자':'worker','협조 및 지원 부서':'support',
        '적용 시스템':'system','문서 URL':'document_url'
    }
    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})
    if 'document_url' not in df.columns:
        df['document_url'] = ''
    return df

BASE_DIR = os.path.dirname(__file__)
# hierarchy_df = load_hierarchy(os.path.join(BASE_DIR, "SI_FULL_PROCESS_HIERARCHY.csv"))  # removed per user request

# ── Only Q&A tab ───────────────────────────────────────────────────────────────
st.header("Q&A")
query = st.text_input("질문 입력")
if st.button("질문하기") and query:
    # 1) 대표질문 JSON match
    try:
        rep = json.load(open(os.path.join(BASE_DIR,"vdc_a_대표질문.json"),encoding='utf-8'))
    except FileNotFoundError:
        rep = []
    if rep:
        norms = {re.sub(r'[^a-z0-9]','',e['question'].lower()):e for e in rep}
        key = re.sub(r'[^a-z0-9]','',query.lower())
        m = difflib.get_close_matches(key, norms.keys(),n=1,cutoff=0.7)
        if m:
            entry = norms[m[0]]
            st.subheader("🔎 대표 질문 매칭 답변")
            st.write(entry.get('answer',''))
            if entry.get('출처'): st.caption("출처: "+entry['출처'])
            st.stop()

    # 2) RetrievalQA on process docs
    prompt = PromptTemplate(
        input_variables=["context","question"],
        template="""
        당신은 SI 프로세스 문서 기반 질문에 답하는 어시스턴트입니다.

        [문서]
        {context}

        [질문]
        {question}

        [답변 형식]
        💡 핵심 요약
        📋 절차 또는 판단 주체
        """
    )
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini", temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=process_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt":prompt},
        return_source_documents=True
    )
    res = qa.invoke({"query":query})
    st.markdown(f"**💡 핵심 요약**\n{res['result']}")
    with st.expander("📎 문서 근거"):
        for i,doc in enumerate(res["source_documents"], start=1):
            st.write(f"[{i}] {doc.metadata.get('source_name','unknown')}")
            st.code(doc.page_content[:200] + "…")

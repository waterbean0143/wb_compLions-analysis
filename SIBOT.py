import streamlit as st
import pandas as pd
import os
import requests
import tempfile
import re
import json
import difflib
from dotenv import load_dotenv

# 0. load .env
load_dotenv()

# 1. load secrets/env
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY")  or os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = st.secrets.get("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY")

# 2. fail fast if missing
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Secrets 또는 .env 를 확인하세요.")
    st.stop()

# 3. now inject into environment
os.environ["OPENAI_API_KEY"]  = OPENAI_API_KEY
os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY

# 4. now it’s safe to import and instantiate LangChain models
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_upstage import UpstageGroundednessCheck
from rank_bm25 import BM25Okapi

# LangChain imports (after API key resolved)
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Google Drive 파일 ID 하드코딩
PROCESS_DOC_ID = "1TNOhmUds7hMpwz3NO4QD-mO-J1sUJoEa"
QNA_DOC_ID     = "17M1mnMZVl29EahbSVqzcyZEX8LYsx5ER"

# PDF 다운로드 함수
def download_gdrive_pdf(file_id: str, dst_path: str):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    r.raise_for_status()
    with open(dst_path, "wb") as f:
        f.write(r.content)

# 문서 로딩 및 분할
@st.cache_resource
def load_and_split(ids: list[str]):
    paths = []
    for fid in ids:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        download_gdrive_pdf(fid, tmp.name)
        paths.append(tmp.name)
    docs = []
    for p in paths:
        docs += PyMuPDFLoader(p).load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# 앱 초기화
st.set_page_config(page_title="SI 방법론 Q&A", layout="wide")
st.title("💬 SI 방법론 문서 기반 Q&A")

# 문서 로드·분할
process_docs = load_and_split([PROCESS_DOC_ID])
qna_docs     = load_and_split([QNA_DOC_ID])

# 벡터 DB 구축
@st.cache_resource
def build_vectorstores():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    process_vs = FAISS.from_documents(process_docs, embeddings)
    qna_vs     = FAISS.from_documents(qna_docs, embeddings)
    return process_vs, qna_vs

process_vs, qna_vs = build_vectorstores()
process_retriever = process_vs.as_retriever(search_kwargs={"k":5})
qna_retriever     = qna_vs.as_retriever(search_kwargs={"k":5})

# LLM 초기화 (모델명 고정)
llm = ChatOpenAI(model_name=DEFAULT_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0)

# 대화 이력
if "history" not in st.session_state:
    st.session_state.history = []

# 사용자 질문 입력
query = st.chat_input("질문을 입력하세요:")
if query:
    # 프로세스 문서 기반 답변
    proc_chain = RetrievalQA.from_chain_type(llm=llm, retriever=process_retriever, chain_type="stuff")
    ans1 = proc_chain.run(query)
    # 대표질문 문서 기반 답변
    qna_chain = RetrievalQA.from_chain_type(llm=llm, retriever=qna_retriever, chain_type="stuff")
    ans2 = qna_chain.run(query)
    st.session_state.history.append((query, ans1, ans2))

# 이력 렌더링
for q, a1, a2 in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").markdown(f"**[프로세스 문서]**\n{a1}")
    st.chat_message("assistant").markdown(f"**[대표질문 문서]**\n{a2}")

import streamlit as st
import os
import requests
import tempfile
import re
import json
import difflib
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_upstage import UpstageGroundednessCheck
from rank_bm25 import BM25Okapi

# 환경변수 로드
load_dotenv()

# 1) 읽기
OPENAI_API_KEY  = st.secrets.get("OPENAI_API_KEY")  or os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = st.secrets.get("UPSTAGE_API_KEY") or os.getenv("UPSTAGE_API_KEY")

# 2) 없는 경우 중단
if not OPENAI_API_KEY:
    st.error("🔑 OPENAI_API_KEY가 설정되지 않았습니다. Streamlit Secrets 또는 .env 를 확인하세요.")
    st.stop()
if not UPSTAGE_API_KEY:
    st.error("🔑 UPSTAGE_API_KEY가 설정되지 않았습니다. Streamlit Secrets 또는 .env 를 확인하세요.")
    st.stop()

# 3) 환경변수에 할당
os.environ["OPENAI_API_KEY"]  = OPENAI_API_KEY
os.environ["UPSTAGE_API_KEY"] = UPSTAGE_API_KEY

# Hard‑coded Google Drive IDs
_PROCESS_DOC_ID = "1TNOhmUds7hMpwz3NO4QD-mO-J1sUJoEa"
_QNA_DOC_ID     = "17M1mnMZVl29EahbSVqzcyZEX8LYsx5ER"

# 문서 로딩 및 분할: 프로세스
@st.cache_resource
def load_and_split_process_docs():
    url = f"https://drive.google.com/uc?export=download&id={_PROCESS_DOC_ID}"
    resp = requests.get(url)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(resp.content)
        tmp.flush()
        docs = PyMuPDFLoader(tmp.name).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)

# 문서 로딩 및 분할: 대표 Q&A
@st.cache_resource
def load_and_split_qna_docs():
    url = f"https://drive.google.com/uc?export=download&id={_QNA_DOC_ID}"
    resp = requests.get(url)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(resp.content)
        tmp.flush()
        docs = PyMuPDFLoader(tmp.name).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)

# 벡터스토어 구축
@st.cache_resource
def build_vectorstores():
    emb = OpenAIEmbeddings()
    process_docs = load_and_split_process_docs()
    qna_docs     = load_and_split_qna_docs()
    process_vs = FAISS.from_documents(process_docs, emb)
    qna_vs     = FAISS.from_documents(qna_docs,     emb)
    return process_vs, qna_vs

process_vs, qna_vs = build_vectorstores()
process_retriever = process_vs.as_retriever(search_kwargs={"k":4})
qna_retriever     = qna_vs.as_retriever(search_kwargs={"k":4})

# LLM 초기화 (모델명 고정)
llm = ChatOpenAI(model_name=DEFAULT_MODEL, openai_api_key=OPENAI_API_KEY, temperature=0)

# 대화 이력
if "history" not in st.session_state:
    st.session_state.history = []

# 사용자 질문 입력
query = st.chat_input("질문을 입력하세요:")
if query:
    proc_chain = RetrievalQA.from_chain_type(llm=llm, retriever=process_retriever, chain_type="stuff")
    ans1 = proc_chain.run(query)
    qna_chain = RetrievalQA.from_chain_type(llm=llm, retriever=qna_retriever, chain_type="stuff")
    ans2 = qna_chain.run(query)
    st.session_state.history.append((query, ans1, ans2))

# 이력 렌더링
for q, a1, a2 in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").markdown(f"**[프로세스 문서]**\n{a1}")
    st.chat_message("assistant").markdown(f"**[대표질문 문서]**\n{a2}")

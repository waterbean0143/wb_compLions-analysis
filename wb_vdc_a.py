import streamlit as st
import os
import json
import tempfile
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 환경 변수 로드 및 페이지 설정 (한 번만 호출)
load_dotenv()
st.set_page_config(
    page_title="VDC-A 사용자 프로필 기반 Q&A",
    page_icon="👤",
    layout="wide"
)

# 앱 제목
st.title("👤 사용자 프로필 기반 Q&A - VDC-A")

# 문서 로딩 함수
@st.cache_resource
def load_process_documents():
    url = "https://drive.google.com/uc?export=download&id=1lSEWk7KDgHR71yHcjEKniWzhqwL7T2fC"
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(response.content)
            docs = PyMuPDFLoader(f.name).load()
            for d in docs:
                d.metadata["source_name"] = "vdc_a_프로세스"
            return docs
    return []

# 문서 분할 및 벡터화
process_docs = load_process_documents()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
process_chunks = splitter.split_documents(process_docs)
process_retriever = FAISS.from_documents(process_chunks, OpenAIEmbeddings())

# LLM 및 프롬프트 설정
llm = ChatOpenAI(temperature=0)
prompt_template = (
    "당신은 VDC-A 프로세스 문서를 바탕으로 질문에 답하는 AI 어시스턴트입니다."
    "\n\n질문: {question}\n\n문서 내용: {context}\n"
)
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# 채팅 기록 초기화
if "history" not in st.session_state:
    st.session_state["history"] = []

# 사용자 질문 입력
query = st.text_input("질문을 입력하세요")
if query:
    with st.spinner("문서 기반 응답 생성 중..."):
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=process_retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": query})
        st.session_state["history"].append(
            (query, result["result"], result.get("source_documents", []))
        )

# 대화 및 근거 출력
for q, a, sources in st.session_state["history"]:
    st.chat_message("user").write(q)
    st.chat_message("assistant").markdown(f"### 💡 핵심 요약\n{a.strip()}")
    if sources:
        with st.expander("📎 문서 근거 보기"):
            for i, doc in enumerate(sources):
                name = doc.metadata.get("source_name", "unknown")
                content = doc.page_content or ""
                st.markdown(f"**[{i+1}]** `{name}`")
                st.code(content[:400] + ("..." if len(content) > 400 else ""))

# 추가 Q&A 로직은 필요 시 이 아래에 구현하세요.

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
from sklearn.metrics.pairwise import cosine_similarity

# 기본 사용자 프로필 (향후 확장 가능)
DEFAULT_PROFILE = {
    "소속": "AI이행2본부",
    "역할": "이행 PM",
    "사업": "기존 예정대로 VDC-A 절차를 밟는 사업"
}

# 환경 설정
load_dotenv()
st.set_page_config(page_title="VDC-A 사용자 프로필 기반 Q&A", page_icon="👤")
st.title("👤 사용자 프로필 기반 VDC-A Q&A")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

# QNA 로딩
with open("vdc_a_대표질문.json", "r", encoding="utf-8") as f:
    qna = json.load(f)
qna_questions = [q["question"] for q in qna]
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
qna_vectors = embeddings.embed_documents(qna_questions)

# 문서 로딩 및 벡터화
@st.cache_resource
def load_documents():
    urls = {
        "vdc_a_프로세스": "https://drive.google.com/uc?export=download&id=1cEFCFC7fp3JuDRgdPS3BdJuHPKhLF3yn",
        "vdc_a_qna": "https://drive.google.com/uc?export=download&id=1KGJv9ttGD7ErcSWymE-0jiMjOzbnq6iI"
    }
    all_docs = []
    for name, url in urls.items():
        response = requests.get(url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(response.content)
                f.flush()
                from langchain.document_loaders import PyMuPDFLoader
                docs = PyMuPDFLoader(f.name).load()
                for d in docs:
                    d.metadata["source_name"] = name
                all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(all_docs)

split_docs = load_documents()
vectordb = FAISS.from_documents(split_docs, embeddings)

# 사용자 프로필
if "user_profile" not in st.session_state:
    st.session_state["user_profile"] = DEFAULT_PROFILE.copy()

# 프롬프트 정의
profile_info = st.session_state["user_profile"]
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""당신은 다음 조건을 가진 사용자의 질문에 답하는 문서 기반 AI 어시스턴트입니다.

[사용자 조건]
- 소속: {profile_info['소속']}
- 역할: {profile_info['역할']}
- 사업 유형: {profile_info['사업']}

[질문]
{{question}}

[문서 내용]
{{context}}

[응답 형식]
💡 핵심 요약
📋 절차 또는 판단 주체
"""
)

# 질의 입력
if "history" not in st.session_state:
    st.session_state["history"] = []

query = st.chat_input("VDC-A 관련 질문을 입력하세요:")
if not query:
    st.stop()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o", openai_api_key=openai_api_key)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

with st.spinner("문서 기반 응답 생성 중..."):
    result = qa_chain.invoke({"query": query})
    st.session_state["history"].append((query, result["result"], result["source_documents"]))

# 출력
for q, a, sources in st.session_state["history"]:
    st.chat_message("user").write(q)
    st.chat_message("assistant").markdown(f"### 💡 핵심 요약\n{a.strip()}")
    with st.expander("📎 문서 근거 보기"):
        for i, doc in enumerate(sources):
            st.markdown(f"**[{i+1}]** `{doc.metadata.get('source_name', 'unknown')}`")
            st.code(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))

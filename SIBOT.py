import streamlit as st
import os
import tempfile
import requests
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# OpenAI API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 문서 다운로드 함수
def download_pdf_from_gdrive(gdrive_url, save_path):
    file_id = gdrive_url.split('/d/')[1].split('/')[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# 문서 벡터화 함수
def load_and_vectorize(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Streamlit 앱 시작
st.set_page_config(page_title="SIBOT - SI 방법론 Q&A", layout="wide")
st.title("📘 SIBOT: SI 방법론 문서 기반 Q&A 시스템")

# 탭 구성
tabs = st.tabs(["📖 소개", "🗂 프로세스 문서", "❓ 대표 질문", "💬 질문하기"])

with tabs[0]:
    st.markdown("""
    ## 🔍 소개
    SIBOT은 SI 방법론 전반에 걸친 문서 기반 Q&A 시스템입니다.
    Google Drive에서 문서를 자동으로 다운로드하고, 이를 벡터화하여 사용자 질문에 대한 답변을 제공합니다.
    """)

with tabs[1]:
    st.markdown("### 📥 프로세스 문서 다운로드 및 벡터화")
    process_url = st.text_input("Google Drive 프로세스 문서 URL을 입력하세요:")
    if st.button("프로세스 문서 로드"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            download_pdf_from_gdrive(process_url, tmp_file.name)
            st.session_state['process_vectorstore'] = load_and_vectorize(tmp_file.name)
            st.success("프로세스 문서가 성공적으로 로드되었습니다.")

with tabs[2]:
    st.markdown("### 📥 대표 질문 문서 다운로드 및 벡터화")
    qna_url = st.text_input("Google Drive 대표 질문 문서 URL을 입력하세요:")
    if st.button("대표 질문 문서 로드"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            download_pdf_from_gdrive(qna_url, tmp_file.name)
            st.session_state['qna_vectorstore'] = load_and_vectorize(tmp_file.name)
            st.success("대표 질문 문서가 성공적으로 로드되었습니다.")

with tabs[3]:
    st.markdown("### 💬 질문하기")
    user_question = st.text_input("질문을 입력하세요:")
    if st.button("질문하기"):
        if 'process_vectorstore' in st.session_state and 'qna_vectorstore' in st.session_state:
            # 프로세스 문서에서 검색
            process_qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
                chain_type="stuff",
                retriever=st.session_state['process_vectorstore'].as_retriever()
            )
            process_answer = process_qa.run(user_question)

            # 대표 질문 문서에서 검색
            qna_qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
                chain_type="stuff",
                retriever=st.session_state['qna_vectorstore'].as_retriever()
            )
            qna_answer = qna_qa.run(user_question)

            st.markdown("#### 📋 프로세스 문서 기반 답변")
            st.write(process_answer)

            st.markdown("#### 📋 대표 질문 문서 기반 답변")
            st.write(qna_answer)
        else:
            st.warning("먼저 프로세스 문서와 대표 질문 문서를 로드해주세요.")

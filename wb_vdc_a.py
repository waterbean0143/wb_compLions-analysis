import streamlit as st
import os
import json
import tempfile
import requests
import numpy as np
from dotenv import load_dotenv

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 1. 환경 설정 및 페이지 설정
load_dotenv()
st.set_page_config(
    page_title="VDC-A Multi-Doc Q&A",
    page_icon="🤖",
    layout="wide"
)

# 1. 환경 설정 및 페이지 설정
load_dotenv()
st.set_page_config(
    page_title="VDC-A Multi-Doc Q&A",
    page_icon="🤖",
    layout="wide"
)

# 2. 로그인 정보
users = {
    "admin":     {"password": "admin",     "name": "관리자"},
    "test":      {"password": "test",      "name": "테스트 사용자"},
    "10154371": {"password": "10154371", "name": "배수빈"},
    "10154372": {"password": "10154372", "name": "김도완"},
    "10156350": {"password": "10156350", "name": "박영준"},
    "10151647": {"password": "10151647", "name": "류주현"},
}

# 로그인 함수 정의

def check_password():
    def password_entered():
        user = st.session_state["username"]
        pw   = st.session_state["password"]
        if user in users and users[user]["password"] == pw:
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"]   = users[user]["name"]
            st.session_state["is_admin"] = (user == "admin")
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.markdown("---")
        return False
    elif not st.session_state.get("password_correct", False):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("❌ 잘못된 아이디 또는 비밀번호입니다.")
        del st.session_state["password_correct"]
        return False
    else:
        st.sidebar.success(f"안녕하세요, {st.session_state['logged_in_user']}님!")
        return True

# 로그인 체크
if not check_password():
    st.stop()

# 3. Embeddings & Prompt
embeddings = OpenAIEmbeddings()
llm_default = ChatOpenAI(temperature=0)
prompt_template = (
    "당신은 VDC-A 프로세스 및 대표질문 문서를 바탕으로 질문에 답하는 AI 어시스턴트입니다."
    "\n\n질문: {question}\n\n문서 내용: {context}\n"
)
prompt = PromptTemplate(input_variables=["context","question"], template=prompt_template)

# 4. 데이터 로드 및 FAISS 초기화
@st.cache_resource


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
# default LLM
# llm_default = ChatOpenAI(temperature=0)
prompt_template = (
    "당신은 VDC-A 프로세스 및 대표질문 문서를 바탕으로 질문에 답하는 AI 어시스턴트입니다."
    "\n\n질문: {question}\n\n문서 내용: {context}\n"
)
prompt = PromptTemplate(input_variables=["context","question"], template=prompt_template)

# 4. 데이터 로드 및 FAISS 초기화
@st.cache_resource
def init_faiss_from_pdf(url, chunk_size=300, chunk_overlap=50):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(response.content)
        docs = PyMuPDFLoader(f.name).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ).split_documents(docs)
    return FAISS.from_documents(chunks, embeddings), docs

@st.cache_resource
def init_faiss_from_json(path,chunk_size=300,chunk_overlap=50):
    items = json.load(open(path, encoding="utf-8"))
    docs = [Document(page_content=i['answer'], metadata={'question': i['question'], 'source': 'qna'}) for i in items]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ).split_documents(docs)
    return FAISS.from_documents(chunks, embeddings), docs

# 5. 데이터 로드
process_url = "https://drive.google.com/uc?export=download&id=1lSEWk7KDgHR71yHcjEKniWzhqwL7T2fC"
qna_json_path = "./vdc_a_대표질문.json"
faiss_proc, proc_docs = init_faiss_from_pdf(process_url)
faiss_qna, qna_docs = init_faiss_from_json(qna_json_path)

# 6. 역할 기반 UI
is_admin = st.session_state.get("is_admin", False)
if is_admin:
    k = st.number_input("Top-k Retrieval 수", min_value=1, max_value=10, value=4)
    temperature = st.slider("LLM 온도", 0.0, 1.0, 0.0)
else:
    k = 4
    temperature = 0.0

# 7. 탭 구성
tab1, tab2 = st.tabs(["프로세스 문서","대표질문 Q&A"])
for tab, (index, docs) in zip(
    [tab1, tab2],
    [(faiss_proc, proc_docs),(faiss_qna, qna_docs)]
):
    with tab:
        query = st.text_input("질문을 입력하세요", key=tab.title)
        if query:
            results = index.search(query, k=k)
            docs_res, scores = zip(*results)
            if is_admin:
                st.write("#### 검색 결과 (Score)")
                for idx, (d, s) in enumerate(results, 1):
                    st.write(f"{idx}. Score: {s:.4f}")
                    st.write(d.page_content[:200] + '...')
            llm = ChatOpenAI(temperature=temperature)
            chain = LLMChain(llm=llm, prompt=prompt)
            context = "\n\n".join([d.page_content for d in docs_res])
            answer = chain.run(context=context, question=query)
            st.markdown(f"### 💡 핵심 요약\n{answer.strip()}")
            with st.expander("📎 문서 근거 보기"):
                for idx, d in enumerate(docs_res, 1):
                    source = d.metadata.get('source','unknown')
                    st.markdown(f"**[{idx}]** `{source}`")
                    st.code(d.page_content[:400] + ('...' if len(d.page_content)>400 else ''))

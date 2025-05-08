import streamlit as st
import os
import json
import tempfile
import requests
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 1. 환경 설정 및 페이지 설정 (최상단에 한 번만 호출)
load_dotenv()
st.set_page_config(
    page_title="VDC-A Multi-Doc Q&A",
    page_icon="🤖",
    layout="wide"
)

# 2. 로그인 정보
users = {
    "admin": {"password": "admin", "name": "관리자"},
    "test": {"password": "test", "name": "테스트 사용자"},
    "10154371": {"password": "10154371", "name": "배수빈"},
    "10154372": {"password": "10154372", "name": "김도완"},
    "10156350": {"password": "10156350", "name": "박영준"},
    "10151647": {"password": "10151647", "name": "류주현"},
}

# 3. 로그인 함수 정의

def check_password():
    def password_entered():
        user = st.session_state["username"]
        pw = st.session_state["password"]
        if user in users and users[user]["password"] == pw:
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = users[user]["name"]
            st.session_state["is_admin"] = (user in ["admin", "test"])
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.markdown("---")
        return False
    if not st.session_state.get("password_correct", False):
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("❌ 잘못된 아이디 또는 비밀번호입니다.")
        del st.session_state["password_correct"]
        return False
    st.sidebar.success(f"안녕하세요, {st.session_state['logged_in_user']}님!")
    return True

# 로그인 체크
if not check_password():
    st.stop()

# 4. Embeddings & Prompt
embeddings = OpenAIEmbeddings()
prompt_template = (
    "당신은 VDC-A 프로세스 및 대표질문 문서를 바탕으로 질문에 답하는 AI 어시스턴트입니다."
    "\n\n질문: {question}\n\n문서 내용: {context}\n"
)
prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

# 5. 데이터 로드 및 FAISS 초기화 함수
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
    return FAISS.from_documents(chunks, embeddings)

@st.cache_resource
def init_faiss_from_json(path, chunk_size=300, chunk_overlap=50):
    items = json.load(open(path, encoding="utf-8"))
    docs = [Document(page_content=i['answer'], metadata={'question': i['question'], 'source': 'qna'}) for i in items]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    ).split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

# 6. FAISS 인덱스 생성
process_url = "https://drive.google.com/uc?export=download&id=1lSEWk7KDgHR71yHcjEKniWzhqwL7T2fC"
qna_json_path = "./vdc_a_대표질문.json"
faiss_proc = init_faiss_from_pdf(process_url)
faiss_qna = init_faiss_from_json(qna_json_path)

# 7. 대화 이력 초기화 (프로세스/대표질문 별, 최대 5개)
if "history_proc" not in st.session_state:
    st.session_state["history_proc"] = []
if "history_qna" not in st.session_state:
    st.session_state["history_qna"] = []

# 8. 기본 파라미터 설정
default_k = 4
default_temperature = 0.0
default_score_threshold = 0.0

# 9. 역할 및 설정 Tab 구성
tabs = ["프로세스 문서", "대표질문 Q&A"]
if st.session_state.get("is_admin", False):
    tabs.append("설정")
tab1, tab2, *rest = st.tabs(tabs)

# 10. 설정 탭에서는 파라미터 확인 및 조정
if st.session_state.get("is_admin", False):
    tab_set = rest[0]
    with tab_set:
        st.header("⚙️ 관리자 설정")
        st.markdown(f"- 기본 Top-k: **{default_k}**")
        st.markdown(f"- 기본 LLM 온도: **{default_temperature}**")
        st.markdown(f"- 기본 Score Threshold: **{default_score_threshold}**")
        k = st.number_input("Top-k Retrieval 수", min_value=1, max_value=20, value=default_k)
        temperature = st.slider("LLM 온도", 0.0, 1.0, default_temperature)
        score_threshold = st.slider("Score 필터 기준", 0.0, 1.0, default_score_threshold)
else:
    k, temperature, score_threshold = default_k, default_temperature, default_score_threshold

# 11. 각 탭별 채팅 인터페이스
for tab, history_key, store in [
    (tab1, "history_proc", faiss_proc),
    (tab2, "history_qna", faiss_qna),
]:
    with tab:
        # 과거 대화 표시
        for q, a in st.session_state[history_key]:
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(a)

        # 질문 입력 폼
        with st.form(f"form_{history_key}"):
            new_query = st.text_input("질문을 입력하세요", key=f"input_{history_key}")
            submitted = st.form_submit_button("전송")

        if submitted and new_query:
            # 벡터 검색
            results = store.search(new_query, k=k)
            # score threshold 적용
            filtered = [(d, s) for d, s in results if s >= score_threshold] or results
            docs_res, scores = zip(*filtered)

            # 관리자: 검색 결과 스코어 표시
            if st.session_state.get("is_admin", False):
                st.write("#### 검색 결과 (Score)")
                for idx, (d, s) in enumerate(filtered, 1):
                    st.write(f"{idx}. Score: {s:.4f}")
                    st.write(d.page_content[:200] + '...')

            # LLM 답변 생성
            llm = ChatOpenAI(temperature=temperature)
            chain = LLMChain(llm=llm, prompt=prompt)
            context = "\n\n".join([d.page_content for d in docs_res])
            answer = chain.run(context=context, question=new_query)

            # 대화 이력 업데이트 및 표시
            st.session_state[history_key].append((new_query, answer))
            st.session_state[history_key] = st.session_state[history_key][-5:]
            # 입력 필드 초기화
            st.session_state[f"input_{history_key}"] = ""

            st.chat_message("user").write(new_query)
            st.chat_message("assistant").write(answer)

            # 근거 보기
            with st.expander("📎 문서 근거 보기"):
                for idx, d in enumerate(docs_res, 1):
                    source = d.metadata.get('source', 'unknown')
                    st.markdown(f"**[{idx}]** `{source}`")
                    st.code(d.page_content[:400] + ('...' if len(d.page_content) > 400 else ''))

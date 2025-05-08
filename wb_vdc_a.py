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

# 1. 환경 설정 및 페이지 설정
load_dotenv()
st.set_page_config(
    page_title="SI 프로세스 Q&A",
    page_icon="🤖",
    layout="wide"
)

# 2. 로그인 로직
users = {
    "admin": {"password": "admin", "name": "관리자"},
    "test": {"password": "test", "name": "테스트 사용자"},
    # 기타 사용자 추가
}

def check_password():
    def login():
        u = st.session_state["username"]
        p = st.session_state["password"]
        if u in users and users[u]["password"] == p:
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = users[u]["name"]
            # admin/test 계정만 관리자 권한
            st.session_state["is_admin"] = (u in ["admin", "test"])
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=login)
        st.stop()
    elif not st.session_state.get("password_correct", False):
        st.error("❌ 잘못된 아이디 또는 비밀번호입니다.")
        del st.session_state["password_correct"]
        st.stop()
    else:
        st.sidebar.success(f"안녕하세요, {st.session_state['logged_in_user']}님!")

check_password()

# 3. 환경 변수 및 모델 설정
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 환경변수로 모델 선택 가능
# 4. SI 프로세스 정의 로드
BASE_DIR = os.path.dirname(__file__)
si = json.load(open(os.path.join(BASE_DIR, "SI_PROCESS_FULL.json"), encoding="utf-8"))
steps = sorted({item["step"] for item in si.get("4_Main_Process", [])})

# 4. Embeddings & Prompt 설정
embeddings = OpenAIEmbeddings()
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=("당신은 SI 프로세스 문서를 바탕으로 질문에 답하는 AI 어시스턴트입니다."
              "\n\n질문: {question}\n\n문서 내용: {context}\n")
)

# 5. FAISS 인덱스 초기화 함수
@st.cache_resource
def build_index_from_pdf(source):
    # source가 URL이면 다운로드, 아니면 로컬 파일
    if source.startswith("http"):
        resp = requests.get(source)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(resp.content)
            path = f.name
    else:
        path = os.path.join(BASE_DIR, source)
    docs = PyMuPDFLoader(path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

@st.cache_resource
def build_index_from_json(path):
    full = os.path.join(BASE_DIR, path)
    items = json.load(open(full, encoding="utf-8"))
    docs = [Document(page_content=i['answer'], metadata={'question':i['question'],'source':'QnA'}) for i in items]
    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)

# 6. 문서 소스 인덱스 생성
# VDC-A 예시: 프로세스 PDF와 QnA JSON은 git에 이미 있음
proc_source = "https://drive.google.com/uc?export=download&id=1lSEWk7KDgHR71yHcjEKniWzhqwL7T2fC"
qna_source  = "vdc_a_대표질문.json"
faiss_proc = build_index_from_pdf(proc_source)
faiss_qna  = build_index_from_json(qna_source)

# 7. 대화 이력 초기화 (단계별 최대 5개)
for step in steps:
    key = f"history_{step}"
    if key not in st.session_state:
        st.session_state[key] = []

# 8. 관리자 파라미터 기본값
default_k = 4
default_temp = 0.0
default_thr = 0.0

# 9. 탭 구성 (소개, 개요, Q&A, [설정])
tabs = ["소개", "절차 개요", "Q&A"]
if st.session_state.get("is_admin", False):
    tabs.append("설정")
intro_tab, overview_tab, qa_tab, *rest = st.tabs(tabs)

# 소개 탭
with intro_tab:
    st.header("SI 프로세스 전체개요")
    st.info("Flowchart는 나중에 draw.io 파일 기반으로 렌더링 예정입니다.")

# 절차 개요 탭
with overview_tab:
    st.header("절차별 개요")
    step_tabs = st.tabs(steps)
    for step, tab in zip(steps, step_tabs):
        with tab:
            st.subheader(step)
            for act in [i for i in si.get("4_Main_Process", []) if i["step"]==step]:
                st.markdown(f"**활동**: {act['activity']}")
                st.markdown(f"- 시기: {act.get('timing','-')}")
                st.markdown(f"- 책임자: {', '.join(act.get('owner',[]))}")
                st.markdown(f"- 수행자: {', '.join(act.get('worker',[]))}")
                st.markdown(f"- 지원: {act.get('support','-')}")
                st.markdown(f"- 시스템: {act.get('system','-')}")

# 설정 탭 (관리자 전용)
if st.session_state.get("is_admin", False):
    set_tab = rest[0]
    with set_tab:
        st.header("⚙️ 관리자 설정")
        st.write(f"- 기본 Top-k: **{default_k}**")
        st.write(f"- 기본 온도: **{default_temp}**")
        st.write(f"- Score 임계: **{default_thr}**")
        k = st.number_input("Top-k Retrieval", 1, 20, default_k)
        temp = st.slider("LLM 온도", 0.0, 1.0, default_temp)
        thr = st.slider("Score Threshold", 0.0, 1.0, default_thr)
else:
    k, temp, thr = default_k, default_temp, default_thr

# 10. Q&A 탭
with qa_tab:
    st.header("Q&A")
    sel = st.selectbox("단계 선택", steps)
    hkey = f"history_{sel}"

    # 과거 대화
    for q,a in st.session_state[hkey]:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    # 질문 폼
    with st.form("form_qna"):
        q = st.text_input("질문을 입력하세요", key="input_qna")
        submitted = st.form_submit_button("전송")
    if submitted and q:
        # 소스 결정: VDC-A 단계만 QnA도 확인
        sources = [("프로세스문서", faiss_proc)]
        if sel == "제안/계약":
            sources.append(("대표질문QnA", faiss_qna))

        # 벡터 검색 후 필터링
        all_res = []
        for label, store in sources:
            for d,s in store.search(q, k=k):
                d.metadata['source'] = label
                if s >= thr:
                    all_res.append((d,s))
        if not all_res:
            all_res = [(d,s) for label, store in sources for d,s in store.search(q, k=k)]
        all_res = sorted(all_res, key=lambda x: x[1], reverse=True)[:k]

        docs_res, scores = zip(*all_res)

        # LLMChain 답변 (GPT-4o-mini 사용)
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temp)
        chain = LLMChain(llm=llm, prompt=prompt)
        ctx = "\n\n".join(d.page_content for d in docs_res)
        ans = chain.run(context=ctx, question=q)

        # 히스토리 업데이트
        st.session_state[hkey].append((q,ans))
        st.session_state[hkey] = st.session_state[hkey][-5:]
        st.session_state["input_qna"] = ""

        # 출력
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(ans)
        with st.expander("📎 문서 근거 보기"):
            for idx,d in enumerate(docs_res,1):
                st.markdown(f"**[{idx}]** `{d.metadata['source']}`")
                st.code(d.page_content[:300] + ('...' if len(d.page_content)>300 else ''))

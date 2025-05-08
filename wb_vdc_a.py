import streamlit as st
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

# ——— 1. 페이지 설정 ———
load_dotenv()
st.set_page_config(
    page_title="SI 프로세스 Q&A",
    page_icon="🤖",
    layout="wide"
)

# ——— 2. 로그인 로직 ———
users = {
    "admin": {"password": "admin", "name": "관리자"},
    "test": {"password": "test", "name": "테스트 사용자"},
    # 추가 사용자...
}

def check_password():
    def login():
        u = st.session_state["username"]
        p = st.session_state["password"]
        if u in users and users[u]["password"] == p:
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = users[u]["name"]
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
        st.error("❌ 아이디 또는 비밀번호가 잘못되었습니다.")
        del st.session_state["password_correct"]
        st.stop()
    else:
        st.sidebar.success(f"안녕하세요, {st.session_state['logged_in_user']}님!")

check_password()

# ——— 3. 리소스 설정 ———
# Load SI process definition
si = json.load(open("/mnt/data/SI_PROCESS_FULL.json", encoding="utf-8"))
steps = sorted({item["step"] for item in si.get("4_Main_Process", [])})

# Embeddings & Prompt
emb = OpenAIEmbeddings()
prompt = PromptTemplate(
    input_variables=["context","question"],
    template=("당신은 SI 프로세스 문서를 바탕으로 질문에 답하는 AI 어시스턴트입니다."
              "\n\n질문: {question}\n\n문서 내용: {context}\n")
)

# FAISS 초기화 함수
def build_index_from_pdf(url):
    resp = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(resp.content)
        docs = PyMuPDFLoader(f.name).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
    return FAISS.from_documents(chunks, emb)

def build_index_from_json(path):
    items = json.load(open(path, encoding="utf-8"))
    docs = [Document(page_content=i['answer'], metadata={'question':i['question'],'source':'QnA'}) for i in items]
    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
    return FAISS.from_documents(chunks, emb)

# 예시 VDC-A 자료 인덱스 (추후 config 로 대체)
proc_url = "https://drive.google.com/uc?export=download&id=1lSEWk7KDgHR71yHcjEKniWzhqwL7T2fC"
qna_path = "./vdc_a_대표질문.json"
faiss_proc = build_index_from_pdf(proc_url)
faiss_qna = build_index_from_json(qna_path)

# ——— 4. 세션 상태 초기화 ———
for step in steps:
    key = f"history_{step}"
    if key not in st.session_state:
        st.session_state[key] = []

# 관리자 파라미터 defaults
default_k = 4
default_temp = 0.0
default_score_thr = 0.0

# ——— 5. 탭 구성 ———
main_tabs = ["소개","절차 개요","Q&A"]
if st.session_state.get("is_admin", False):
    main_tabs.append("설정")
intro_tab, overview_tab, qa_tab, *rest = st.tabs(main_tabs)

# — 소개 탭 —
with intro_tab:
    st.header("SI 프로세스 개요")
    st.info("Flowchart 파일을 나중에 업로드하면 여기서 렌더링됩니다.")

# — 절차 개요 탭 —
with overview_tab:
    st.header("절차별 개요")
    step_tabs = st.tabs(steps)
    for step, tab in zip(steps, step_tabs):
        with tab:
            st.subheader(step)
            activities = [i for i in si.get("4_Main_Process", []) if i["step"]==step]
            for act in activities:
                st.markdown(f"**활동**: {act['activity']}")
                st.markdown(f"- 시기: {act.get('timing','-')}")
                st.markdown(f"- 책임자: {', '.join(act.get('owner',[]))}")
                st.markdown(f"- 수행자: {', '.join(act.get('worker',[]))}")
                st.markdown(f"- 지원: {act.get('support','-')}")
                st.markdown(f"- 시스템: {act.get('system','-')}")

# — 설정 탭 (관리자) —
if st.session_state.get("is_admin", False):
    set_tab = rest[0]
    with set_tab:
        st.header("⚙️ 관리자 설정")
        st.write(f"기본 Top-k: **{default_k}**")
        st.write(f"기본 온도: **{default_temp}**")
        st.write(f"스코어 임계: **{default_score_thr}**")
        k = st.number_input("Top-k", 1, 20, default_k)
        temp = st.slider("LLM 온도", 0.0, 1.0, default_temp)
        thr = st.slider("스코어 임계", 0.0, 1.0, default_score_thr)
else:
    k, temp, thr = default_k, default_temp, default_score_thr

# — Q&A 탭 —
with qa_tab:
    st.header("질문 및 답변")
    sel = st.selectbox("단계 선택", steps)
    hist_key = f"history_{sel}"

    # 과거 대화
    for q,a in st.session_state[hist_key]:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    # 질문 입력 폼
    with st.form("form_qna"):
        q = st.text_input("질문을 입력하세요", key="input_qna")
        submitted = st.form_submit_button("전송")
    if submitted and q:
        # VDC-A 대비 예시 처리
        sources = []
        if sel == "제안/계약":
            sources = [("프로세스문서", faiss_proc), ("대표질문QnA", faiss_qna)]
        else:
            sources = [("프로세스문서", faiss_proc)]
        # 벡터 검색
        all_res = []
        for label, store in sources:
            for d,s in store.search(q, k=k):
                d.metadata['source'] = label
                all_res.append((d,s))
        # 정렬/필터
        all_res.sort(key=lambda x: x[1], reverse=True)
        filtered = [rs for rs in all_res if rs[1]>=thr] or all_res[:k]
        docs_res, scores = zip(*filtered)

        # LLM 답변
        llm = ChatOpenAI(temperature=temp)
        chain = LLMChain(llm=llm, prompt=prompt)
        context = "\n\n".join([d.page_content for d in docs_res])
        ans = chain.run(context=context, question=q)

        # 히스토리 업데이트
        st.session_state[hist_key].append((q,ans))
        st.session_state[hist_key] = st.session_state[hist_key][-5:]
        st.session_state["input_qna"] = ""

        # 대화 출력
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(ans)

        # 근거 보기
        with st.expander("📎 문서 근거 보기"):
            for idx,d in enumerate(docs_res,1):
                st.markdown(f"**[{idx}]** `{d.metadata['source']}`")
                st.code(d.page_content[:300] + ('...' if len(d.page_content)>300 else ''))

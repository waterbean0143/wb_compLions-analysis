import os
import streamlit as st

# ─────────────────────────────────────────────────────
# 1) 기본 라이브러리 임포트
# ─────────────────────────────────────────────────────
import re
from io import BytesIO
from kiwipiepy import Kiwi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import RetrievalQA
from langchain.schema import Document
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# ─────────────────────────────────────────────────────
# 2) 글로벌 변수 및 설정
# ─────────────────────────────────────────────────────
executor = ThreadPoolExecutor(max_workers=5)
bm25_weight = 0.3
faiss_weight = 0.7

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="AX SI 방법론 이행봇", page_icon="🤖", layout='wide')

# ─────────────────────────────────────────────────────
# 3) Secrets에서 키 가져와 환경변수에 세팅
# ─────────────────────────────────────────────────────
os.environ["OPENAI_API_KEY"]      = st.secrets["openai"]["api_key"]
st.write("🔑 OPENAI_API_KEY env:", os.getenv("OPENAI_API_KEY"))
os.environ["UPSTAGE_API_KEY"]     = st.secrets["upstage"]["api_key"]
os.environ["LANGCHAIN_API_KEY"]   = st.secrets["langchain"]["api_key"]
os.environ["LANGCHAIN_ENDPOINT"]  = st.secrets["langchain"]["endpoint"]
os.environ["LANGCHAIN_PROJECT"]   = st.secrets["langchain"]["project"]
os.environ["LANGCHAIN_TRACING_V2"]= st.secrets["langchain"]["tracing_v2"]
os.environ["LANGSMITH_API_KEY"]   = st.secrets["langsmith"]["api_key"]

# ─────────────────────────────────────────────────────
# 4) 로그인
# ─────────────────────────────────────────────────────
users = {
    "10154371": {"password": "10154371", "name": "배수빈"},
    "10154372": {"password": "10154372", "name": "김도완"},
    "10156350": {"password": "10156350", "name": "박영준"},
}
if 'logged_in' not in st.session_state:
    st.sidebar.title("🔒 로그인")
    uid = st.sidebar.text_input("ID", key="login_id")
    pwd = st.sidebar.text_input("PW", type="password", key="login_pw")
    if st.sidebar.button("로그인"):
        if uid in users and users[uid]['password'] == pwd:
            st.session_state['logged_in'] = True
            st.session_state['user_id']   = uid
            st.sidebar.success(f"환영합니다, {users[uid]['name']}님!")
        else:
            st.sidebar.error("ID 또는 비밀번호가 올바르지 않습니다.")
    st.stop()

# ─────────────────────────────────────────────────────
# 5) UI 설정
# ─────────────────────────────────────────────────────
st.sidebar.title("⚙️ 설정")
answer_mode = st.sidebar.radio("답변 모드 선택", ['빠른 답변', '정확한 답변'], index=0)

tabs = st.tabs(["Q&A", "Feedback", "사례관리"])
qa_tab, fb_tab, case_tab = tabs

# ─────────────────────────────────────────────────────
# 6) PDF URL 매핑
# ─────────────────────────────────────────────────────
PROCESS_PDF_URLS = {
    "제안/계약": "https://drive.google.com/uc?export=download&id=1TNOhmUds7hMpwz3NO4QD-mO-J1sUJoEa",
}
QNA_PDF_URLS = {
    "제안/계약": "https://drive.google.com/uc?export=download&id=17M1mnMZVl29EahbSVqzcyZEX8LYsx5ER",
}

# ─────────────────────────────────────────────────────
# 7) 형태소 분석기 & 전처리 함수
# ─────────────────────────────────────────────────────
kiwi = Kiwi()
def preprocess(text: str) -> str:
    res = kiwi.analyze(text)
    tokens = [t.form for t in res[0][0] if t.tag.startswith(('N','V','MA'))]
    return ' '.join(tokens)

# ─────────────────────────────────────────────────────
# 8) Q&A 탭 구현
# ─────────────────────────────────────────────────────
with qa_tab:
    st.header("AX SI 방법론 이행봇")
    st.subheader("📂 절차 선택 및 Q&A")

    # 8.1 절차 단계 선택
    steps = list(PROCESS_PDF_URLS.keys())
    step  = st.selectbox("📂 절차 단계 선택", steps)

    # 8.2 PDF 링크
    st.markdown("**관련 프로세스 문서(PDF)**")
    st.markdown(f"- [프로세스 PDF]({PROCESS_PDF_URLS[step]})")
    st.markdown(f"- [Q&A PDF]({QNA_PDF_URLS.get(step,'')})")

    # 8.3 PDF 로딩 & 청크 분할
    splitter    = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    proc_loader = PyMuPDFLoader(PROCESS_PDF_URLS[step])
    proc_docs   = splitter.split_documents(proc_loader.load())
    qna_loader  = PyMuPDFLoader(QNA_PDF_URLS[step])
    qna_docs    = splitter.split_documents(qna_loader.load())

    # 8.4 전처리
    for d in proc_docs + qna_docs:
        d.page_content = preprocess(d.page_content)

    # 8.5 하이브리드 리트리버 초기화 (파라미터 없이)
    @st.cache_resource
    def init_retriever():
        emb    = OpenAIEmbeddings(
                    model="text-embedding-ada-002",
                    openai_api_key=st.secrets["openai"]["api_key"]
                )
        docs   = proc_docs + qna_docs
        faiss_db = FAISS.from_documents(docs, emb)
        bm25_db  = BM25Retriever(documents=docs)
        return EnsembleRetriever(
            retrievers=[bm25_db, faiss_db],
            weights=[bm25_weight, faiss_weight]
        )

    retriever = init_retriever()

    # 8.6 RetrievalQA 체인 생성
    qa_chain = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=st.secrets["openai"]["api_key"]
    )
    qa = RetrievalQA.from_chain_type(
        llm=retriever and qa_chain,  # retriever와 llm 인자를 순서대로 넘겨 줍니다
        chain_type="stuff",
        retriever=retriever
    )

    # 8.7 사용자 질문 입력 & 답변
    query = st.text_input("💬 질문을 입력하세요", key="proc_query")
    if query:
        with st.spinner("답변 생성 중…"):
            answer = qa.run(query)
        st.markdown("**답변:**")
        st.write(answer)

# ─────────────────────────────────────────────────────
# 9) Feedback 탭
# ─────────────────────────────────────────────────────
with fb_tab:
    st.header("📝 Feedback")
    st.write("Feedback 기능은 추후 구현 예정입니다.")

# ─────────────────────────────────────────────────────
# 10) 사례관리 탭
# ─────────────────────────────────────────────────────
with case_tab:
    st.header("📂 사례관리 (SI Q&A)")
    st.write("사례관리 기능은 추후 구현 예정입니다.")

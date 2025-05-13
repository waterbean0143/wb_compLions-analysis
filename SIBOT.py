import streamlit as st
import gdown
import pandas as pd
import re
from datetime import datetime, timezone, timedelta
from io import BytesIO
from kiwipiepy import Kiwi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.document_transformers import LongContextReorder
from sklearn.metrics.pairwise import cosine_similarity
from langchain_upstage import UpstageGroundednessCheck
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import json

# -----------------------------
# 글로벌 선언 및 초기화
# -----------------------------
global memory
global si_qna_vectordbs, si_qna_docs
global for_show_si_process_vectordbs
global si_process_docs, si_process_vectordbs
global similar_cases_db, fcpa_retrievers
global si_process_retrievers, si_qna_retrievers

similar_cases_db = {}
fcpa_retrievers = {}
executor = ThreadPoolExecutor(max_workers=5)

# FCPA (미사용)
fcpa_docs = []
fcpa_vectordbs = {}

# SI 프로세스 문서 및 벡터DB
si_process_docs = []
si_process_vectordbs = {}

# SI Q&A 문서 및 벡터DB
si_qna_docs = []
si_qna_vectordbs = {}

# 화면 표시용
for_show_si_process_vectordbs = {}
selected_for_show_si_process_vectordbs = {}

# 리트리버
si_process_retrievers = {}
si_qna_retrievers = {}

# 하이브리드 가중치
bm25_weight = 0.3
faiss_weight = 0.7

# 한글 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(page_title="AX SI 방법론 이행봇", page_icon="🤖", layout='wide')

# -----------------------------
# 1) 로그인
# -----------------------------
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
            st.session_state['user_id'] = uid
            st.sidebar.success(f"환영합니다, {users[uid]['name']}님!")
        else:
            st.sidebar.error("ID 또는 비밀번호가 올바르지 않습니다.")
    st.stop()

# -----------------------------
# 2) UI 설정
# -----------------------------
st.sidebar.title("⚙️ 설정")
answer_mode = st.sidebar.radio("답변 모드 선택", ['빠른 답변', '정확한 답변'], index=0)

tabs = st.tabs(["Q&A", "Feedback", "사례관리"])
qa_tab, fb_tab, case_tab = tabs

# -----------------------------
# 3) 데이터 로드
# -----------------------------
PROCESS_INDEX_CSV_URLS = {
    "전체 절차 개요": "https://drive.google.com/uc?export=download&id=19Qj33WiDioAlu58fr-UjO9IZmhsoI0Eb"
}
# PDF URL 매핑
PROCESS_PDF_URLS = {
    "제안/계약": "https://drive.google.com/uc?export=download&id=1TNOhmUds7hMpwz3NO4QD-mO-J1sUJoEa",
}
QNA_PDF_URLS = {
    "제안/계약": "https://drive.google.com/uc?export=download&id=17M1mnMZVl29EahbSVqzcyZEX8LYsx5ER",
}

@st.cache_data
def load_csv():
    # PDF 매핑하신 것처럼
    url = PROCESS_CSV_URLS["전체 절차 개요"]
    local_path = "SI_FULL_PROCESS_HIERARCHY.csv"
    gdown.download(url, local_path, quiet=True)

    # TSV + CP949 로드
    df = pd.read_csv(local_path, sep="\t", encoding="cp949", engine="python")
    df.columns = df.columns.str.replace("\ufeff","").str.strip()
    return df

df = load_csv()

# 계층 구조 파싱
def parse_number(s: str) -> str:
    m = re.match(r"^(\d+(?:\.\d+)*)", s)
    return m.group(1) if m else ''
df['번호'] = df['주요 활동'].apply(parse_number)

# 트리 빌드
tree: dict = {}
for step, sub in df.groupby('주요 단계'):
    nodes: dict = {}
    for _, row in sub.iterrows():
        num = row['번호']
        title = row['주요 활동']
        if num.count('.') == 1 and num != '':
            parent = num.split('.')[0]
            parent_key = parent + '.'
            if parent_key in nodes:
                nodes[parent_key]['children'].append({'num': num, 'title': title})
        else:
            key = num + '.' if num else title
            nodes[key] = {'num': num, 'title': title, 'children': []}
    tree[step] = nodes

# 형태소 분석기 초기화
kiwi = Kiwi()
def preprocess(text: str) -> str:
    res = kiwi.analyze(text)
    tokens = [t.form for t in res[0][0] if t.tag.startswith(('N','V','MA'))]
    return ' '.join(tokens)

# -----------------------------
# 4) Q&A 탭 구현
# -----------------------------
with qa_tab:
    st.header("AX SI 방법론 이행봇")
    st.subheader("📋 전체 SI 프로세스 목록")
    st.dataframe(df[['주요 단계','주요 활동','시기','책임자','실무자','협조 및 지원 부서','적용 시스템']])

    # 단계 선택
    step = st.selectbox("📂 절차 단계 선택", list(tree.keys()))

    # 최상위 활동 선택
    tops = list(tree[step].keys())
    top_choice = st.selectbox("📝 주요 활동 선택", tops, format_func=lambda k: tree[step][k]['title'])

    # 하위 활동 선택
    children = tree[step][top_choice]['children']
    if children:
        sub_choice = st.selectbox(
            "🔹 세부 활동 선택", [c['num'] for c in children],
            format_func=lambda num: next(c['title'] for c in children if c['num']==num)
        )
        prefix = sub_choice
    else:
        prefix = top_choice.rstrip('.')

    # PDF 링크 표시
    st.markdown("**관련 프로세스 문서(PDF)**")
    if step in PROCESS_PDF_URLS:
        st.markdown(f"- [프로세스 PDF]({PROCESS_PDF_URLS[step]})")
    if step in QNA_PDF_URLS:
        st.markdown(f"- [Q&A PDF]({QNA_PDF_URLS[step]})")

    # 문서 로딩 및 분할
    proc_loader = PyMuPDFLoader(PROCESS_PDF_URLS[step])
    proc_pages = proc_loader.load()
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    proc_docs = splitter.split_documents(proc_pages)

    qna_loader = PyMuPDFLoader(QNA_PDF_URLS[step])
    qna_pages = qna_loader.load()
    qna_docs = splitter.split_documents(qna_pages)

    # CSV 문서 필터링
    mask = df['번호'].astype(str).str.startswith(prefix)
    csv_docs = []
    for _, row in df[mask].iterrows():
        content = '\n'.join(f"{col} : {row[col]}" for col in df.columns)
        csv_docs.append(Document(page_content=content, metadata={'step':step,'num':row['번호']}))

    # 전체 문서 결합 및 전처리
    all_docs = proc_docs + qna_docs + csv_docs
    for d in all_docs:
        d.page_content = preprocess(d.page_content)

    # 리트리버 초기화
    @st.cache_resource
    def init_retriever(docs):
        emb = OpenAIEmbeddings(model='gpt-4o-mini')
        faiss = FAISS.from_documents(docs, emb)
        bm25 = BM25Retriever(documents=docs)
        ens = EnsembleRetriever(retrievers=[bm25, faiss], weights=[bm25_weight, faiss_weight])
        return ens

    retriever = init_retriever(all_docs)
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model='gpt-4o-mini', temperature=0),
        chain_type='stuff',
        retriever=retriever
    )

    # 질의 응답
    query = st.text_input("💬 질문을 입력하세요")
    if query:
        with st.spinner("답변 생성 중…"):
            answer = qa_chain.run(query)
        st.markdown("**답변:**")
        st.write(answer)

# -----------------------------
# 5) Feedback 탭
# -----------------------------
with fb_tab:
    st.header("📝 Feedback")
    st.write("Feedback 기능은 추후 구현 예정입니다.")

# -----------------------------
# 6) 사례관리 탭
# -----------------------------
with case_tab:
    st.header("📂 사례관리 (SI Q&A)")
    st.write("사례관리 기능은 추후 구현 예정입니다.")

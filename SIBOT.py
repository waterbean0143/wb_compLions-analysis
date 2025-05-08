import streamlit as st
import os
import requests
import tempfile

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# 1) 하드코딩된 Google Drive 파일 ID
PROCESS_DOC_ID = "1TNOhmUds7hMpwz3NO4QD-mO-J1sUJoEa"
QNA_DOC_ID     = "17M1mnMZVl29EahbSVqzcyZEX8LYsx5ER"

# 2) PDF를 GDrive에서 내려받아 임시파일에 저장
def download_gdrive_pdf(file_id: str, dst_path: str):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url)
    r.raise_for_status()
    with open(dst_path, "wb") as f:
        f.write(r.content)

# 3) 문서 로딩 + 청크 분할
@st.cache_resource
def load_and_split(ids: list[str]):
    paths = []
    for fid in ids:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        download_gdrive_pdf(fid, tmp.name)
        paths.append(tmp.name)
    docs = []
    for p in paths:
        docs += PyMuPDFLoader(p).load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# 4) 벡터 DB 생성
@st.cache_resource
def build_faiss(docs):
    emb = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.from_documents(docs, emb)

# 애플리케이션 시작
st.set_page_config(page_title="SIBOT Q&A", layout="wide")
st.title("💬 SI 방법론 문서 기반 Q&A")

# 5) 앱 기동 시 자동으로 문서 로드·벡터화
process_docs = load_and_split([PROCESS_DOC_ID])
qna_docs     = load_and_split([QNA_DOC_ID])

process_vs = build_faiss(process_docs)
qna_vs     = build_faiss(qna_docs)

process_retriever = process_vs.as_retriever(search_kwargs={"k":5})
qna_retriever     = qna_vs.as_retriever(search_kwargs={"k":5})

# 항상 gpt-4o-mini 사용하도록 model_name 고정
llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0)

# 채팅 이력 저장
if "history" not in st.session_state:
    st.session_state.history = []

# 사용자 입력
query = st.chat_input("질문을 입력하세요:")
if query:
    # 프로세스 문서 기반 QA
    proc_chain = RetrievalQA.from_chain_type(llm=llm, retriever=process_retriever, chain_type="stuff")
    ans_proc = proc_chain.run(query)
    # 대표질문 문서 기반 QA
    qna_chain = RetrievalQA.from_chain_type(llm=llm, retriever=qna_retriever, chain_type="stuff")
    ans_qna = qna_chain.run(query)
    st.session_state.history.append((query, ans_proc, ans_qna))

# 대화 이력 렌더링
for q, a1, a2 in st.session_state.history:
    st.chat_message("user").write(q)
    st.chat_message("assistant").markdown(f"**[프로세스 문서 답변]**\n{a1}")
    st.chat_message("assistant").markdown(f"**[대표질문 문서 답변]**\n{a2}")

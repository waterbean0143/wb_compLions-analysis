import os
import difflib
import streamlit as st
import pandas as pd

from streamlit import error
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI



# --- 환경 및 경로 설정 ---
# OpenAI API key는 환경변수로 설정되어 있다고 가정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("⚠️ OPENAI_API_KEY가 설정되어 있지 않습니다.")
    st.stop()

# 스크립트 기준 디렉터리
BASE_DIR = os.path.dirname(__file__)

# --- 1. 프로세스 테이블 로드 함수 ---
@st.cache_data
def load_process_table(path: str) -> pd.DataFrame:
    base_dir = os.path.dirname(__file__)
    abs_path = os.path.join(base_dir, path)
    try:
        df = pd.read_csv(abs_path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(abs_path, dtype=str, encoding="cp949")
    df = df.rename(columns={
        '주요 단계': 'step_name',
        '주요 활동': 'major',
        '시기': 'timing',
        '책임자': 'owner',
        '실무자': 'worker',
        '협조 및 지원 부서': 'support',
        '적용 시스템': 'system'
    })
    return df

df = load_process_table("SI_FULL_PROCESS_HIERARCHY.csv")

# --- 2. PDF 다운로드 함수 (예시) ---
def download_pdfs(urls: list[str]) -> list[str]:
    # URL→파일 경로 리스트 반환 (미구현 부분)
    paths = [...]
    return paths

pdf_paths = download_pdfs(VDC_PROCESS_DOC_URLS)

# --- 3. Retriever 빌드 함수 수정 버전 ---
@st.cache_resource
def build_retriever(pdf_paths: list[str]):
    if not pdf_paths:
        st.error("❌ 프로세스 문서 로드에 실패했습니다. PDF 경로가 없습니다.")
        from langchain.schema import Document
        empty_doc = Document(page_content="문서가 없습니다.", metadata={})
        emb = OpenAIEmbeddings(openai_api_key=openai_api_key)
        return FAISS.from_documents([empty_doc], emb).as_retriever(search_kwargs={"k": 1})

    loader = PyMuPDFLoader(pdf_paths[0])
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    emb = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = FAISS.from_documents(chunks, emb)
    return vectordb.as_retriever(search_kwargs={"k": 4})

retriever = build_retriever(pdf_paths)

# --- 4. PromptTemplate 정의 ---
PROMPT = PromptTemplate(
    input_variables=["table","doc_context","question"],
    template="""
당신은 SI 프로세스 전문가입니다.

아래 표에는 ‘{question}’ 에 해당하는 단계의 
주요 활동/시기/책임자 정보가 정리되어 있습니다:

{table}

추가로, 관련 문서 내용은 다음과 같습니다:
{doc_context}

위 정보를 종합하여
💡 핵심 요약
📋 절차 또는 판단 주체
🔢 정확한 수치
형식으로 답변해주세요."""
)

# --- 5. RetrievalQA 생성 ---
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)

# --- 6. 사용자 입력 및 답변 표시 ---
question = st.text_input("VDC‑A 관련 질문을 입력하세요")
if question:
    # 단계명 매칭
    import difflib
    matches = difflib.get_close_matches(question, df['step_name'].unique(), n=1, cutoff=0.5)
    step = matches[0] if matches else None

    if step:
        sub = df[df['step_name']==step][['major','timing','owner','worker','support','system']]
        sub = sub.rename(columns={
            'major':'주요 활동','timing':'시기','owner':'책임자',
            'worker':'실무자','support':'협조 및 지원 부서','system':'적용 시스템'
        })
        table_html = sub.to_html(index=False)
    else:
        table_html = ""

    docs_for_q = retriever.get_relevant_documents(question)
    doc_context = "\n\n".join([d.page_content for d in docs_for_q])

    answer = qa.invoke({
        "table": table_html,
        "doc_context": doc_context,
        "question": question
    })["result"]

    # 출력
    lines = answer.splitlines()
    st.markdown(f"### 💡 핵심 요약\n{lines[0] if len(lines)>0 else ''}")
    st.markdown(f"### 📋 절차 또는 판단 주체\n{lines[1] if len(lines)>1 else ''}")
    st.markdown(f"### 🔢 관련 수치\n{lines[2] if len(lines)>2 else ''}")
    with st.expander("📎 문서 소스 보기"):
        for d in docs_for_q:
            st.write(f"- {d.metadata.get('source_name','unknown')}")

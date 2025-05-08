import os
import difflib
import streamlit as st
import pandas as pd

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
def load_process_table(rel_path: str) -> pd.DataFrame:
    abs_path = os.path.join(BASE_DIR, rel_path)
    df = pd.read_csv(abs_path, dtype=str, encoding="utf-8-sig")
    # 컬럼명 통일
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

# --- 2. PDF 다운로드 & Retriever 준비 함수 ---
# (여기에 download_pdfs 함수, VDC_PROCESS_DOC_URLS 정의가 있다고 가정)
VDC_PROCESS_DOC_URLS = [
    # 예: "https://drive.google.com/uc?export=download&id=..."
]

@st.cache_resource
def download_pdfs(urls):
    from pathlib import Path
    import requests
    paths = []
    for url in urls:
        fname = url.split("id=")[-1] + ".pdf"
        out_path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(out_path):
            resp = requests.get(url)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
        paths.append(out_path)
    return paths

@st.cache_resource
def build_retriever(pdf_paths):
    # (A) 로드 & 청크
    loader = PyMuPDFLoader(pdf_paths[0])
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    # (B) 임베딩 & FAISS
    emb = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = FAISS.from_documents(chunks, emb)
    return vectordb.as_retriever(search_kwargs={"k": 4})

pdf_paths = download_pdfs(VDC_PROCESS_DOC_URLS)
retriever = build_retriever(pdf_paths)

# --- 3. PromptTemplate 정의 (vdc‑a 스타일) ---
PROMPT = PromptTemplate(
    input_variables=["table", "doc_context", "question"],
    template="""
당신은 SI 프로세스 전문가입니다.

아래 표에는 ‘{question}’ 에 해당하는 단계의 
주요 활동/시기/책임자/실무자/협조 및 지원 부서/적용 시스템 정보가 정리되어 있습니다:

{table}

추가로, 관련 문서 내용은 다음과 같습니다:
{doc_context}

위 정보를 종합하여
💡 핵심 요약
📋 절차 또는 판단 주체
🔢 정확한 수치
형식으로 답변해주세요.
""".strip()
)

# --- 4. RetrievalQA 체인 생성 ---
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)

# --- 5. Streamlit UI ---
st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")
st.title("📋 SI 프로세스 기반 Q&A")

question = st.text_input("VDC‑A 관련 질문을 입력하세요:")
if question:
    # 5.1 질문에서 단계(step_name)와 매칭되는 행을 추출하여
    # 원하는 6개 컬럼만 선택하도록 수정
    if match:
        sub = df[df['step_name'].str.match(match, na=False)][
            ['major', 'timing', 'owner', 'worker', 'support', 'system']
        ]
        table_html = sub.to_html(index=False, escape=False)
    else:
        table_html = "해당 단계 정보를 찾을 수 없습니다."

    # 5.2 문서 컨텍스트 검색
    docs_for_q = retriever.get_relevant_documents(question)
    doc_context = "\n\n".join([d.page_content for d in docs_for_q])

    # 5.3 QA 실행
    answer = qa.invoke({
        "table": table_html,
        "doc_context": doc_context,
        "question": question
    })["result"]

    # 6. 결과 파싱 & 출력
    lines = answer.splitlines()
    summary = lines[0] if len(lines)>0 else ""
    procedure = lines[1] if len(lines)>1 else ""
    numbers = lines[2] if len(lines)>2 else ""

    st.markdown(f"### 💡 핵심 요약\n{summary}")
    st.markdown(f"### 📋 절차 또는 판단 주체\n{procedure}")
    if numbers:
        st.markdown(f"### 🔢 관련 수치\n{numbers}")

    with st.expander("📎 문서 소스 보기"):
        for d in docs_for_q:
            src = d.metadata.get("source_name", d.metadata.get("source", "unknown"))
            st.write(f"- `{src}`, …")

mport os
import streamlit as st
import pandas as pd
import difflib
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# --- 0. 페이지 설정: 반드시 Streamlit 호출 전에 ---
st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")

# --- 1. 환경 변수 및 API 키 체크 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

# --- 2. 절차 개요 테이블 로드 ---
@st.cache_data
def load_process_table():
    import os
    # CSV 파일 경로: 스크립트 기준 상대 경로
    base = os.path.dirname(__file__)
    path = os.path.join(base, "SI_FULL_PROCESS_HIERARCHY.csv")
    # pandas로 직접 읽기, euc-kr 혹은 latin-1 인코딩 시도
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    # 컬럼명 매핑
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

df = load_process_table()

# --- 3. 문서 retriever 구축 (demo.py 방식) ---
VDC_PROCESS_DOC_URLS = [
    "https://drive.google.com/uc?export=download&id=1cEFCFC7fp3JuDRgdPS3BdJuHPKhLF3yn"
]

@st.cache_resource
def build_retriever(urls):
    import requests, tempfile
    paths = []
    for url in urls:
        r = requests.get(url)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(r.content)
        tmp.close()
        paths.append(tmp.name)
    loader = PyMuPDFLoader(paths[0])
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(chunks, emb)
    return vectordb.as_retriever(search_kwargs={"k": 4})

retriever = build_retriever(VDC_PROCESS_DOC_URLS)

# --- 4. PromptTemplate 정의 (vdc-a 스타일) ---
PROMPT = PromptTemplate(
    input_variables=["table", "doc_context", "question"],
    template="""
당신은 SI 프로세스 전문가입니다.

아래 표에는 ‘{question}’ 에 해당하는 단계의 주요 활동/시기/책임자 정보가 정리되어 있습니다:

{table}

추가로, 관련 문서 내용은 다음과 같습니다:
{doc_context}

위 정보를 종합하여  
💡 핵심 요약  
📋 절차 또는 판단 주체  
🔢 정확한 수치  
형식으로 답변해주세요.
"""
)

# --- 5. RetrievalQA 체인 생성 ---
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY),
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT}
)

# --- 6. Streamlit UI ---
st.title("📝 SI 프로세스 챗봇")
question = st.text_input("VDC‑A 관련 질문을 입력하세요")

if question:
    # 6.1 단계 추출
    matches = difflib.get_close_matches(question, df['step_name'].dropna().unique(), n=1, cutoff=0.5)
    if matches:
        step = matches[0]
        sub = df[df['step_name'] == step][['major','timing','owner','worker','support','system']]
        sub = sub.rename(columns={
            'major':'주요 활동','timing':'시기','owner':'책임자',
            'worker':'실무자','support':'협조 및 지원 부서','system':'적용 시스템'
        })
        # HTML 테이블로 변환, escape=False로 HTML 허용
        table_html = sub.to_html(index=False, escape=False)
    else:
        table_html = "해당 단계 정보를 찾을 수 없습니다."

    # 6.2 문서 컨텍스트
    docs_for_q = retriever.get_relevant_documents(question)
    doc_context = "\n\n".join([d.page_content for d in docs_for_q])

    # 6.3 QA 실행
    result = qa.invoke({"table": table_html, "doc_context": doc_context, "question": question})
    answer = result.get('result', '')

    # 6.4 응답 파싱 및 출력
    parts = [line for line in answer.splitlines() if line.strip()]
    if parts:
        st.markdown(f"### 💡 핵심 요약\n{parts[0]}")
    if len(parts) > 1:
        st.markdown(f"### 📋 절차 또는 판단 주체\n{parts[1]}")
    if len(parts) > 2:
        st.markdown(f"### 🔢 정확한 수치\n{parts[2]}")

    # 6.5 소스 문서 표시
    with st.expander("📎 문서 소스 보기"):
        for d in docs_for_q:
            src = d.metadata.get("source_name", "unknown")
            st.write(f"- `{src}`")

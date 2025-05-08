# ── 0. 반드시 최상단에 페이지 설정 ───────────────────────────────────
import streamlit as st
st.set_page_config(page_title="SI 방법론 챗봇", layout="wide")

# ── 1. 필요한 패키지 임포트 ────────────────────────────────────────
import os
import tempfile
import pandas as pd
import difflib
import gdown

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# ── 2. 구글 드라이브 PDF 다운로드 헬퍼 ───────────────────────────────
PROCESS_DOC_IDS = ["1TNOhmUds7hMpwz3NO4QD-mO-J1sUJoEa"]
QNA_DOC_IDS     = ["17M1mnMZVl29EahbSVqzcyZEX8LYsx5ER"]

@st.cache_resource(ttl=3600*24)
def download_pdfs(ids: list[str]) -> list[str]:
    os.makedirs("pdfs", exist_ok=True)
    paths = []
    for fid in ids:
        out = f"pdfs/{fid}.pdf"
        if not os.path.exists(out):
            gdown.download(
                f"https://drive.google.com/uc?export=download&id={fid}",
                out, quiet=True
            )
        paths.append(out)
    return paths

# ── 3. CSV 로드 및 칼럼명 통일 ────────────────────────────────────
@st.cache_data
def load_process_table(path: str) -> pd.DataFrame:
    base = os.path.dirname(__file__)
    abs_path = os.path.join(base, path)
    try:
        # sep, engine 명시
        df = pd.read_csv(
            abs_path,
            sep=",",
            engine="C",
            dtype=str,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_NONE,      # ← 이 줄을 추가
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            abs_path,
            sep=",",
            engine="C",
            dtype=str,
            encoding="cp949",
            quoting=csv.QUOTE_NONE,      # ← 이 줄도 추가
        )

    # 한글→영문 컬럼명 통일 (rename만 사용)
    df = df.rename(columns={
        '주요 단계':'step_name',
        '주요 활동':'major',
        '시기':'timing',
        '책임자':'owner',
        '실무자':'worker',
        '협조 및 지원 부서':'support',
        '적용 시스템':'system'
    })

    return df

# 호출부
df = load_process_table("SI_FULL_PROCESS_HIERARCHY.csv")

# 확인용 (디버그)
st.write("▶︎ CSV 컬럼명:", df.columns.tolist())

# ── 4. FAISS 리트리버 빌더 ────────────────────────────────────────
@st.cache_resource
def build_faiss_retriever(pdf_paths: list[str], k: int=4):
    if not pdf_paths:
        st.error("문서 로드 실패: PDF 경로가 없습니다.")
        from langchain.schema import Document
        empty = Document(page_content="문서가 없습니다.", metadata={})
        emb_empty = OpenAIEmbeddings()
        return FAISS.from_documents([empty], emb_empty).as_retriever(search_kwargs={"k":1})

    loader = PyMuPDFLoader(pdf_paths[0])
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    emb = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = FAISS.from_documents(chunks, emb)
    return vectordb.as_retriever(search_kwargs={"k":k})

proc_paths = download_pdfs(PROCESS_DOC_IDS)
qna_paths  = download_pdfs(QNA_DOC_IDS)

proc_retriever = build_faiss_retriever(proc_paths, k=4)
qna_retriever  = build_faiss_retriever(qna_paths, k=4)

# ── 5. Streamlit UI ──────────────────────────────────────────────
st.title("📝 SI 방법론 Q&A")
question = st.text_input("SI 방법론 관련 질문을 입력하세요")

if question:
    # 5‑1) 단계 이름 매칭
    match = difflib.get_close_matches(question, df['step_name'].unique(), n=1, cutoff=0.5)
    if match:
        step = match[0]
        sub = df[df['step_name']==step][['major','timing','owner','worker','support','system']]
        sub = sub.rename(columns={
            'major':'주요 활동','timing':'시기','owner':'책임자',
            'worker':'실무자','support':'협조 및 지원 부서','system':'적용 시스템'
        })
        table_html = sub.to_html(index=False, escape=False)
    else:
        table_html = "해당 단계 정보가 없습니다."

    # 5‑2) 문서 컨텍스트 취합
    docs_proc = proc_retriever.get_relevant_documents(question)
    docs_qna  = qna_retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs_proc+docs_qna])

    # 5‑3) LLM 프롬프트
    PROMPT = PromptTemplate(
        input_variables=["table","doc_context","question"],
        template="""
당신은 SI 방법론 전문가입니다.

아래 표는 질문 “{question}” 과 매칭된 단계의 개요입니다:
{table}

추가로, 관련 문서 내용:
{doc_context}

위 정보를 종합하여 아래 형식으로 답변하세요.

💡 핵심 요약  
📋 절차 또는 판단 주체  
🔢 관련 수치  
"""
    )

    # 5‑4) RetrievalQA 체인
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
        retriever=proc_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT}
    )

    # 5‑5) 실행
    out = qa.invoke({
        "table": table_html,
        "doc_context": context,
        "question": question
    })["result"]

    # 5‑6) 결과 렌더링
    lines = out.splitlines()
    st.markdown(f"### 💡 핵심 요약\n{lines[0] if len(lines)>0 else ''}")
    st.markdown(f"### 📋 절차 또는 판단 주체\n{lines[1] if len(lines)>1 else ''}")
    st.markdown(f"### 🔢 관련 수치\n{lines[2] if len(lines)>2 else ''}")

    with st.expander("📎 문서 소스 보기"):
        for d in docs_proc+docs_qna:
            name = d.metadata.get("source_name", os.path.basename(d.metadata.get("source","")))
            st.write(f"- {name}")

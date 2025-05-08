import streamlit as st
import pandas as pd
import os
import requests
import tempfile
import re
import json
import difflib
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_upstage import UpstageGroundednessCheck

# 환경변수 로드 (OpenAI API Key 등)
load_dotenv()

@st.cache_data
def load_steps_config(path: str) -> list[dict]:
    """
    CSV 포맷으로 변환된 SI 프로세스 인덱스를 읽어옵니다.
    """
    df = pd.read_csv(path, dtype=str, encoding='cp949')
    return df.to_dict(orient='records')

@st.cache_data
def load_hierarchy(path: str) -> pd.DataFrame:
    """
    CSV로 변환된 SI 프로세스 단계별 상세 정보를 읽어옵니다.
    """
    df = pd.read_csv(path, dtype=str, encoding='cp949')
    df = df.rename(columns={
        '주요 단계': 'step_name',
        '주요 활동': 'major',
        '시기': 'timing',
        '책임자': 'owner',
        '실무자': 'worker',
        '협조 및 지원 부서': 'support',
        '적용 시스템': 'system'
    })
    return df.fillna('')

@st.cache_data
def load_rep_qna(path: str) -> list[dict]:
    """
    JSON 포맷의 대표 질문 리스트를 로드합니다.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())

def find_matching_majors(query: str, majors: list[str]) -> list[str]:
    ni = normalize(query)
    matches = []
    for m in majors:
        nm = normalize(m)
        if nm.startswith(ni) or ni in nm:
            matches.append(m)
    return matches

@st.cache_resource
def download_to_temp(url: str) -> str:
    r = requests.get(url)
    r.raise_for_status()
    suffix = '.pdf' if url.lower().endswith('pdf') else '.docx'
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'wb') as f:
        f.write(r.content)
    return tmp_path

# 주요 로직

def main():
    st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")
    # CSV 인덱스 불러오기
    configs = load_steps_config("4. full process index.CSV")
    # 단계별 상세정보 CSV 불러오기 (hierarchy.csv)
    hier = load_hierarchy("SI_FULL_PROCESS_HIERARCHY.CSV")
    # 대표 Q&A JSON
    rep_qna = load_rep_qna("vdc_a_대표질문.json")

    st.sidebar.title("SI 챗봇")
    step = st.sidebar.selectbox("프로세스 단계 선택", [c['step_name'] for c in configs])
    cfg = next(c for c in configs if c['step_name'] == step)

    intro_tab, overview_tab, qa_tab = st.tabs(["소개", "절차 개요", "Q&A"])

    with intro_tab:
        st.markdown("## SI 전체 프로세스 챗봇에 오신 것을 환영합니다.")
        st.write("좌측에서 단계 선택 후 개요·Q&A 이용")

    with overview_tab:
        st.header(f"{step} 단계 상세")
        df_step = hier[hier['step_name'] == step]
        # 각 major별 상세 내용 표시
        for major_label in df_step['major'].dropna().unique():
            with st.expander(major_label):
                display_major_detail(df_step, major_label)

    with qa_tab:
        st.header("Q&A")
        query = st.text_input("질문 입력")
        if query:
            # 1) 대표 Q&A 매칭
            questions = [e['question'] for e in rep_qna]
            match = difflib.get_close_matches(query, questions, n=1, cutoff=0.6)
            if match:
                entry = next(e for e in rep_qna if e['question'] == match[0])
                st.subheader("🔎 대표 질문 매칭 답변")
                st.write(entry['answer'])
                st.caption(f"출처: {entry.get('출처','unknown')}")
            else:
                # 2) 문서 기반 Q&A
                df_step = hier[hier['step_name'] == step]
                majors = df_step['major'].dropna().unique().tolist()
                matched = find_matching_majors(query, majors)
                sel_major = matched[0] if matched else majors[0]

                loader = PyMuPDFLoader(download_to_temp(cfg['document_url']))
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                embeddings = OpenAIEmbeddings()
                vectorstore = FAISS.from_documents(chunks, embeddings)

                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
                    당신은 SI 프로세스 문서 기반 질문에 답하는 어시스턴트입니다.

                    [문서]
                    {context}

                    [질문]
                    {question}

                    [답변 형식]
                    💡 핵심 요약
                    📋 절차 또는 판단 주체
                    """
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(temperature=0.0),
                    retriever=vectorstore.as_retriever(),
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": prompt}
                )
                upstage_checker = UpstageGroundednessCheck()
                with st.spinner("답변 생성 중..."):
                    ans = qa_chain.run(query)
                    relevance = upstage_checker.invoke(ans)
                if not relevance.get("grounded", False):
                    st.warning("📌 경고: 문서 근거가 부족한 응답일 수 있습니다.")
                st.markdown(f"**답변:**\n> {ans}")

if __name__ == "__main__":
    main()

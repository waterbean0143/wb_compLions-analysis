import streamlit as st
import pandas as pd
import os
import requests
import tempfile
import re
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
    df = pd.read_csv(path, dtype=str)
    return df.to_dict(orient='records')

@st.cache_data
def load_hierarchy(path: str, sheet_index: int = 3):
    df = pd.read_excel(path, sheet_name=sheet_index, dtype=str)
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

# 이하 주요 함수 정의들...


def main():
    st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")
    # CSV로 변환된 인덱스 파일 경로로 변경
    configs = load_steps_config("4. full process index.CSV")
    # 계층 정보는 기존 Excel 파일 사용
    hier = load_hierarchy("SI_FULL_PROCESS_EXCEL.xlsx")

    st.sidebar.title("SI 챗봇")
    step = st.sidebar.selectbox("프로세스 단계 선택", [c['step_name'] for c in configs])
    cfg = next(c for c in configs if c['step_name'] == step)

    intro_tab, overview_tab, qa_tab = st.tabs(["소개", "절차 개요", "Q&A"])

    with intro_tab:
        st.markdown("## SI 전체 프로세스 챗봇에 오신 것을 환영합니다.")
        st.write("좌측에서 단계 선택 후 개요·Q&A 이용")

    with overview_tab:
        st.header(f"{step} 단계 상세")
        # 개요 표시 로직
        # display_major_detail 등 기존 함수 사용
        df_step = hier[hier['step_name'] == step]
        for major_label in df_step['major'].dropna().unique():
            with st.expander(major_label):
                display_major_detail(df_step, major_label)

    with qa_tab:
        st.header("Q&A")
        query = st.text_input("질문 입력")
        if query:
            df_step = hier[hier['step_name'] == step]
            majors = df_step['major'].dropna().unique().tolist()
            matched = find_matching_majors(query, majors)
            if not matched:
                sel_major = majors[0]
            elif len(matched) == 1:
                sel_major = matched[0]
            else:
                sel_major = st.selectbox("여러 단계가 있습니다. 어느 단계를 원하시나요?", matched)

            # Q&A 체인 생성 및 실행
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
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
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

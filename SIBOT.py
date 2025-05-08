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
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_upstage import UpstageGroundednessCheck

# 환경변수 로드 (OpenAI API Key 등)
load_dotenv()

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """
    UTF-8-sig 인코딩된 CSV를 읽어옵니다.
    """
    return pd.read_csv(path, dtype=str, encoding='utf-8-sig')

@st.cache_resource
def download_to_temp(url: str) -> str:
    r = requests.get(url)
    r.raise_for_status()
    suffix = '.pdf' if url.lower().endswith('pdf') else '.docx'
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, 'wb') as f:
        f.write(r.content)
    return tmp_path

# 텍스트 정규화 및 유사도 매칭

def normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())

def find_matching_majors(query: str, majors: list[str]) -> list[str]:
    ni = normalize(query)
    return [m for m in majors if normalize(m).startswith(ni) or ni in normalize(m)]

# 메인 로직

def main():
    st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")

    # 단계별 상세정보 CSV 로드
    df = load_csv("SI_FULL_PROCESS_HIERARCHY.CSV")

    # configs: 고유 단계명과 문서 URL 추출
    configs = []
    for step in df['step_name'].dropna().unique():
        urls = df.loc[df['step_name'] == step, 'document_url'].dropna().unique()
        doc_url = urls[0] if len(urls) > 0 else ''
        configs.append({'step_name': step, 'document_url': doc_url})

    # 대표 Q&A JSON 로드
    with open("vdc_a_대표질문.json", encoding='utf-8') as f:
        rep_qna = json.load(f)

    # 사이드바: 단계 선택
    st.sidebar.title("SI 챗봇")
    step_names = [c['step_name'] for c in configs]
    step = st.sidebar.selectbox("프로세스 단계 선택", step_names)
    cfg = next(c for c in configs if c['step_name'] == step)

    intro_tab, overview_tab, qa_tab = st.tabs(["소개", "절차 개요", "Q&A"])

    with intro_tab:
        st.markdown("## SI 전체 프로세스 챗봇에 오신 것을 환영합니다.")
        st.write("좌측에서 단계 선택 후 개요·Q&A 이용")

    with overview_tab:
        st.header(f"{step} 단계 상세")
        df_step = df[df['step_name'] == step]
        for major_label in df_step['major'].dropna().unique():
            with st.expander(major_label):
                display_major_detail(df_step, major_label)

    with qa_tab:
        st.header("Q&A")
        query = st.text_input("질문 입력")
        if query:
            # 1) 대표 Q&A 유사도 매칭
            questions = [e['question'] for e in rep_qna]
            close = difflib.get_close_matches(query, questions, n=1, cutoff=0.6)
            if close:
                entry = next(e for e in rep_qna if e['question'] == close[0])
                st.subheader("🔎 대표 질문 매칭 답변")
                st.write(entry['answer'])
                st.caption(f"출처: {entry.get('출처','없음')}")
            else:
                # 2) 문서 기반 Q&A
                df_step = df[df['step_name'] == step]
                majors = df_step['major'].dropna().unique().tolist()
                sel_major = find_matching_majors(query, majors)[0] if find_matching_majors(query, majors) else majors[0]

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
                checker = UpstageGroundednessCheck()
                with st.spinner("답변 생성 중..."):
                    ans = qa_chain.run(query)
                    relevance = checker.invoke(ans)
                if not relevance.get("grounded", False):
                    st.warning("📌 경고: 문서 근거가 부족할 수 있습니다.")
                st.markdown(f"**답변:**\n> {ans}")

if __name__ == "__main__":
    main()

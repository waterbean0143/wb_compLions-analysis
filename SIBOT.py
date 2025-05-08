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

# 환경변수 로드
load_dotenv()

# 사용자 인증 정보
users = {
    "admin": {"password": "admin", "name": "관리자"},
    "test": {"password": "test", "name": "테스트 사용자"},
}

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """UTF-8-sig 인코딩된 CSV를 읽어옵니다."""
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


def normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())


def display_table(df: pd.DataFrame):
    """
    주요 활동 테이블을 보여줍니다.
    """
    cols = ['major','timing','owner','worker','support','system']
    table_df = df[cols].rename(columns={
        'major':'주요 활동',
        'timing':'시기',
        'owner':'책임자',
        'worker':'실무자',
        'support':'협조 및 지원 부서',
        'system':'적용 시스템'
    }).reset_index(drop=True)
    st.table(table_df)


def main():
    st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")

    # 로그인
    if not st.session_state.get('logged_in', False):
        st.header("📋 SI 프로세스 챗봇 로그인")
        username = st.text_input("아이디", key="login_user")
        password = st.text_input("비밀번호", type="password", key="login_pw")
        if st.button("로그인", key="login_button"):
            if username in users and password == users[username]["password"]:
                st.session_state['logged_in'] = True
                st.session_state['logged_in_user'] = username
                st.experimental_rerun()
            else:
                st.error("로그인 정보가 올바르지 않습니다.")
        st.stop()

    display_name = users[st.session_state['logged_in_user']]['name']

    # CSV 로드 및 변환
    df = load_csv("SI_FULL_PROCESS_HIERARCHY.csv")
    rename_map = {
        '주요 단계':'step_name', '주요 활동':'major', '시기':'timing',
        '책임자':'owner', '실무자':'worker', '협조 및 지원 부서':'support',
        '적용 시스템':'system', '문서 URL':'document_url'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    if 'document_url' not in df.columns: df['document_url'] = ''

    # 단계 목록 및 URL 매핑
    step_names = df['step_name'].dropna().unique().tolist()
    steps = [{'step_name': s, 'document_url': df.loc[df['step_name']==s, 'document_url'].dropna().unique()[0] if s in df['step_name'].values else ''} for s in step_names]

    # sidebar
    st.sidebar.header(f"[접속자] {display_name}님, 환영합니다!")
    if st.sidebar.button("새로운 대화 주제", key="clear_button"):
        st.session_state.clear(); st.experimental_rerun()
    st.sidebar.selectbox("언어 모델 선택", ["o4-mini"], key="model_select",
                        help="현재 사용할 LLM 모델을 선택합니다. o4-mini만 지원됩니다.")
    st.sidebar.selectbox("답변 모드 선택", ["빠른 답변","정확한 답변"], index=0, key="answer_mode_select",
                        help="• 빠른 답변: 빠르게 핵심만 요약\n• 정확한 답변: 보다 상세히 답변합니다.")
    st.sidebar.selectbox("추론 수준 선택", ["low","medium","high"], index=1, key="reasoning_effort_select",
                        help="추론 깊이를 설정합니다: low/medium/high")
    st.sidebar.header("피드백")
    st.sidebar.text_area("전반적인 사용후기를 입력해주세요:", key="feedback_text")
    col1,col2=st.sidebar.columns([2,1])
    with col1:
        if st.sidebar.button("피드백 제출", key="feedback_submit"): st.sidebar.success("피드백이 제출되었습니다.")
    with col2:
        if st.sidebar.button("로그아웃", key="logout_button"): st.session_state.clear(); st.experimental_rerun()
    st.sidebar.write("🐧 저작자: @AI이행봇")

    # tabs
    intro_tab, overview_tab, qa_tab = st.tabs(["소개","절차 개요","Q&A"])

    with intro_tab:
        st.markdown("## SI 전체 프로세스 챗봇")
        st.write("사이드바에서 설정 후 아래 탭을 사용하세요.")

    # 라이브러리 대신 하드코딩으로 Roman numerals
    roman_numerals = ["I","II","III","IV","V","VI","VII","VIII","IX","X"]

    with overview_tab:
        st.header("절차 개요")
        for idx, step in enumerate(steps):
            num = roman_numerals[idx] if idx < len(roman_numerals) else str(idx+1)
            st.subheader(f"{num}. {step['step_name']}")
            display_table(df[df['step_name']==step['step_name']])

    with qa_tab:
        st.header("Q&A")
        scope_options = ["전체"] + step_names
        scope = st.selectbox("질문 범위 선택 (전체 또는 단계)", scope_options, key="qa_scope")
        query = st.text_input("질문 입력")
        if query:
            if scope != "전체" and scope not in step_names:
                st.error("잘못된 범위 선택입니다. 범위를 다시 선택해주세요.")
            else:
                # 문서 기반 Q&A
                docs = []
                if scope == "전체":
                    for s in steps:
                        if s['document_url']:
                            docs += PyMuPDFLoader(download_to_temp(s['document_url'])).load()
                else:
                    url = next((s['document_url'] for s in steps if s['step_name']==scope), '')
                    if url:
                        docs = PyMuPDFLoader(download_to_temp(url)).load()
                if not docs:
                    st.error("선택된 범위에 문서가 없습니다. 다른 범위를 선택해주세요.")
                else:
                    chunks = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100).split_documents(docs)
                    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(temperature=0.0),
                        retriever=vectorstore.as_retriever(),
                        chain_type="stuff",
                        chain_type_kwargs={"prompt": PromptTemplate(
                            input_variables=["context","question"],
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
                        )}
                    )
                    with st.spinner("답변 생성 중..."):
                        ans = qa_chain.run(query)
                        rel = UpstageGroundednessCheck().invoke(ans)
                    if not rel.get("grounded",False):
                        st.warning("문서 근거가 부족할 수 있습니다.")
                    st.markdown(f"**답변:**\n> {ans}")

if __name__ == "__main__":
    main()

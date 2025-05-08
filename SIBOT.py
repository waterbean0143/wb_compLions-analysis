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
from rank_bm25 import BM25Okapi

# 환경변수 로드
load_dotenv()

# 사용자 인증 정보
users = {
    "admin": {"password": "admin", "name": "관리자"},
    "test": {"password": "test", "name": "테스트 사용자"},
}

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
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

# 주요 활동 테이블 표시

def display_table(df: pd.DataFrame):
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

# 텍스트 정규화

def normalize(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())

# 메인 함수

def main():
    st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")

    # 로그인 처리
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

    # CSV 로드
    BASE_DIR = os.path.dirname(__file__)
    csv_path = os.path.join(BASE_DIR, "SI_FULL_PROCESS_HIERARCHY.csv")
    if not os.path.isfile(csv_path):
        st.error(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        return
    df = load_csv(csv_path)
    rename_map = {
        '주요 단계':'step_name','주요 활동':'major','시기':'timing',
        '책임자':'owner','실무자':'worker','협조 및 지원 부서':'support',
        '적용 시스템':'system','문서 URL':'document_url'
    }
    df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})
    if 'document_url' not in df.columns:
        df['document_url'] = ''

    # 대표 Q&A JSON 로드
    try:
        with open(os.path.join(BASE_DIR, "vdc_a_대표질문.json"), encoding='utf-8') as f:
            rep_qna = json.load(f)
    except FileNotFoundError:
        rep_qna = []

    # 단계 목록 및 URL 매핑
    topics = df['major'].dropna().unique().tolist()
    norm_topics = {normalize(t): t for t in topics}

    # 사이드바
    st.sidebar.header(f"[접속자] {display_name}님, 환영합니다!")
    if st.sidebar.button("새로운 대화 주제", key="clear_button"):
        st.session_state.clear(); st.experimental_rerun()
    st.sidebar.selectbox("언어 모델 선택", ["gpt-4o-mini"], key="model_select")
    st.sidebar.selectbox("답변 모드 선택", ["빠른 답변","정확한 답변"], index=0, key="answer_mode_select")
    st.sidebar.selectbox("추론 수준 선택", ["low","medium","high"], index=1, key="reasoning_effort_select")
    st.sidebar.header("피드백")
    st.sidebar.text_area("사용 후기:", key="feedback_text")
    c1, c2 = st.sidebar.columns([2,1])
    with c1:
        if st.sidebar.button("제출", key="feedback_submit"): st.sidebar.success("감사합니다!")
    with c2:
        if st.sidebar.button("로그아웃", key="logout_button"): st.session_state.clear(); st.experimental_rerun()
    st.sidebar.write("🐧 저작자: @AI이행봇")

    # 탭
    intro_tab, overview_tab, qa_tab = st.tabs(["소개","절차 개요","Q&A"])

    with intro_tab:
        st.markdown("## SI 전체 프로세스 챗봇")
        st.write("사이드바에서 설정 후 탭을 사용하세요.")

    with overview_tab:
        st.header("절차 개요")
        for idx, major in enumerate(topics, start=1):
            roman = ["I","II","III","IV","V","VI","VII","VIII","IX","X"][idx-1]
            st.subheader(f"{roman}. {major}")
            display_table(df[df['major'] == major])

    with qa_tab:
        st.header("Q&A")
        query = st.text_input("질문 입력", key="qna_input")
        if st.button("질문하기", key="qna_button") and query:
            # 1) 대표 Q&A 우선 매칭
            if rep_qna:
                rep_norms = {normalize(e.get('question','')): e for e in rep_qna}
                norm_q = normalize(query)
                m = difflib.get_close_matches(norm_q, rep_norms.keys(), n=1, cutoff=0.7)
                if m:
                    entry = rep_norms[m[0]]
                    st.subheader("🔎 대표 질문 매칭 답변")
                    st.write(entry.get('answer',''))
                    if entry.get('출처'):
                        st.caption(f"출처: {entry.get('출처')}")
                    return
            # 2) topic 매칭
            norm_q = normalize(query)
            key = None
            m2 = difflib.get_close_matches(norm_q, norm_topics.keys(), n=1, cutoff=0.3)
            if m2:
                key = m2[0]
                topic = norm_topics[key]
                st.subheader(f"🔎 주제: {topic}")
            else:
                st.error("죄송합니다. 질문에 맞는 주제를 찾지 못했습니다. 다시 시도해 주세요.")
                return
            # 3) 문서 로드 및 BM25
            row = df[df['major'] == topic]
            url = row['document_url'].iloc[0] if not row.empty else ''
            if not url:
                st.warning("문서 URL이 없습니다.")
                return
            st.markdown(f"**원본 문서 URL:** {url}")
            docs = PyMuPDFLoader(download_to_temp(url)).load()
            texts = [d.page_content for d in docs]
            bm25 = BM25Okapi([t.split() for t in texts])
            top5 = bm25.get_top_n(query.split(), texts, n=5)
            context = "\n\n".join(top5)
            # 4) LLM 답변
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
            prompt = PromptTemplate(
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
            )
            answer = llm.predict(prompt.format(context=context, question=query))
            rel = UpstageGroundednessCheck().invoke(answer)
            if not rel.get('grounded', False):
                st.warning("📌 문서 근거가 부족할 수 있습니다.")
            st.markdown(f"**답변:**\n> {answer}")

if __name__ == "__main__":
    main()

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

# 메인

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

    # 단계 목록
    step_names = df['step_name'].dropna().unique().tolist()
    steps = [
        { 'step_name': s,
          'document_url': df.loc[df['step_name']==s, 'document_url'].dropna().unique()[0] if s in df['step_name'].values else '' }
        for s in step_names
    ]

    # 사이드바
    st.sidebar.header(f"[접속자] {display_name}님, 환영합니다!")
    if st.sidebar.button("새로운 대화 주제", key="clear_button"):
        st.session_state.clear()
        st.experimental_rerun()
    st.sidebar.selectbox("언어 모델 선택", ["o4-mini"], key="model_select",
                        help="현재 사용할 LLM 모델을 선택합니다. o4-mini만 지원됩니다.")
    st.sidebar.selectbox("답변 모드 선택", ["빠른 답변","정확한 답변"], index=0, key="answer_mode_select",
                        help="• 빠른: 핵심 요약 • 정확: 상세 설명")
    st.sidebar.selectbox("추론 수준 선택", ["low","medium","high"], index=1, key="reasoning_effort_select",
                        help="추론 깊이를 설정합니다.")
    st.sidebar.header("피드백")
    st.sidebar.text_area("사용 후기:", key="feedback_text")
    c1,c2 = st.sidebar.columns([2,1])
    with c1:
        if st.sidebar.button("제출", key="feedback_submit"): st.sidebar.success("감사합니다!")
    with c2:
        if st.sidebar.button("로그아웃", key="logout_button"):
            st.session_state.clear(); st.experimental_rerun()
    st.sidebar.write("🐧 저작자: @AI이행봇")

    # 탭
    intro_tab, overview_tab, qa_tab = st.tabs(["소개","절차 개요","Q&A"])

    with intro_tab:
        st.markdown("## SI 전체 프로세스 챗봇")
        st.write("사이드바에서 설정 후 탭을 사용하세요.")

    # overview (unchanged)
    with overview_tab:
        st.header("절차 개요")
        for idx, step in enumerate(steps, start=1):
            num = ["I","II","III","IV","V","VI","VII","VIII","IX","X"][idx-1]
            st.subheader(f"{num}. {step['step_name']}")
            display_table(df[df['step_name']==step['step_name']])

    # Q&A: 범위 필터 + RetrievalQA + Groundedness
            with qa_tab:
        st.header("Q&A")
        query = st.text_input("질문 입력", key="qna_input")
        if st.button("질문하기", key="qna_button") and query:
            # 분류된 단계 찾기
            step_list = [
                "영업기회 등록","VDC-A 심의 준비","VDC-A 심의 실시 및 공조 요청",
                "제안환경 구성","사전공고 분석 및 Risk 검토","RFP 분석","제안서 작성",
                "하도급사 검증","견적 확보 및 원가 산정","Risk 검토 및 대응방안 수립",
                "VDC-B 실시","입찰 및 제안서 제출","제안 발표",
                "기술협상 준비 및 실시","이행원가 재산정 및 계약서 검토",
                "VDC-C 실시","계약 체결","프로젝트 발주 처리"
            ]
            norm = lambda t: re.sub(r'[^a-z0-9]', '', t.lower())
            match = difflib.get_close_matches(query, step_list, n=1, cutoff=0.4)
            if not match:
                st.error("해당 질문에 맞는 주제를 찾지 못했습니다. 다른 표현으로 다시 시도해주세요.")
            else:
                topic = match[0]
                st.subheader(f"🔎 주제: {topic}")
                # 원본 문서 URL 가져오기
                df_row = df[df['major'].str.contains(re.escape(topic), na=False)]
                if df_row.empty:
                    st.warning("주제에 해당하는 문서 URL이 없습니다.")
                    return
                doc_url = df_row['document_url'].iloc[0]
                st.markdown(f"**원본 문서 URL:** {doc_url}")
                # 문서 로드
                temp_path = download_to_temp(doc_url)
                loader = PyMuPDFLoader(temp_path)
                docs = loader.load()
                                # BM25 인덱싱
                texts = [d.page_content for d in docs]
                tokenized = [t.split() for t in texts]
                bm25 = BM25Okapi(tokenized)
                q_tokens = query.split()
                top_n = bm25.get_top_n(q_tokens, texts, n=5)
                # 상위 5개 결과를 두 줄 개행으로 연결
                context = "

".join(top_n)

                # LLM에 질의
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
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=None,
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": prompt}
                )
                # chain.llm.predict가 아닌 chain.run을 사용해 context를 인라인 전달
                answer = chain.run({"context": context, "question": query})

                # 근거 검사
                relevance = UpstageGroundednessCheck().invoke(answer)
                if not relevance.get("grounded", False):
                    st.warning("📌 문서 근거가 부족할 수 있습니다.")
                st.markdown(f"**답변:**
> {answer}")

            # Q&A 블록 끝

if __name__ == "__main__":
    main()
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
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=None,
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": prompt}
                )
                # chain.run won't use retriever; pass context manually
                answer = chain.llm.predict(prompt.format(context=context, question=query))
                # 근거 검사
                relevance = UpstageGroundednessCheck().invoke(answer)
                if not relevance.get("grounded", False):
                    st.warning("📌 문서 근거가 부족할 수 있습니다.")
                st.markdown(f"**답변:**
> {answer}")

if __name__ == "__main__":
    main()
    main()
    main()

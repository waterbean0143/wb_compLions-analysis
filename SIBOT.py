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
def load_steps_config(path: str):
    df = pd.read_excel(path, sheet_name=0, dtype=str)
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


def find_matching_majors(user_input: str, majors: list[str]) -> list[str]:
    ni = normalize(user_input)
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

@st.cache_resource
def get_vector_store(cfg: dict):
    vs_path = f"vs_{cfg['step_id']}.faiss"
    if os.path.exists(vs_path):
        return FAISS.load_local(vs_path, OpenAIEmbeddings())
    local_file = download_to_temp(cfg['doc_url'])
    if cfg['doc_type'].lower() == 'pdf':
        docs = PyMuPDFLoader(local_file).load()
    else:
        docs = UnstructuredWordDocumentLoader(local_file).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50).split_documents(docs)
    vs = FAISS.from_documents(chunks, OpenAIEmbeddings())
    vs.save_local(vs_path)
    return vs


def create_major_tabs(df_step: pd.DataFrame) -> dict[str, st.delta_generator.DeltaGenerator]:
    parsed = []
    for m in df_step['major'].dropna().unique():
        mo = re.match(r'^\s*(\d+(?:\.\d+)?)', m)
        num = float(mo.group(1)) if mo else float('inf')
        parsed.append((num, m))
    parsed.sort(key=lambda x: x[0])
    labels = [label for _, label in parsed]
    tabs = st.tabs(labels)
    return {labels[i]: tabs[i] for i in range(len(labels))}


def display_major_detail(df_step: pd.DataFrame, major_label: str):
    st.subheader(f"▶ 주요 활동: {major_label}")
    subset = df_step[df_step['major'] == major_label]
    for _, row in subset.iterrows():
        st.markdown(f"- **시기**: {row['timing']}")
        st.markdown(f"- **책임자**: {row['owner']}")
        st.markdown(f"- **실무자**: {row['worker']}")
        st.markdown(f"- **협조/지원 부서**: {row['support']}")
        st.markdown(f"- **적용 시스템**: {row['system']}")


def main():
    st.set_page_config(page_title="SI 프로세스 챗봇", layout="wide")
    configs = load_steps_config("SI_FULL_PROCESS_EXCEL.xlsx")
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
        df_step = hier[hier['step_name'] == step]
        major_tabs = create_major_tabs(df_step)
        for major_label, tab in major_tabs.items():
            with tab:
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

            st.subheader(f"🔎 {step} – {sel_major} 단계에 답변합니다")
            vs = get_vector_store(cfg)
            retriever = vs.as_retriever(search_kwargs={"k":4})
            prompt = PromptTemplate(
                input_variables=["context","question"],
                template=f"""
당신은 SI 전체 프로세스 중 **{sel_major}** 단계에 특화된 AI 어시스턴트입니다.

질문: {{question}}

문서 내용:
{{context}}
"""
            )
                        # Upstage Groundedness Check 초기화
            upstage_checker = UpstageGroundednessCheck()
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
                retriever=retriever,
                chain_type="stuff",
                chain_type_kwargs={"prompt": prompt}
            )
            with st.spinner("답변 생성 중..."):
                ans = qa_chain.run(query)
                # Groundedness 판별
                relevance = upstage_checker.invoke(ans)

            if not relevance.get("grounded", False):
                st.warning("📌 경고: 문서 근거가 부족한 응답일 수 있습니다.")
            st.markdown(f"**답변:**\n> {ans}")

if __name__ == "__main__":
    main()

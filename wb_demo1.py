
import streamlit as st
st.set_page_config(page_title="AI이행봇 - VDC-A Q&A", page_icon="🤖")

import os
import json
import pandas as pd
import requests
import tempfile
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain

from create_process_retrievers import create_process_retrievers

# 상단 소개 영역
st.title("🤖 AI이행봇")
st.subheader("현재 지원하는 단계: `VDC-A`")
st.markdown("""
본 챗봇은 **AI사업의 제안 검토 프로세스를 지원하는 Q&A 시스템**입니다.  
현재는 `VDC-A` 단계에 특화되어 있으며, 향후 선행 단계 및 이후 절차까지 순차적으로 확장될 예정입니다.  
추후 전체 프로세스 흐름도 함께 안내될 수 있도록 구성될 예정입니다.

📄 참고 문서:
- [VDC_A_통합.pdf](https://drive.google.com/file/d/1cEFCFC7fp3JuDRgdPS3BdJuHPKhLF3yn/view)
- [VDC_A_분할.pdf](https://drive.google.com/file/d/1RD1bcaS_BPj8tP76GHE6bTbHuMMLmGeF/view)
- [VDC_A_QNA.pdf](https://drive.google.com/file/d/1vckiOua15aD-olMGwV2fDz8wvELCnU6o/view)
""")

# Load 주제 커버리지
with open("vdc_coverage.json", "r", encoding="utf-8") as f:
    topic_data = json.load(f)
    supported = topic_data.get("supported_topics", [])
    unsupported = topic_data.get("unsupported_topics", [])

max_len = max(len(supported), len(unsupported))
supported += [""] * (max_len - len(supported))
unsupported += [""] * (max_len - len(unsupported))
coverage_df = pd.DataFrame({
    "✅ 현재 답변 가능한 주제": supported,
    "❌ 현재 미지원 주제": unsupported
})
st.markdown("### 📚 문서 커버리지 요약")
st.dataframe(coverage_df, use_container_width=True, hide_index=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None or openai_api_key.strip() == "":
    st.error("OPENAI API 키가 설정되지 않았습니다. .env 파일 또는 환경변수에서 OPENAI_API_KEY를 확인하세요.")
    st.stop()

# 프롬프트 정의
vdc_supported_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
당신은 SI 제안사업의 내부 전략 회의인 VDC-A 단계에 특화된 컨설턴트입니다.
사용자의 질문에 대해 반드시 다음 문서에서 관련 절차, 기준, 또는 실무 적용 사례를 참고하여 답변하세요.
출처 없는 일반적인 추론은 삼가고, 회사 내부 기준을 기반으로 설명하세요.

사용자가 질문한 내용:
{question}

아래는 관련 문서에서 추출된 근거입니다:
{context}

답변 형식은 다음과 같이 구성하세요:
1. 💡 핵심 조치 요약
2. 📋 해당 절차의 위치 및 단계 (예: 사전검토 > 회의자료작성)
3. 📎 참고 문서명 및 근거 요약
"""
)

vdc_unsupported_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
다음 사용자의 질문은 현재 제공된 문서들로는 답변이 어렵습니다.
회사의 다른 지침서나 관련 부서 담당자에게 확인해주시기 바랍니다.

질문: {question}

(※ 참고 문서 없음)

Context: {context}
"""
)

def load_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            loader = PyMuPDFLoader(tmp_file.name)
            return loader.load()
    else:
        print(f"❌ PDF 다운로드 실패: {url}")
        return []

def create_law_retrievers():
    urls = [
        "https://drive.google.com/uc?export=download&id=1cEFCFC7fp3JuDRgdPS3BdJuHPKhLF3yn",
        "https://drive.google.com/uc?export=download&id=1RD1bcaS_BPj8tP76GHE6bTbHuMMLmGeF",
        "https://drive.google.com/uc?export=download&id=1vckiOua15aD-olMGwV2fDz8wvELCnU6o"
    ]
    all_docs = []
    for url in urls:
        docs = load_pdf_from_url(url)
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = text_splitter.split_documents(all_docs)
    for doc in docs_split:
        doc.metadata["source_name"] = "remote_pdf"

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs_split, embeddings)
    return vectorstore.as_retriever()

def retrieve_answer_by_type(retriever, query, is_supported):
    llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key)
    if is_supported:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": vdc_supported_prompt},
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": query})
        return result["result"], result.get("source_documents", [])
    else:
        llm_chain = LLMChain(llm=llm, prompt=vdc_unsupported_prompt)
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context"
        )
        return chain.invoke({"input_documents": [], "question": query}), []

if "history" not in st.session_state:
    st.session_state["history"] = []

retriever = create_law_retrievers()

query = st.chat_input("질문을 입력하세요:")

# 🔧 디버깅: 벡터 리트리버가 반환하는 문서 수와 내용 확인
if query.lower().strip() in ["vdc-a 절차", "vdc 절차"]:
    st.markdown("## 🐞 디버깅: 리트리버 검색 결과")
    docs = retriever.get_relevant_documents(query)
    st.write(f"🔍 관련 문서 수: {len(docs)}")
    for i, doc in enumerate(docs):
        st.markdown(f"**문서 {i+1}:** `{doc.metadata.get('source_name', 'N/A')}`")
        st.code(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""), language="markdown")

if query:
    is_supported = True
    for u in topic_data.get("unsupported_topics", []):
        if u in query:
            is_supported = False
            break
    with st.spinner("답변 생성 중..."):
        answer, sources = retrieve_answer_by_type(retriever, query, is_supported)
        st.session_state["history"].append((query, answer, sources, is_supported))

for q, a, sdocs, supported in st.session_state["history"]:
    st.chat_message("user").write(q)
    if supported:
        st.chat_message("assistant").markdown(f"""
### 💡 핵심 조치 요약

{a.strip()}

---
### 📋 해당 절차의 위치 및 단계

_(상세 내용은 문서 참조)_

---
### 📎 참고 문서명 및 근거 요약

{len(sdocs)}개 문서에서 근거가 추출되었습니다. 아래 '관련 문서 블록 보기'를 참고하세요.
""")
        if sdocs:
            with st.expander("📎 관련 문서 블록 보기"):
                for i, doc in enumerate(sdocs):
                    st.markdown(f"**[{i+1}]** `{doc.metadata.get('source_name', '알 수 없음')}`")
                    st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
    else:
        text = a["output_text"] if isinstance(a, dict) and "output_text" in a else a
        st.chat_message("assistant").warning(text)

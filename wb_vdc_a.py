import streamlit as st
import os
import json
import tempfile
import requests
import numpy as np
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
st.set_page_config(page_title="VDC-A 임베딩 유사 질문 연결", page_icon="🧠")
st.title("🤖 VDC-A 문서 기반 + 벡터 유사 Q&A 시스템")

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

# QNA 불러오기
with open("vdc_a_대표질문.json", "r", encoding="utf-8") as f:
    qna = json.load(f)
qna_questions = [q["question"] for q in qna]

# QNA 임베딩 벡터 생성
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
qna_vectors = embeddings.embed_documents(qna_questions)

# 벡터 기반 유사 질문 연결 함수
def find_most_similar_qna_vector(user_question, qna_data, qna_vectors, threshold=0.82):
    user_vec = embeddings.embed_query(user_question)
    sims = cosine_similarity([user_vec], qna_vectors)[0]
    best_idx = int(np.argmax(sims))
    if sims[best_idx] >= threshold:
        return qna_data[best_idx]
    return None

# 문서 기반 리트리버
def load_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(response.content)
            f.flush()
            loader = PyMuPDFLoader(f.name)
            return loader.load()
    return []

def create_pdf_vector_retriever():
    urls = {
        "vdc_a_프로세스": "https://drive.google.com/uc?export=download&id=1lSEWk7KDgHR71yHcjEKniWzhqwL7T2fC",
        "vdc_a_qna": "https://drive.google.com/uc?export=download&id=1KGJv9ttGD7ErcSWymE-0jiMjOzbnq6iI"
    }
    all_docs = []
    for name, url in urls.items():
        docs = load_pdf_from_url(url)
        for doc in docs:
            doc.metadata["source_name"] = name
        all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(all_docs)
    vectordb = FAISS.from_documents(split_docs, embeddings)
    return vectordb.as_retriever()

retriever = create_pdf_vector_retriever()

# 프롬프트 정의
hybrid_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 VDC-A 문서 기반 전문가입니다.

이전 대화 문맥과 질문:
{question}

관련 문서 내용:
{context}

답변 형식:
1. 💡 핵심 요약
2. 📋 관련 절차 위치 또는 판단 주체
"""
)

# 챗봇 이력
if "history" not in st.session_state:
    st.session_state["history"] = []

query = st.chat_input("VDC-A 관련 질문을 입력하세요:")
if not query:
    st.stop()

# QNA 유사도 매칭 (임베딩 기반)
matched_qna = find_most_similar_qna_vector(query, qna, qna_vectors)

if matched_qna:
    # QNA 직접 응답
    st.session_state["history"].append((query, matched_qna["answer"], []))
else:
    # 문서 기반 GPT 사용
    chat_history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in st.session_state["history"][-3:]])
    contextual_query = chat_history_text + f"\nQ: {query}"

    llm = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": hybrid_prompt},
        return_source_documents=True
    )
    with st.spinner("문서 기반 답변 생성 중..."):
        result = qa_chain.invoke({"query": contextual_query})
        st.session_state["history"].append((query, result["result"], result["source_documents"]))

# 채팅 출력
for q, a, sources in st.session_state["history"]:
    st.chat_message("user").write(q)
    st.chat_message("assistant").markdown(f"### 💡 핵심 요약\n{a.strip()}")
    if sources:
        with st.expander("📎 문서 근거 보기"):
            for i, doc in enumerate(sources):
                st.markdown(f"**[{i+1}]** `{doc.metadata.get('source_name', 'unknown')}`")
                st.code(doc.page_content[:400] + ("..." if len(doc.page_content) > 400 else ""))
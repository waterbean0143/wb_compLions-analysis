import os
import streamlit as st
import gdown
from pathlib import Path
from langchain.schema import Document, SystemMessage, HumanMessage
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

os.environ["OPENAI_API_KEY"] = ""

# --- 0. Google Drive 파일 설정 ---
st.set_page_config(page_title="Ensemble Retriever Chatbot", layout="wide")
VDC_DRIVE_IDS = [
    '1Rre8JcRdym4tc8xUWM80I0AlCcnJlBVS',
]

QA_DRIVE_IDS = [
    '1E_HR9q8bQT7bEcptXRV55UXdnW9QMJ7k',
]
IMAGE_DRIVE_IDS = {
    "14": '13VbBQi0_jOc5YrOskr820tJf6VpUvUcD', #14라는 키워드가 나오면 해당 이미지 파일이 나오도록 설정함.
}

# 다운로드 폴더 생성
os.makedirs('pdfs', exist_ok=True)
os.makedirs('images', exist_ok=True)

# --- 파일 다운로드 함수 ---
@st.cache_resource(ttl=3600*24)
def download_pdfs(ids):
    paths = []
    for fid in ids:
        out = f'pdfs/{fid}.pdf'
        if not os.path.exists(out):
            gdown.download(f'https://drive.google.com/uc?id={fid}', out, quiet=True)
        paths.append(out)
    return paths

@st.cache_resource(ttl=3600*24)
def download_images(map_ids):
    img_map = {}
    for kw, fid in map_ids.items():
        out = f'images/{kw}.png'
        if not os.path.exists(out):
            gdown.download(f'https://drive.google.com/uc?id={fid}', out, quiet=True)
        img_map[kw] = out
    return img_map

# PDF 및 이미지 다운로드
vdc_paths = download_pdfs(VDC_DRIVE_IDS)
qa_paths = download_pdfs(QA_DRIVE_IDS)
all_pdf_paths = vdc_paths + qa_paths
IMAGE_MAP = download_images(IMAGE_DRIVE_IDS)

# --- 문서 로드 및 청크 분할 ---
@st.cache_resource
def prepare_chunks(pdf_paths):
    raw_docs = []
    for path in pdf_paths:
        loader = PyMuPDFLoader(path)
        for d in loader.load():
            # 메타데이터 구분
            d.metadata['source_type'] = 'VDC' if any(fid in path for fid in VDC_DRIVE_IDS) else 'Q&A'
            content = d.page_content
            if '///' in content:
                # 분할자 기준으로 우선 분할
                for part in content.split('///'):
                    raw_docs.append(Document(page_content=part, metadata=d.metadata))
            else:
                raw_docs.append(d)
    # 나머지는 사이즈 기반 분할
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(raw_docs)

chunks = prepare_chunks(all_pdf_paths)

# --- 임베딩 및 Retriever 빌드 ---
@st.cache_resource(ttl=3600*24)
def build_retrievers(_chunks):
    emb = OpenAIEmbeddings(model="text-embedding-ada-002")
    faiss_store = FAISS.from_documents(_chunks, embedding=emb)
    faiss_r = faiss_store.as_retriever(search_kwargs={"k": 5})

    # BM25Retriever.from_documents를 사용해 vectorizer까지 같이 초기화
    bm25_r = BM25Retriever.from_documents(_chunks, k=5)

    return EnsembleRetriever(retrievers=[faiss_r, bm25_r], weights=[0.6, 0.4])


ensemble = build_retrievers(chunks)
chat = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# --- 질의 처리 함수 ---
def query_fn(query: str):
     # 1) 문서 검색 & 답변 생성
    docs = ensemble.get_relevant_documents(query)
    context = "\n\n".join([f"[{d.metadata['source_type']}] {d.page_content}" for d in docs])

    msgs = [
        SystemMessage(content=(
            "You are a helpful assistant. "
            "Answer the question using ONLY the provided context. "
            "Do NOT use any information outside of that context. "
            "If the answer cannot be found in the context, respond with "
            "\"죄송하지만 제공된 문서에 해당 정보가 없습니다.\""
        )),
         HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
     ]
    answer = chat(msgs).content

    if '14' in answer:
         img = IMAGE_MAP.get('14')
    else:
         # 키워드 기반 나머지 이미지
         img = next((path for kw, path in IMAGE_MAP.items() if kw.lower() in query.lower()), None)

    return answer, img


# --- Streamlit UI ---
st.title("Google Drive 기반 Ensemble Retriever Chatbot")
st.write("VDC 설명 및 Q&A 문서를 모두 고려하여 답변을 생성합니다.")
query = st.text_input("질문을 입력하세요")
if st.button("질문하기") and query:
    with st.spinner("답변 생성 중..."):
        ans, img_path = query_fn(query)
    st.subheader("Answer")
    st.write(ans)
    if img_path:
        st.image(img_path, caption="키워드 이미지")
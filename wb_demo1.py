import streamlit as st
import requests
import tempfile
import os
import pandas as pd
import re
from datetime import datetime, timezone, timedelta
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from io import BytesIO
from kiwipiepy import Kiwi
from langgraph.graph import END, StateGraph
from langchain_upstage import UpstageGroundednessCheck
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.document_transformers import LongContextReorder
from sklearn.metrics.pairwise import cosine_similarity
from typing import TypedDict, Dict, List, Tuple
import uuid
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import asyncio
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from functools import partial
import threading
import openai

global memory
global case_vectordbs
global case_docs
global for_show_law_vectordbs
global law_docs
global law_vectordbs
global similar_cases_db
global fcpa_retrievers 

similar_cases_db = {}
fcpa_retrievers = {} 

executor = ThreadPoolExecutor(max_workers=5)

fcpa_docs = []
fcpa_vectordbs = {}
law_docs = []
law_vectordbs = {}
case_vectordbs = {}
case_docs = []
for_show_law_vectordbs = {}
selected_for_show_law_vectordbs = {}
law_retrievers = {}
case_retrievers = {}

bm25_weight = 0.3
faiss_weight = 0.7

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 페이지 설정
st.set_page_config(page_title="AI 이행 컴플라이언스 봇", page_icon="🐧")

# Kiwi 초기화
kiwi = Kiwi()

# 사용자 인증 정보
users = {
    "10128722": {"password": "10128722", "name": "이광준"},
    "10154372": {"password": "10154372", "name": "김도완"},
    "10154502": {"password": "10154502", "name": "김나현"},
    "10154455": {"password": "10154455", "name": "이령현"},
    "10030124": {"password": "10030124", "name": "김근우"},
    "10050490": {"password": "10050490", "name": "이동원"},
    "10053105": {"password": "10053105", "name": "윤재호"},
    "10054788": {"password": "10054788", "name": "이민아"},
    "10073609": {"password": "10073609", "name": "양수경"},
    "10027921": {"password": "10027921", "name": "허주영"},
    "10083224": {"password": "10083224", "name": "김상동"},
    "10076957": {"password": "10076957", "name": "신유식"},
    "10133827": {"password": "10133827", "name": "최형윤"},
    "10116407": {"password": "10116407", "name": "김종선"},
    "10106703": {"password": "10106703", "name": "윤형섭"},
    "10156389": {"password": "10156389", "name": "이희선"},
    "10155624": {"password": "10155624", "name": "박내희"},
    "10155900": {"password": "10155900", "name": "양희정"},
    "10153729": {"password": "10153729", "name": "조영주"},
    "10156380": {"password": "10156380", "name": "지석준"},
    "10155306": {"password": "10155306", "name": "함석주"},
    "10154620": {"password": "10154620", "name": "권혁노"},
    "10155598": {"password": "10155598", "name": "임현주"},
    "10155623": {"password": "10155623", "name": "이경진"},
    "10074201": {"password": "10074201", "name": "김주섭"},
    "10143967": {"password": "10143967", "name": "임태은"},
    "10121591": {"password": "10121591", "name": "박성수"},
    "10110686": {"password": "10110686", "name": "김성식"},
    "10153632": {"password": "10153632", "name": "하원"},
    "10153055": {"password": "10153055", "name": "노경환"},
    "10153040": {"password": "10153040", "name": "강신구"},
    "10152758": {"password": "10152758", "name": "임채일"},
    "10085421": {"password": "10085421", "name": "전현호"},
    "10107793": {"password": "10107793", "name": "최양주"},
    "10154423": {"password": "10154423", "name": "이수영"},
    "10080462": {"password": "10080462", "name": "조혜진"},
    "10076132": {"password": "10076132", "name": "김은미"},
    "10143947": {"password": "10143947", "name": "임찬"},
    "10149765": {"password": "10149765", "name": "장지인"},
    "10155588": {"password": "10155588", "name": "이창엽"},
    "10155603": {"password": "10155603", "name": "홍주연"},
    "10155592": {"password": "10155592", "name": "정채영"},
    "10107793": {"password": "10107793", "name": "최양주"},
    "10151095": {"password": "10151095", "name": "신유현"},
    "10085454": {"password": "10085454", "name": "이용우"},
    "10067133": {"password": "10067133", "name": "이건복"},
    "10061208": {"password": "10061208", "name": "이기혁"},
    "10156201": {"password": "10156201", "name": "이상인"},
    "10153631": {"password": "10153631", "name": "김종명"},
    "10154371": {"password": "10154371", "name": "배수빈"},
    "10154440": {"password": "10154440", "name": "권세희"},
    "10153090": {"password": "10153090", "name": "박경일"},
    "10023332": {"password": "10023332", "name": "김봉규"},
    "10097768": {"password": "10097768", "name": "곽정환"},
    "10084629": {"password": "10084629", "name": "성문관"},
    "10156208": {"password": "10156208", "name": "홍상기"},
    "10154342": {"password": "10154342", "name": "권동희"},
    "10013698": {"password": "10013698", "name": "황인석"},
    "10156350": {"password": "10156350", "name": "박영준"},
    "10156378": {"password": "10156378", "name": "이종구"},
    "10098527": {"password": "10098527", "name": "김종혁"},
    "10082909": {"password": "10082909", "name": "이미희"},
    "10146454": {"password": "10146454", "name": "양송희"},
    "10143714": {"password": "10143714", "name": "윤영훈"},
    "10148692": {"password": "10148692", "name": "이동훈"},
    "10153370": {"password": "10153370", "name": "김세환"},
    "10028389": {"passowrd": "10028389", "name": "진호섭"},
    "10079098": {"password": "10079098", "name": "전관용"},
    "10058650": {"password": "10058650", "name": "김은성"},
    "10142602": {"password": "10142602", "name": "강보문"},
    "admin": {"password": "admin", "name": "관리자"},
    "test": {"password": "test", "name": "테스트 사용자"},
    "10139784": {"password": "10139784", "name": "이홍철"},
    "10014632": {"password": "10014632", "name": "배기동"},
    "10132778": {"password": "10132778", "name": "이경로"},
    "10012384": {"password": "10012384", "name": "김대현"},
    "10063841": {"password": "10063841", "name": "최준혁"},
    "10150440": {"password": "10150440", "name": "이민지"},
    "10153591": {"password": "10153591", "name": "이기찬"},
    "10122965": {"password": "10122965", "name": "배원탁"},
    "10143675": {"password": "10143675", "name": "유지원"},
    "10154458": {"password": "10154458", "name": "백종안"},
    "10154352": {"password": "10154352", "name": "오지윤"},
}

# 법령 및 사례 PDF 파일 URL
LAW_PDF_URLS = {
    "청탁금지법": "https://drive.google.com/uc?export=download&id=1ZxmCf7dOEd8Y8pYp9ojRgXDhxsitGF44",
    "중대재해처벌법": "https://drive.google.com/uc?export=download&id=1AvA8fwxChGkNcE4O34R088F5sZz1Sbwz",
    "산업안전보건법": "https://drive.google.com/uc?export=download&id=1uO_uTf1xIpa87MRkuUQnVilCw1GPIjeQ",
    "하도급법": "https://drive.google.com/uc?export=download&id=1RNeYXHY1zENKXF9J6J_G3YgPboJt02lg",
    "상생협력법": "https://drive.google.com/uc?export=download&id=183pgpXkYbtmacFcdnUQdc5uvvpZNE6Im",
    "공정거래법": "https://drive.google.com/uc?export=download&id=11SWqG4p7WNY4Gb4pJdkl8dElMYszU6Pz",
    "정보통신공사업법": "https://drive.google.com/uc?export=download&id=1qRCnYXa6Vcp3VOh4vajOzaKpeHsotH-f",
    "국가계약법": "https://drive.google.com/uc?export=download&id=1YYTe7UXkCkf0coGZ_0zl7Cz5YCUf3W4S",
    "소프트웨어진흥법": "https://drive.google.com/uc?export=download&id=1spVRGsFELrvy7Cs4vJdplAMpiYuqcxTq",
    "회사 내규(부패방지 행동강령 및 윤리경영원칙 실천지침)":"https://drive.google.com/uc?export=download&id=1OsQL7g81l74vULtzniYrOoZFpDyngqo4",
}
CASE_PDF_URLS = {
    "청탁금지법":"https://drive.google.com/uc?export=download&id=1GNJOF_4iD3JGKib2LrbnzEECnHrtKD6g",
    "중대재해처벌법":"https://drive.google.com/uc?export=download&id=1NE8o8XWJxfXb2yCVan66ZeexzsFTV81F",
    "산업안전보건법":"https://drive.google.com/uc?export=download&id=1s4szmiDuCUvf8KaN7AbmCwQSVMYj6X5Q",
    "하도급법":"https://drive.google.com/uc?export=download&id=1FlP338Gz42-w2aXGC7WcTx38rJ-Zp0M3",
    "상생협력법":"https://drive.google.com/uc?export=download&id=1At3yWegX8fTqebWCeCY5wPmKyIioHQJw",
    "공정거래법":"https://drive.google.com/uc?export=download&id=1x3PG4zug0-ALHcLyDeJOLdb4FwvFP8TB",
    "정보통신공사업법": "", 
    "국가계약법":"https://drive.google.com/uc?export=download&id=1zm61TCNuZFW6cdjAkrzT-PJ6RA4rNpla",
    "소프트웨어진흥법": "" ,
    "회사 내규(부패방지 행동강령 및 윤리경영원칙 실천지침)":"", 
}
FOR_SHOW_LAW_PDF_URLS = {
    "청탁금지법": "https://drive.google.com/uc?export=download&id=1Eqx5xE8ewtoFkWG3GDQuivURXRmE2hIZ",
    "중대재해처벌법": "https://drive.google.com/uc?export=download&id=1h8C4GAUOHsqB5uP1MxrKLgovubxLUFsJ",
    "산업안전보건법": "https://drive.google.com/uc?export=download&id=1r2WOFNEzPy0pnANAZ486H8MFbeN4YMhD",
    "하도급법": "https://drive.google.com/uc?export=download&id=1-u6DlMQVQ7qe1DsJoHLisn1EbXuGEFHo",
    "상생협력법": "https://drive.google.com/uc?export=download&id=1iI1GG6Ob2o-rbZ1qfjPOZ0s8RZplJxy9",
    "공정거래법": "https://drive.google.com/uc?export=download&id=1k7WUJX8geVZb-0mH8sWZG5KUMsLLnCoQ",
    "정보통신공사업법": "https://drive.google.com/uc?export=download&id=1mPWlPDjp3MS26b7sqex-dlKxvIOZIOZ5",
    "국가계약법": "https://drive.google.com/uc?export=download&id=1YYTe7UXkCkf0coGZ_0zl7Cz5YCUf3W4S",
    "소프트웨어진흥법": "https://drive.google.com/uc?export=download&id=1vK2k7l6PoNbL63u1dnfFAiPp6_wuHyUR",
    "회사 내규(부패방지 행동강령 및 윤리경영원칙 실천지침)":"https://drive.google.com/uc?export=download&id=1OsQL7g81l74vULtzniYrOoZFpDyngqo4",
}
FCPA_PDF_URLS = {
    "FCPA적용대상": "https://drive.google.com/uc?export=download&id=1C95Vfr660bltDYqoz6Tb13yMm2lfgzbd"
}
def check_password():
    def password_entered():
        if (
            st.session_state["username"] in users
            and st.session_state["password"] == users[st.session_state["username"]]["password"]
        ):
            st.session_state["password_correct"] = True
            st.session_state["logged_in_user"] = users[st.session_state["username"]]["name"]
            st.session_state["is_admin"] = st.session_state["username"] == "admin"
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.markdown("---")
        # 사용방법 expander
        with st.expander("📋 사용방법", expanded=False):
            st.markdown("""
            <ol>
                <li>로그인하기 (ID/PW는 본인 사번입니다.)</li>
                <li>챗봇에 메시지를 입력하여 질문을 던집니다.(❗만약, 실제 법률과 관련된 질문을 넣었는데 법과 관련이 없다라는 답변을 받게 된다면, "기존 질문 + ~~~ 법률 검토를 해주세요."라는 식으로 수정하여 다시 질문을 해보세요.)</li>
                <li>답변에 대한 피드백을 남깁니다. (필수는 아니나, 따로 피드백을 주고 싶은 부분이 있다면 남겨주시면 되십니다.)</li>
                <li>최종적으로 이용을 한 후기를, 왼쪽 sidebar하단에 '전반적인 사용후기'에 적어주세요.</li>
                <li>로그아웃합니다.</li>
            </ol>
            """, unsafe_allow_html=True)
        
        # Tip expander
        with st.expander("💡 활용 Tip", expanded=False):
            st.markdown("""
            <ol>
                <li>질문 유형에 따라 답변의 형식이 다르게 나옵니다. 질문의 상황에 맞게 유형을 선택해야 더욱 정확한 답변을 얻으실 수 있을 것입니다.</li>
                <li>법률 9가지 + 회사 내규 1가지가 근거 문서로 제공되어있습니다.
                    <ul>
                        <li>법령, 시행령, 사례: 청탁금지법, 하도급법, 상생협력법, 산업안전보건법, 공정거래법, 중대재해처벌법 </li>
                        <li>법령, 사례: 국가계약법</li>
                        <li>법령: 정보통신공사업법, 소프트웨어진흥법, 회사내규 </li>
                    </ul>
                        * 활용 데이터가 많을 수록 정확도가 높을 가능성이 있습니다.
                </li>
                <li>질문은 할 때마다 비용이 듭니다. 신중하게 질문을 해주시면 감사하겠습니다.</li>
                <li>정확도 향상을 위해 질문 주제가 바뀐다면, '새로운 대화 주제' 버튼을 클릭해주세요. </li>
                <li>비용과 보안 이슈로, 법과 관련된 질문이 아니면 답을 하지 않도록 필터링해놓았습니다. </li>
            </ol>
            """, unsafe_allow_html=True)

        with st.expander("🔍 연속적인 질문 TIP", expanded=False):
            st.markdown("""
            <ol>
                <li><strong>이전 답변 참조하기<br>
                이전 답변을 참조하여 연속성을 유지하세요.<br>
                (예: "이전 답변에서 언급된 '부정청탁'의 범위에 대해 더 자세히 설명해 주시고, 이 사례가 해당되는지 분석해 주세요.")</li>
                <li><strong>답변에서 제공되는 '질문 TIP'의 내용을 활용하여 질문 구체화하기<br>
                '질문 TIP'에서 제시된 추가 정보나 상황을 직접 질문에 포함시키세요.<br>
                (예: "이전에 언급하신 A씨의 정확한 직위는 공공기관의 과장이며, 업무적 관계에서 발생한 상황입니다. 이 경우 청탁금지법 적용 여부를 알고 싶습니다.")</li>
                <li><strong>법률 용어 사용하기<br>
                질문에 관련 법률 용어를 포함시켜 법률 관련성을 높이세요.<br>
                (예: "청탁금지법상 '직무관련성'과 '대가성'의 기준에 대해 설명해주시고, 이 사례에 적용해 주세요.")</li>
                <li><strong>법률 조항 언급하기<br>
                특정 법률 조항을 언급하여 질문의 구체성을 높이세요. <br>
                (예:"청탁금지법 제8조와 관련하여, 이 상황에서 '직무관련성'이 인정될 수 있는지 설명해주세요.")</li>
        
            </ol>
            
            """, unsafe_allow_html=True)
            
    elif not st.session_state["password_correct"]:
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("😕 알 수 없는 사용자이거나 비밀번호가 틀립니다.")
        return False
    else:
        return True

def save_question(user_id, question, selected_laws):
    filename = 'user_questions.csv'
    kst = timezone(timedelta(hours=9))
    timestamp = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
    selected_laws_str = ", ".join(selected_laws)
    new_question = pd.DataFrame([[user_id, timestamp, question, selected_laws_str]], 
                                columns=['User ID', 'Timestamp', 'Question', 'Selected Laws'])
    
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df = pd.concat([df, new_question], ignore_index=True)
    else:
        df = new_question
    
    df.to_csv(filename, index=False, encoding='utf-8-sig')

# 텍스트 전처리
def preprocess_text(text):
    result = kiwi.analyze(text)
    keywords = [token.form for token in result[0][0] if token.tag.startswith('N') or token.tag.startswith('V') or token.tag.startswith('MA')]
    return ' '.join(keywords)

# 법령별 리트리버 설정
def create_law_retrievers(selected_law_vectordbs):
    law_retrievers = {}
    embeddings = OpenAIEmbeddings()
    reordering = LongContextReorder()
    
    for law_name, vectordb in selected_law_vectordbs.items():
        faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 10})
        
        # 해당 법령에 대한 문서 필터링
        law_specific_docs = [doc for doc in law_docs if doc.metadata['law_name'] == law_name]
        
        if law_specific_docs:  # 문서가 있는 경우에만 BM25Retriever 생성
            bm25_retriever = BM25Retriever.from_documents(law_specific_docs)
            bm25_retriever.k = 10
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], weights=[bm25_weight, faiss_weight]
            )
        else:
            # 문서가 없는 경우 FAISS 리트리버만 사용
            ensemble_retriever = faiss_retriever
        
        def retrieve_and_rerank(query, retriever=ensemble_retriever, embed=embeddings, reorder=reordering):
            preprocessed_query = preprocess_text(query)
            docs = retriever.get_relevant_documents(preprocessed_query)
            
            # 빈 문서 제거
            docs = [doc for doc in docs if doc.page_content.strip()]
            
            if not docs:
                return []  # 문서가 없으면 빈 리스트 반환
            
            query_embedding = embed.embed_query(preprocessed_query)
            doc_embeddings = embed.embed_documents([doc.page_content for doc in docs])
            
            # NaN 또는 inf 값 검사 및 제거
            valid_indices = np.isfinite(doc_embeddings).all(axis=1)
            valid_docs = [doc for doc, is_valid in zip(docs, valid_indices) if is_valid]
            valid_embeddings = [emb for emb, is_valid in zip(doc_embeddings, valid_indices) if is_valid]
            
            if not valid_docs:
                return []  # 유효한 문서가 없으면 빈 리스트 반환
            
            try:
                similarities = cosine_similarity([query_embedding], valid_embeddings)[0]
                reranked_docs = [doc for _, doc in sorted(zip(similarities, valid_docs), key=lambda x: x[0], reverse=True)]
                return reorder.transform_documents(reranked_docs[:3])
            except ValueError as e:
                print(f"Error in cosine_similarity calculation: {e}")
                return []  # 오류 발생 시 빈 리스트 반환
        
        law_retrievers[law_name] = retrieve_and_rerank
    
    return law_retrievers


# 문서 분할
@st.cache_data(ttl=3600*24)
def fcpa_split_docs(_docs):
    text_splitter = CharacterTextSplitter(
        separator="////",
        chunk_size=600,
        chunk_overlap=0,
    )
    return text_splitter.split_documents(_docs)

@st.cache_data(ttl=3600*24)
def law_split_docs(_docs):
    text_splitter = CharacterTextSplitter(
        separator="\n\n\n\n",
        chunk_size=1000,
        chunk_overlap=500,
    )
    return text_splitter.split_documents(_docs)

@st.cache_data(ttl=3600*24)
def case_split_docs(_docs):
    text_splitter = CharacterTextSplitter(
        separator="\n\n\n\n",
        chunk_size=4500,
        chunk_overlap=0,
    )
    return text_splitter.split_documents(_docs)

@st.cache_data(ttl=3600*24)
def show_law_split_docs(_docs):
    text_splitter = CharacterTextSplitter(
        separator="\n\n\n\n",
        chunk_size=4500,
        chunk_overlap=0,
    )
    return text_splitter.split_documents(_docs)

# 벡터 데이터베이스 로딩
@st.cache_resource(ttl=3600*24)
def load_fcpa_vectordbs(_splits):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    fcpa_vectordbs = {}
    for fcpa_name in FCPA_PDF_URLS.keys():
        fcpa_splits = [split for split in _splits if split.metadata['fcpa_name'] == fcpa_name]
        if fcpa_splits:
            try:
                fcpa_vectordbs[fcpa_name] = FAISS.from_documents(documents=fcpa_splits, embedding=embedding)
                print(f"Successfully created FCPA vector database for '{fcpa_name}'")
            except Exception as e:
                print(f"Error creating FCPA vector database for '{fcpa_name}': {e}")
                empty_doc = Document(
                    page_content=f"FCPA 문서 '{fcpa_name}' 벡터화에 실패했습니다.",
                    metadata={'fcpa_name': fcpa_name, 'source': fcpa_name, 'page': 0}
                )
                fcpa_vectordbs[fcpa_name] = FAISS.from_documents([empty_doc], embedding=embedding)
    return fcpa_vectordbs

@st.cache_resource(ttl=3600*24)
def load_law_vectordbs(_splits):
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    law_vectordbs = {}
    for law_name in LAW_PDF_URLS.keys():
        law_splits = [split for split in _splits if split.metadata['law_name'] == law_name]
        law_vectordbs[law_name] = FAISS.from_documents(documents=law_splits, embedding=embedding)
    return law_vectordbs

@st.cache_resource(ttl=3600*24)
def load_case_vectordbs(_splits):
    global case_vectordbs
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    for law_name in CASE_PDF_URLS.keys():
        case_splits = [split for split in _splits if split.metadata['law_name'] == law_name]
        if case_splits:
            case_vectordbs[law_name] = FAISS.from_documents(documents=case_splits, embedding=embedding)
        else:
            case_vectordbs[law_name] = FAISS.from_texts(["빈 문서"], embedding=embedding)
    return case_vectordbs

@st.cache_resource(ttl=3600*24)
def load_for_show_law_vectordbs(_splits):
    global for_show_law_vectordbs
    for_show_law_vectordbs = {}
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    for for_show_law_name in FOR_SHOW_LAW_PDF_URLS.keys():
        for_show_law_splits = [split for split in _splits if split.metadata['for_show_law_name'] == for_show_law_name]
        if for_show_law_splits:
            for_show_law_vectordbs[for_show_law_name] = FAISS.from_documents(documents=for_show_law_splits, embedding=embedding)
        else:
            st.warning(f"'{for_show_law_name}'에 대한 문서가 없습니다. 빈 벡터 데이터베이스를 생성합니다.")
            for_show_law_vectordbs[for_show_law_name] = FAISS.from_texts(["빈 문서"], embedding=embedding)
    return for_show_law_vectordbs

# 사례별 리트리버 설정
def create_case_retrievers(selected_case_vectordbs):
    case_retrievers = {}
    embeddings = OpenAIEmbeddings()
    reordering = LongContextReorder()
    
    for law_name, vectordb in selected_case_vectordbs.items():
        if vectordb.docstore._dict:  # vectordb에 문서가 있는 경우
            faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 10})
            
            # 해당 법령에 대한 사례 문서 필터링
            law_specific_cases = [doc for doc in case_docs if doc.metadata['law_name'] == law_name and doc.page_content.strip()]
            
            if law_specific_cases:  # 사례 문서가 있는 경우에만 BM25Retriever 생성
                try:
                    bm25_retriever = BM25Retriever.from_documents(law_specific_cases)
                    bm25_retriever.k = 10
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, faiss_retriever], weights=[bm25_weight, faiss_weight]
                    )
                except Exception as e:
                    print(f"Error creating BM25Retriever for {law_name}: {e}")
                    ensemble_retriever = faiss_retriever
            else:
                # 사례 문서가 없는 경우 FAISS 리트리버만 사용
                ensemble_retriever = faiss_retriever
            
            def retrieve_and_rerank(query, retriever=ensemble_retriever, embed=embeddings, reorder=reordering):
                preprocessed_query = preprocess_text(query)
                docs = retriever.get_relevant_documents(preprocessed_query)
                
                # 빈 문서 제거
                docs = [doc for doc in docs if doc.page_content.strip()]
                
                if not docs:
                    return []  # 문서가 없으면 빈 리스트 반환
                
                query_embedding = embed.embed_query(preprocessed_query)
                doc_embeddings = embed.embed_documents([doc.page_content for doc in docs])
                
                # NaN 또는 inf 값 검사 및 제거
                valid_indices = np.isfinite(doc_embeddings).all(axis=1)
                valid_docs = [doc for doc, is_valid in zip(docs, valid_indices) if is_valid]
                valid_embeddings = [emb for emb, is_valid in zip(doc_embeddings, valid_indices) if is_valid]
                
                if not valid_docs:
                    return []  # 유효한 문서가 없으면 빈 리스트 반환
                
                try:
                    similarities = cosine_similarity([query_embedding], valid_embeddings)[0]
                    reranked_docs = [doc for _, doc in sorted(zip(similarities, valid_docs), key=lambda x: x[0], reverse=True)]
                    return reorder.transform_documents(reranked_docs[:3])
                except ValueError as e:
                    print(f"Error in cosine_similarity calculation: {e}")
                    return []  # 오류 발생 시 빈 리스트 반환
            
            case_retrievers[law_name] = retrieve_and_rerank
        else:
            # 사례집이 없는 경우 더미 데이터 반환
            case_retrievers[law_name] = lambda x: [Document(page_content="이 법령에 대한 사례 정보가 없습니다.", metadata={"source": law_name})]
    
    return case_retrievers

def create_fcpa_retrievers(fcpa_vectordbs):
    fcpa_retrievers = {}
    embeddings = OpenAIEmbeddings()
    reordering = LongContextReorder()
    
    for fcpa_name, vectordb in fcpa_vectordbs.items():
        faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 1})
        
        # 해당 FCPA 문서에 대한 필터링
        fcpa_specific_docs = [doc for doc in fcpa_docs if doc.metadata.get('fcpa_name') == fcpa_name]
        
        if fcpa_specific_docs and len(fcpa_specific_docs) > 0 and fcpa_specific_docs[0].page_content != "FCPA 문서 로드 실패":
            # 문서가 있고 유효한 경우에만 BM25Retriever 생성
            try:
                bm25_retriever = BM25Retriever.from_documents(fcpa_specific_docs)
                bm25_retriever.k = 1
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever], weights=[1, 0]
                )
            except Exception as e:
                print(f"Error creating BM25Retriever for {fcpa_name}: {e}")
                # BM25Retriever 생성 실패 시 FAISS 리트리버만 사용
                ensemble_retriever = faiss_retriever
        else:
            # 문서가 없거나 로드 실패한 경우 FAISS 리트리버만 사용
            ensemble_retriever = faiss_retriever
        
        def retrieve_and_rerank(query, retriever=ensemble_retriever, embed=embeddings, reorder=reordering):
            preprocessed_query = preprocess_text(query)
            
            try:
                docs = retriever.get_relevant_documents(preprocessed_query)
                
                # 빈 문서 또는 실패 메시지 제거
                docs = [doc for doc in docs if doc.page_content.strip() and "FCPA 문서 로드 실패" not in doc.page_content]
                
                if not docs:
                    return []  # 문서가 없으면 빈 리스트 반환
                
                query_embedding = embed.embed_query(preprocessed_query)
                doc_embeddings = embed.embed_documents([doc.page_content for doc in docs])
                
                # NaN 또는 inf 값 검사 및 제거
                valid_indices = np.isfinite(doc_embeddings).all(axis=1)
                valid_docs = [doc for doc, is_valid in zip(docs, valid_indices) if is_valid]
                valid_embeddings = [emb for emb, is_valid in zip(doc_embeddings, valid_indices) if is_valid]
                
                if not valid_docs:
                    return []  # 유효한 문서가 없으면 빈 리스트 반환
                
                similarities = cosine_similarity([query_embedding], valid_embeddings)[0]
                
                # 유사도 임계값 설정 (필요하면 조정)
                threshold = 0.3
                filtered_docs_with_scores = [(doc, score) for doc, score in zip(valid_docs, similarities) if score > threshold]
                
                # 유사도로 정렬하고 상위 문서 선택
                reranked_docs = [doc for doc, _ in sorted(filtered_docs_with_scores, key=lambda x: x[1], reverse=True)]
                
                # 메타데이터에 유사도 점수 추가
                for doc, score in zip(reranked_docs, sorted(similarities, reverse=True)):
                    doc.metadata['score'] = score
                
                return reorder.transform_documents(reranked_docs[:3])
            
            except Exception as e:
                print(f"Error in FCPA retrieval/reranking: {e}")
                return []  # 오류 발생 시 빈 리스트 반환
        
        fcpa_retrievers[fcpa_name] = retrieve_and_rerank
    
    return fcpa_retrievers

def analyze_organization_type(org_name):
    global fcpa_retrievers
    
    if not fcpa_retrievers:
        fcpa_retrievers = create_fcpa_retrievers(fcpa_vectordbs)
    
    # 모든 FCPA 문서에서 관련 정보 검색
    org_docs = []
    for fcpa_name, retriever in fcpa_retrievers.items():
        try:
            search_query = f"{org_name}"
            retrieved_docs = retriever(search_query)
            org_docs.extend(retrieved_docs)
        except Exception as e:
            print(f"Error retrieving from {fcpa_name} for {org_name}: {e}")
    
    # 검색된 문서가 있는 경우 FCPA 문서 내용을 바탕으로 분석
    if org_docs:
        fcpa_context = "\n\n".join([doc.page_content for doc in org_docs])
        
        # 모델에 전달할 컨텍스트를 구성하고 분석 작업을 수행
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        org_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 법률 전문가입니다. 제공된 문서를 분석하여 
            주어진 조직이 1) 어떤 유형의 기관인지(국가기관, 지방자치단체, 교육청, 공공기관, 
            공직유관단체, 교육기관, 언론사, 정당, 국제기구, 민간기업 등), 
            2) 청탁금지법 적용 대상인지, 3) FCPA 적용 대상인지 판단하세요.
            
            문서에 해당 정보가 명시적으로 나와있지 않은 경우, 문서의 내용을 바탕으로 
            가장 합리적인 판단을 내리세요.
            
            답변 형식:
            기관유형: [유형]
            청탁금지법: [적용/미적용/불확실]
            FCPA: [적용/미적용/불확실]
            설명: [문서에 기반한 판단 근거]"""),
            ("human", f"조직명: {org_name}\n\n문서내용: {fcpa_context}")
        ])
        
        org_analysis_chain = org_analysis_prompt | llm
        org_analysis_result = org_analysis_chain.invoke({})
        
        return {
            "analysis": org_analysis_result.content,
            "fcpa_context": fcpa_context,
            "has_fcpa_data": True
        }
    else:
        # FCPA 문서에서 관련 정보를 찾지 못한 경우 일반적인 지식 기반 분석
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        general_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 법률 전문가입니다. 주어진 조직의 유형과 법률 적용 여부를 판단하세요.
            
            1. 한국 정부 기관(중앙행정기관, 지방자치단체, 교육청 등)
            2. 공공기관(공기업, 준정부기관, 기타공공기관)
            3. 공직유관단체
            4. 교육기관(국공립학교, 사립학교)
            5. 언론사
            6. 정당
            7. 국제기구
            8. 민간기업
            9. 기타 단체
            
            조직명을 보고 가장 적합한 유형을 선택하고, 해당 유형에 따라 일반적으로:
            - 청탁금지법 적용 대상인지 (1~7번 유형은 일반적으로 적용대상)
            - FCPA(미국 해외부패방지법) 적용 대상인지 (미국에 상장된 기업 또는 미국과 사업 관계가 있는 기업은 일반적으로 적용대상)
            
            답변 형식:
            기관유형: [유형]
            청탁금지법: [적용/미적용/불확실]
            FCPA: [적용/미적용/불확실]
            설명: [판단 근거] (문서에서 관련 정보를 찾지 못했으며, 일반적인 지식 기반으로 판단했음을 명시)"""),
            ("human", f"조직명: {org_name}")
        ])
        
        general_analysis_chain = general_analysis_prompt | llm
        general_analysis_result = general_analysis_chain.invoke({})
        
        return {
            "analysis": general_analysis_result.content + "\n\n(참고: FCPA 문서에서 직접적인 정보를 찾지 못했습니다.)",
            "fcpa_context": "관련 FCPA 문서 정보 없음",
            "has_fcpa_data": False
        }

def create_selected_law_vectordbs(selected_laws):
    global law_vectordbs
    selected_law_vectordbs = {}
    
    for law in selected_laws:
        # 대괄호 있는 경우 제거
        cleaned_law = law.strip('[]') if law.startswith('[') and law.endswith(']') else law
        
        if cleaned_law in law_vectordbs:
            # 원래 키(대괄호 포함)로 저장
            selected_law_vectordbs[law] = law_vectordbs[cleaned_law]
        else:
            print(f"Warning: No vector database found for {law}")
            # 벡터 데이터베이스가 없는 경우 빈 데이터베이스 생성
            embedding = OpenAIEmbeddings()
            selected_law_vectordbs[law] = FAISS.from_texts(["빈 문서"], embedding)
    
    return selected_law_vectordbs

def create_selected_case_vectordbs(selected_laws):
    global case_vectordbs
    selected_case_vectordbs = {}
    for law in selected_laws:
        if law not in case_vectordbs:
            # 해당 법령에 대한 벡터 데이터베이스가 없는 경우, 빈 데이터베이스 생성
            embedding = OpenAIEmbeddings()
            case_vectordbs[law] = FAISS.from_texts(["빈 문서"], embedding=embedding)
        selected_case_vectordbs[law] = case_vectordbs[law]
    return selected_case_vectordbs

# PDF 로딩 및 처리 함수들
@st.cache_data(ttl=3600*24)
def load_pdf_from_url(url):
    response = requests.get(url)
    return BytesIO(response.content)

@st.cache_data(ttl=3600*24)
def save_pdf_to_tempfile(pdf_bytes):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(pdf_bytes.read())
    temp_file.close()
    return temp_file.name
    
@st.cache_data(ttl=3600*24)
def load_image(url):
    response = requests.get(url)
    return BytesIO(response.content)

# 동의서 안내 문구 이미지
image_url = "https://drive.google.com/uc?export=download&id=1QqY6JQX7K5DcCOwPtqTmFHxnMrhhTlJL"
image = load_image(image_url)

@st.cache_data(ttl=3600*24)
def load_docs(url, law_name):
    pdf_bytes = load_pdf_from_url(url)
    temp_file_path = save_pdf_to_tempfile(pdf_bytes)
    loader = PyMuPDFLoader(temp_file_path)
    docs = loader.load()
    
    for doc in docs:
        doc.metadata['source'] = law_name  # URL 대신 법률 이름 저장
    
    return docs

# FCPA 문서 로드 함수
@st.cache_data(ttl=3600*24)
def load_fcpa_docs():
    global FCPA_PDF_URLS
    fcpa_docs = []
    
    # 1. URL에서 문서 로드
    for fcpa_name, url in FCPA_PDF_URLS.items():
        try:
            docs = load_docs(url, fcpa_name)
            for doc in docs:
                doc.metadata['fcpa_name'] = fcpa_name
            fcpa_docs.extend(docs)
            print(f"Successfully loaded FCPA document '{fcpa_name}' from URL")
        except Exception as e:
            print(f"Error loading FCPA document '{fcpa_name}' from URL: {e}")
            # 오류 발생 시 빈 문서 생성
            empty_doc = Document(
                page_content=f"FCPA 문서 '{fcpa_name}' 로드에 실패했습니다.", 
                metadata={'fcpa_name': fcpa_name, 'source': fcpa_name, 'page': 0}
            )
            fcpa_docs.append(empty_doc)
    
    # 2. 사용자 추가 데이터 로드
    fcpa_file_path = "fcpa_data.txt"
    if os.path.exists(fcpa_file_path):
        try:
            with open(fcpa_file_path, "r", encoding="utf-8") as file:
                content = file.read()
            
            # 텍스트에서 문서 생성
            custom_doc = Document(
                page_content=content,
                metadata={'fcpa_name': 'FCPA적용대상', 'source': 'user_added', 'page': 0}
            )
            fcpa_docs.append(custom_doc)
            print("Successfully loaded FCPA data from user-added file")
        except Exception as e:
            print(f"Error loading user-added FCPA data: {e}")
    
    return fcpa_docs


@st.cache_data(ttl=3600*24)
def load_law_docs():
    global LAW_PDF_URLS
    law_docs = []
    for law_name, url in LAW_PDF_URLS.items():
        docs = load_docs(url, law_name)
        for doc in docs:
            doc.metadata['law_name'] = law_name
        law_docs.extend(docs)
    return law_docs

@st.cache_data(ttl=3600*24)
def load_case_docs():
    global CASE_PDF_URLS
    case_docs = []
    for law_name, url in CASE_PDF_URLS.items():
        if url:  # URL이 비어있지 않은 경우에만 문서 로드
            docs = load_docs(url, law_name)
            for doc in docs:
                doc.metadata['law_name'] = law_name
            case_docs.extend(docs)
        
        # 추가된 사례 파일 로드
        case_file_path = f"case_{law_name}.txt"
        if os.path.exists(case_file_path):
            with open(case_file_path, "r", encoding="utf-8") as file:
                content = file.read()
            additional_cases = content.split("[[새로운 사례]]")[1:]
            for case in additional_cases:
                additional_doc = Document(page_content=case.strip(), metadata={'law_name': law_name, 'source': '추가된 사례'})
                case_docs.append(additional_doc)
        
        if not case_docs:
            # URL이 비어있고 추가된 사례도 없는 경우 빈 문서 생성
            empty_doc = Document(page_content="사례 정보가 없습니다.", metadata={'law_name': law_name, 'source': law_name, 'page': 0})
            case_docs.append(empty_doc)
    return case_docs

@st.cache_data(ttl=3600*24)
def load_for_show_law_docs():
    for_show_law_docs = []
    for for_show_law_name, url in FOR_SHOW_LAW_PDF_URLS.items():
        docs = load_docs(url, for_show_law_name)
        for doc in docs:
            doc.metadata['for_show_law_name'] = for_show_law_name
        for_show_law_docs.extend(docs)
    return for_show_law_docs

def create_retrievers(selected_laws):
    global law_retrievers, case_retrievers
    embeddings = OpenAIEmbeddings()
    
    for law_name in selected_laws:
        # 법령 리트리버 생성
        if law_name in law_vectordbs:
            law_retrievers[law_name] = law_vectordbs[law_name].as_retriever(search_kwargs={"k": 10})
        else:
            print(f"Warning: No vector database found for {law_name}")
            law_retrievers[law_name] = lambda x: []  # 빈 리스트를 반환하는 더미 함수
        
        # 사례 리트리버 생성
        if law_name in case_vectordbs:
            case_retrievers[law_name] = case_vectordbs[law_name].as_retriever(search_kwargs={"k":10})
        else:
            print(f"Warning: No case vector database found for {law_name}")
            case_retrievers[law_name] = lambda x: []  # 빈 리스트를 반환하는 더미 함수

def create_for_show_law_vectordbs(selected_laws):
    global selected_for_show_law_vectordbs
    for law_name in selected_laws:
        if law_name in for_show_law_vectordbs:
            selected_for_show_law_vectordbs[law_name] = for_show_law_vectordbs[law_name]
        else:
            print(f"Warning: No for_show vector database found for {law_name}")
            # 빈 벡터 데이터베이스 생성
            selected_for_show_law_vectordbs[law_name] = FAISS.from_texts(["빈 문서"], OpenAIEmbeddings())


@st.cache_data(ttl=3600*24)
def load_and_split_case_docs():
    case_docs = load_case_docs()
    if not case_docs:
        st.warning("사례 문서를 로드할 수 없습니다.")
        return []
    return case_split_docs(case_docs)

def delete_case(law_name, case_content):
    global case_vectordbs
    case_file_path = f"case_{law_name}.txt"
    if os.path.exists(case_file_path):
        with open(case_file_path, "r", encoding="utf-8") as file:
            content = file.read()
        
        # 삭제할 사례를 찾아 제거
        new_content = content.replace(f"\n\n[[새로운 사례]]\n{case_content}", "", 1)
        
        if new_content != content:
            with open(case_file_path, "w", encoding="utf-8") as file:
                file.write(new_content)
            
            # 벡터 데이터베이스 재생성
            embedding = OpenAIEmbeddings()
            documents = [Document(page_content=case, metadata={'law_name': law_name, 'source': '추가된 사례'}) 
                         for case in new_content.split("[[새로운 사례]]")[1:]]
            if documents:
                case_vectordbs[law_name] = FAISS.from_documents(documents, embedding)
            else:
                # 모든 사례가 삭제된 경우 빈 벡터 데이터베이스 생성
                case_vectordbs[law_name] = FAISS.from_texts(["빈 문서"], embedding=embedding)
            
            return True
    return False

def update_faiss_index(vectordb, new_docs, embedding):
    if not vectordb or not vectordb.docstore._dict:
        # 벡터 데이터베이스가 비어있는 경우, 새로 생성
        return FAISS.from_documents(new_docs, embedding)
    
    if not new_docs:
        return vectordb
    
    new_embeddings = embedding.embed_documents([doc.page_content for doc in new_docs])
    vectordb.add_embeddings(
        text_embeddings=list(zip([doc.page_content for doc in new_docs], new_embeddings)),
        metadatas=[doc.metadata  for doc in new_docs]
    )
    return vectordb

def update_case_vectordb(law_name):
    global case_vectordbs
    # 기존 사례 파일 로드
    case_file_path = f"case_{law_name}.txt"
    with open(case_file_path, "r", encoding="utf-8") as file:
        content = file.read()
    
    # 문서 생성
    documents = [Document(page_content=content, metadata={'law_name': law_name, 'source': '추가된 사례'})]
    
    # 문서 분할
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(documents)
    
    # 벡터 데이터베이스 업데이트
    embedding = OpenAIEmbeddings()
    case_vectordbs[law_name] = FAISS.from_documents(documents=splits, embedding=embedding)

def add_new_case(law_name, new_case):
    global case_vectordbs, case_docs
    case_file_path = f"case_{law_name}.txt"
    
    # 새로운 사례 추가
    with open(case_file_path, "a", encoding="utf-8") as file:
        file.write(f"\n\n[[새로운 사례]]\n{new_case}")
    
    # 새로운 문서 생성
    new_doc = Document(page_content=new_case, metadata={'law_name': law_name, 'source': '추가된 사례'})
    
    # 벡터 데이터베이스 업데이트
    embedding = OpenAIEmbeddings()
    
    # 전역 case_docs에 새 문서 추가
    case_docs.append(new_doc)
    
    if law_name not in case_vectordbs:
        # 해당 법령에 대한 벡터 데이터베이스가 없는 경우, 새로 생성
        case_vectordbs[law_name] = FAISS.from_documents([new_doc], embedding)
    else:
        # 기존 벡터 데이터베이스가 있는 경우, 업데이트
        case_vectordbs[law_name] = update_faiss_index(case_vectordbs[law_name], [new_doc], embedding)
    
    st.success(f"{law_name}에 새로운 사례가 추가되었습니다.")
    
    # update retrieved documents to ensure they are fresh
    if 'case_retrievers' in globals():
        global case_retrievers
        if law_name in case_retrievers:
            # 해당 법령의 리트리버만 업데이트
            selected_case_vectordbs = {law_name: case_vectordbs[law_name]}
            law_specific_retrievers = create_case_retrievers(selected_case_vectordbs)
            case_retrievers[law_name] = law_specific_retrievers[law_name]

def main():
    st.title("🦁AI 이행 컴플라이언스 봇🦁")

    global selected_laws
    selected_laws = []
    # CSS 스타일 정의
    st.markdown("""
    <style>
        .stButton > button {
            margin: 0px;
            padding: 0px 10px;
            height: 30px;
        }
        .feedback-buttons {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 5px;
        }
        .feedback-container {
            display: flex;
            justify-content: flex-end;
            align-items: center;
        }
        .feedback-message {
            margin-right: 10px;
        }
        .button-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logout-button {
            margin-left: 10px;
        }
        .logout-button .stButton > button {
            background-color: #f63366;
            color: white;
            border: none;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            line-height: 1.6;
            border-radius: 0.25rem;
            margin-top: -5px;  /* 버튼을 약간 위로 올립니다 */
        }
        .stButton > button {
            height: 2.2rem;
        }
        .button-container .element-container {
            margin-bottom: 0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    if check_password():
        if st.session_state.get("is_admin", False):
            st.header("관리자 섹션")
            
            # 로그아웃 버튼을 오른쪽 상단에 배치
            col1, col2 = st.columns([6, 1])
            with col2:
                if st.button("로그아웃", key="admin_logout"):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.experimental_rerun()
            
            with col1:
                st.subheader("데이터 관리")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["사용자 피드백", "사례 관리", "사용 현황 분석", "FCPA 적용대상 데이터 관리","기타 설정 (추가 예정)"])
            
            with tab1:
                if st.button("전반적인 사용후기 확인"):
                    if os.path.exists('feedback.csv'):
                        feedback_df = pd.read_csv('feedback.csv')
                        st.dataframe(feedback_df)
                    else:
                        st.info("아직 제출된 전반적인 사용후기가 없습니다.")
                
                if st.button("챗봇 답변 피드백 확인"):
                    if os.path.exists('chatbot_feedback.csv'):
                        chatbot_feedback_df = pd.read_csv('chatbot_feedback.csv')
                        st.dataframe(chatbot_feedback_df)
                    else:
                        st.info("아직 제출된 챗봇 답변 피드백이 없습니다.")
            
            with tab2:
                st.subheader("사례 관리")
                
                # 데이터 업데이트 버튼 추가
                if st.button("데이터 업데이트", key="update_data_button"):
                    with st.spinner("데이터를 업데이트하는 중..."):
                        # 캐시된 함수 재실행
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        case_splits = load_and_split_case_docs()
                        global case_vectordbs
                        case_vectordbs = load_case_vectordbs(case_splits)
                    st.success("데이터가 성공적으로 업데이트되었습니다.")
            
                st.subheader("새로운 유사 사례 추가")
                law_name = st.selectbox("법령 선택", list(CASE_PDF_URLS.keys()), key="add_law_select")
                new_case = st.text_area("새로운 유사 사례 내용", key="add_case_content")
                if st.button("사례 추가", key="add_case_button"):
                    add_new_case(law_name, new_case)
                    st.success(f"{law_name}에 새로운 사례가 추가되었습니다.")
                    st.info("변경사항이 적용되었습니다.")
            
                st.subheader("유사 사례 삭제")
                delete_law_name = st.selectbox("법령 선택", list(CASE_PDF_URLS.keys()), key="delete_law_select")
                case_file_path = f"case_{delete_law_name}.txt"
                if os.path.exists(case_file_path):
                    with open(case_file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                    cases = content.split("[[새로운 사례]]")[1:]  # 첫 번째 요소는 빈 문자열이므로 제외
                    if cases:
                        case_to_delete = st.selectbox("삭제할 사례 선택", cases, format_func=lambda x: x[:100] + "..." if len(x) > 100 else x)
                        if st.button("사례 삭제", key="delete_case_button"):
                            if delete_case(delete_law_name, case_to_delete.strip()):
                                st.success(f"{delete_law_name}에서 선택한 사례가 삭제되었습니다.")
                                st.info("변경사항이 적용되었습니다.")
                                # 캐시된 함수 재실행
                                st.cache_data.clear()
                                st.cache_resource.clear()
                                case_splits = load_and_split_case_docs()
                                case_vectordbs = load_case_vectordbs(case_splits)
                            else:
                                st.error("사례 삭제 중 오류가 발생했습니다.")
                    else:
                        st.info(f"{delete_law_name}에 추가된 사례가 없습니다.")
                else:
                    st.info(f"{delete_law_name}에 추가된 사례가 없습니다.")

            with tab3:
                st.subheader("사용자 질문 분석")
                if st.button("사용자별 질문 빈도 분석"):
                    if os.path.exists('user_questions.csv'):
                        df = pd.read_csv('user_questions.csv')
                        user_question_counts = df['User ID'].value_counts().reset_index()
                        user_question_counts.columns = ['User ID', '질문 횟수']

                        fig = px.bar(user_question_counts, x='User ID', y='질문 횟수',
                                     title='사용자별 질문 빈도',
                                     labels={'User ID': '사용자 ID', '질문 횟수': '질문 횟수'},
                                     color='질문 횟수',
                                     color_continuous_scale=px.colors.sequential.Viridis)
                        
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            xaxis_title='사용자 ID',
                            yaxis_title='질문 횟수',
                            height=600,
                            width=800
                        )
                
                        st.plotly_chart(fig)
                    else:
                        st.info("아직 기록된 질문이 없습니다.")
                
                if st.button("법령별 질문 빈도 분석"):
                    if os.path.exists('user_questions.csv'):
                        df = pd.read_csv('user_questions.csv')
                        law_counts = df['Selected Laws'].str.split(', ').explode().value_counts().reset_index()
                        law_counts.columns = ['법령', '질문 빈도']
                        
                        fig = px.bar(law_counts, x='질문 빈도', y='법령', orientation='h',
                                     title='법령별 질문 빈도')
                        fig.update_layout(height=600, width=800)
                        st.plotly_chart(fig)
                    else:
                        st.info("아직 기록된 질문이 없습니다.")
                        
                st.subheader("사용자 만족도 분석")    
                # 좋아요/싫어요 분석
                if st.button("좋아요/싫어요 분석"):
                    if os.path.exists('chatbot_feedback.csv'):
                        df = pd.read_csv('chatbot_feedback.csv')
                        feedback_counts = df['Feedback'].value_counts().reset_index()
                        feedback_counts.columns = ['피드백 유형', '빈도']
                        
                        fig = px.bar(feedback_counts, x='피드백 유형', y='빈도',
                                     title='좋아요/싫어요 분포')
                        fig.update_layout(height=500, width=700)
                        st.plotly_chart(fig)
                    else:
                        st.info("아직 제출된 피드백이 없습니다.")
            # FCPA 데이터 관리 탭 내용
            with tab4:
                st.subheader("FCPA 적용대상 데이터 관리")
                
                # 현재 FCPA 데이터 확인
                st.subheader("현재 FCPA 데이터 확인")
                if st.button("FCPA 데이터 조회", key="view_fcpa_data"):
                    if 'FCPA적용대상' in fcpa_retrievers:
                        # 랜덤 검색어로 샘플 데이터 조회
                        sample_query = "기관"
                        sample_docs = fcpa_retrievers['FCPA적용대상'](sample_query)
                        
                        if sample_docs:
                            for i, doc in enumerate(sample_docs[:10]):  # 상위 10개만 표시
                                with st.expander(f"데이터 항목 #{i+1}", expanded=i==0):
                                    st.markdown(f"```\n{doc.page_content}\n```")
                        else:
                            st.warning("FCPA 데이터가 없거나 검색 결과가 없습니다.")
                    else:
                        st.warning("FCPA 적용대상 데이터가 로드되지 않았습니다.")
                
                # 새로운 FCPA 데이터 추가
                st.subheader("새로운 FCPA 데이터 추가")
                
                # 기관 정보 입력 폼
                st.markdown("#### 기관 정보 입력")
                col1, col2 = st.columns(2)
                with col1:
                    org_name = st.text_input("기관명", key="fcpa_org_name")
                with col2:
                    org_type = st.selectbox("기관 유형", 
                                        ["국가기관", "지방자치단체", "교육청", "공공기관", "공직유관단체", 
                                            "교육기관", "언론사", "정당", "국제기구", "민간기업", "기타"],
                                        key="fcpa_org_type")
                
                col1, col2 = st.columns(2)
                with col1:
                    is_fcpa_target = st.radio("FCPA 적용 대상", ["적용", "미적용"], key="fcpa_target")
                with col2:
                    is_anti_corruption_target = st.radio("청탁금지법 적용 대상", ["적용", "미적용"], key="anti_corruption_target")
                
                # 추가 정보 (선택 사항)
                additional_info = st.text_area("추가 정보 (선택 사항)", key="fcpa_additional_info")
                
                # 데이터 추가 버튼
                if st.button("FCPA 데이터 추가", key="add_fcpa_data"):
                    if not org_name.strip():
                        st.error("기관명은 필수 입력 항목입니다.")
                    else:
                        try:
                            # 새로운
                            new_fcpa_data = f"{org_name} = FCPA {is_fcpa_target}대상, {org_name} = {org_type}, {org_name} = 청탁금지법 {'대상' if is_anti_corruption_target == '적용' else '대상아님'}"
                            
                            if additional_info.strip():
                                new_fcpa_data += f"\n# 추가 정보: {additional_info}"
                            
                            # 데이터 저장
                            fcpa_file_path = "fcpa_data.txt"
                            
                            # 파일이 존재하는 경우 내용 읽기
                            if os.path.exists(fcpa_file_path):
                                with open(fcpa_file_path, "r", encoding="utf-8") as file:
                                    existing_data = file.read().strip()
                                
                                # 새 데이터 추가 (구분자로 분리)
                                with open(fcpa_file_path, "w", encoding="utf-8") as file:
                                    file.write(f"{existing_data}////{new_fcpa_data}")
                            else:
                                # 파일이 없는 경우 새로 생성
                                with open(fcpa_file_path, "w", encoding="utf-8") as file:
                                    file.write(new_fcpa_data)
                            
                            # 성공 메시지 표시
                            st.success(f"FCPA 데이터가 성공적으로 추가되었습니다: {org_name}")
                            
                            # 캐시 초기화 버튼 추가
                            if st.button("캐시 초기화 및 데이터 리로드", key="reload_fcpa_data"):
                                st.cache_data.clear()
                                st.cache_resource.clear()
                                st.experimental_rerun()
                        except Exception as e:
                            st.error(f"데이터 추가 중 오류가 발생했습니다: {str(e)}")
                
                # FCPA 데이터 업로드 (CSV 또는 텍스트 파일)
                st.subheader("FCPA 데이터 파일 업로드")
                uploaded_file = st.file_uploader("CSV 또는 텍스트 파일 업로드", type=["csv", "txt"], key="fcpa_file_upload")
                
                if uploaded_file is not None:
                    try:
                        # 파일 읽기
                        content = uploaded_file.read().decode("utf-8")
                        
                        # CSV 또는 텍스트 파일 처리
                        if uploaded_file.name.endswith('.csv'):
                            # CSV 파일 처리
                            import io
                            import csv
                            
                            csv_data = []
                            csv_file = io.StringIO(content)
                            csv_reader = csv.reader(csv_file)
                            headers = next(csv_reader)  # 헤더 읽기
                            
                            # 필수 컬럼 확인
                            required_columns = ['기관명', '기관유형', 'FCPA적용여부', '청탁금지법적용여부']
                            missing_columns = [col for col in required_columns if col not in headers]
                            
                            if missing_columns:
                                st.error(f"CSV 파일에 필수 컬럼이 누락되었습니다: {', '.join(missing_columns)}")
                            else:
                                # 인덱스 찾기
                                col_indices = {col: headers.index(col) for col in required_columns}
                                
                                # 데이터 추출
                                formatted_data = []
                                for row in csv_reader:
                                    org_name = row[col_indices['기관명']].strip()
                                    org_type = row[col_indices['기관유형']].strip()
                                    fcpa_status = row[col_indices['FCPA적용여부']].strip()
                                    anti_corruption_status = row[col_indices['청탁금지법적용여부']].strip()
                                    
                                    if org_name:  # 기관명이 비어있지 않은 경우만 처리
                                        fcpa_target = "적용" if fcpa_status.lower() in ["적용", "yes", "y", "true", "1"] else "미적용"
                                        anti_corruption_target = "대상" if anti_corruption_status.lower() in ["적용", "yes", "y", "true", "1"] else "대상아님"
                                        
                                        entry = f"{org_name} = FCPA {fcpa_target}대상, {org_name} = {org_type}, {org_name} = 청탁금지법 {anti_corruption_target}"
                                        formatted_data.append(entry)
                                
                                # 데이터 저장
                                combined_data = "////".join(formatted_data)
                                
                                # 파일이 존재하는 경우 내용 읽기
                                fcpa_file_path = "fcpa_data.txt"
                                if os.path.exists(fcpa_file_path):
                                    with open(fcpa_file_path, "r", encoding="utf-8") as file:
                                        existing_data = file.read().strip()
                                    
                                    # 새 데이터 추가
                                    with open(fcpa_file_path, "w", encoding="utf-8") as file:
                                        file.write(f"{existing_data}////{combined_data}")
                                else:
                                    # 파일이 없는 경우 새로 생성
                                    with open(fcpa_file_path, "w", encoding="utf-8") as file:
                                        file.write(combined_data)
                                
                                st.success(f"CSV 파일에서 {len(formatted_data)}개의 기관 데이터가 성공적으로 추가되었습니다.")
                        else:
                            # 텍스트 파일 처리
                            # 이미 //// 구분자로 포맷된 데이터라고 가정
                            fcpa_file_path = "fcpa_data.txt"
                            if os.path.exists(fcpa_file_path):
                                with open(fcpa_file_path, "r", encoding="utf-8") as file:
                                    existing_data = file.read().strip()
                                
                                # 새 데이터 추가
                                with open(fcpa_file_path, "w", encoding="utf-8") as file:
                                    file.write(f"{existing_data}////{content}")
                            else:
                                # 파일이 없는 경우 새로 생성
                                with open(fcpa_file_path, "w", encoding="utf-8") as file:
                                    file.write(content)
                            
                            st.success("텍스트 파일의 FCPA 데이터가 성공적으로 추가되었습니다.")
                        
                        # 캐시 초기화 버튼 추가
                        if st.button("캐시 초기화 및 데이터 리로드", key="reload_fcpa_data_after_upload"):
                            st.cache_data.clear()
                            st.cache_resource.clear()
                            st.experimental_rerun()
                            
                    except Exception as e:
                        st.error(f"파일 처리 중 오류가 발생했습니다: {str(e)}")
                
                # 기존 FCPA 데이터 편집/삭제
                st.subheader("FCPA 데이터 편집/삭제")
                
                # 데이터 조회 및 편집
                if os.path.exists("fcpa_data.txt"):
                    with open("fcpa_data.txt", "r", encoding="utf-8") as file:
                        fcpa_data = file.read()
                    
                    # //// 구분자로 항목 분리
                    entries = fcpa_data.split("////")
                    
                    # 검색 기능
                    search_term = st.text_input("기관명으로 검색", key="fcpa_search")
                    
                    filtered_entries = entries
                    if search_term:
                        filtered_entries = [entry for entry in entries if search_term.lower() in entry.lower()]
                    
                    if filtered_entries:
                        selected_entry_index = st.selectbox(
                            "편집/삭제할 데이터 선택", 
                            range(len(filtered_entries)), 
                            format_func=lambda i: filtered_entries[i].split("\n")[0] if filtered_entries[i] else "빈 항목",
                            key="fcpa_entry_select"
                        )
                        
                        selected_entry = filtered_entries[selected_entry_index]
                        
                        # 선택된 항목 표시
                        st.markdown("#### 선택된 데이터")
                        st.markdown(f"```\n{selected_entry}\n```")
                        
                        # 편집 기능
                        edit_entry = st.text_area("데이터 편집", selected_entry, key="fcpa_edit_entry")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("변경사항 저장", key="save_fcpa_edit"):
                                try:
                                    # 원본 항목 인덱스 찾기
                                    orig_index = entries.index(selected_entry)
                                    
                                    # 항목 업데이트
                                    entries[orig_index] = edit_entry
                                    
                                    # 파일에 저장
                                    with open("fcpa_data.txt", "w", encoding="utf-8") as file:
                                        file.write("////".join(entries))
                                    
                                    st.success("데이터가 성공적으로 업데이트되었습니다.")
                                    
                                    # 캐시 초기화 버튼 추가
                                    if st.button("캐시 초기화 및 데이터 리로드", key="reload_fcpa_data_after_edit"):
                                        st.cache_data.clear()
                                        st.cache_resource.clear()
                                        st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"데이터 업데이트 중 오류가 발생했습니다: {str(e)}")
                        
                        with col2:
                            if st.button("데이터 삭제", key="delete_fcpa_entry"):
                                try:
                                    # 원본 항목 인덱스 찾기
                                    orig_index = entries.index(selected_entry)
                                    
                                    # 항목 삭제
                                    del entries[orig_index]
                                    
                                    # 파일에 저장
                                    with open("fcpa_data.txt", "w", encoding="utf-8") as file:
                                        file.write("////".join(entries))
                                    
                                    st.success("데이터가 성공적으로 삭제되었습니다.")
                                    
                                    # 캐시 초기화 버튼 추가
                                    if st.button("캐시 초기화 및 데이터 리로드", key="reload_fcpa_data_after_delete"):
                                        st.cache_data.clear()
                                        st.cache_resource.clear()
                                        st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"데이터 삭제 중 오류가 발생했습니다: {str(e)}")
                    else:
                        st.info("검색 조건에 맞는 데이터가 없습니다.")
                else:
                    st.info("FCPA 데이터 파일이 아직 생성되지 않았습니다.")



        else:    
            if "agreement_accepted" not in st.session_state:
                st.markdown("""
                <style>
                .agreement-box {
                    border: 2px solid #000000;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 20px;
                }
                .agreement-title {
                    font-size: 20px;
                    font-weight: bold;
                    color: #000000;
                    margin-bottom: 10px;
                }
                .agreement-content {
                    font-size: 16px;
                    color: #333;
                    margin-bottom: 15px;
                }
                .agreement-image {
                    display: flex;
                    justify-content: center;
                    margin-top: 20px;
                }
                .agreement-image img {
                    max-width: 300px;
                    height: auto;
                }
                </style>
                
                <div class="agreement-box">
                    <div class="agreement-title">AI 컴플라이언스봇 서비스 이용 동의(필수)</div>
                    <div class="agreement-content">
                        아래 내용에 동의하신 후 서비스를 이용하실 수 있습니다:
                        <ul>
                            <li>AI가 생성하는 답변은 참고용으로 작성된 내용으로 법률적으로 정확하지 않을 수 있습니다. 법률 자문이 필요한 경우 전문 변호사를 통해 상담 부탁드립니다.</li>
                            <li>'AI 이행 컴플라이언스 봇'은 답변 내용에 대하여 어떠한 책임도 지지 않습니다.</li>
                            <li>서비스 이용 과정에서 수집된 데이터는 AI 모델의 성능 개선 및 서비스 품질 향상 목적과 컴플라이언스 업무 개선에 활용될 수 있습니다.</li>
                        </ul>
                    </div>
                    <div class="agreement-image">
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 이미지 표시
                st.image(image, caption="질문 예시", use_column_width=True, output_format="PNG")

                
                if st.checkbox("위의 내용을 확인하고 동의합니다."):
                    st.session_state.agreement_accepted = True
                    st.success("동의가 완료되었습니다. 서비스를 이용하실 수 있습니다.")
                    st.experimental_rerun()

            elif st.session_state.agreement_accepted:
                os.environ["OPENAI_API_KEY"] = ""
                os.environ["UPSTAGE_API_KEY"] = "up_pgmFOF5OgvvdPeGtMYqn02nXL73HF"
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
                os.environ["LANGCHAIN_PROJECT"] = "Complionss"
                os.environ["LANGCHAIN_API_KEY"] = "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"
    
                upstage_ground_checker = UpstageGroundednessCheck()
        
                # CSS를 사용하여 스타일 정의
                st.markdown("""
                <style>
                    .notice-box {
                        border: 2px solid #4CAF50;
                        border-radius: 10px;
                        padding: 15px;
                        margin-top: 20px;
                        margin-bottom: 30px;
                        background-color: #f1f8e9;
                    }
                    .notice-title {
                        color: #4CAF50;
                        font-size: 18px;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    .notice-content {
                        font-size: 14px;
                        color: #333;
                    }
                    .notice-footer {
                        font-size: 12px;
                        color: #666;
                        margin-top: 10px;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # 공지사항 추가
                st.markdown("""
                <div class="notice-box">
                    <div class="notice-title">📢 공지사항</div>
                    <div class="notice-content">
                        안녕하세요! AI 이행 컴플라이언스 봇입니다🦁<br><br>
                        프로젝트 이행 관련 법률*과 회사 내규(부패방지 행동강령 및 윤리경영원칙 실천지침)에 대해 빠른 답변과 관련 법률 항목과 연관 사례들을 제공합니다. <br>
                        답변은 참고용이며, 법적 효력이 없음을 고지드립니다. <br>
                    </div>
                    <div class="notice-footer">
                        * 프로젝트 이행 관련 법률<br>
                        : 청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 정보통신공사업법, 국가계약법, 소프트웨어진흥법<br><br>
                        ※ 정보통신공사업법, 국가계약법, 소프트웨어진흥법, 회사 내규는 현재 사례 데이터 준비중입니다. 이용에 참고부탁드립니다.
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
                session_id = st.session_state.get("current_session_id", str(uuid.uuid4()))
                st.session_state["current_session_id"] = session_id
    
                # 사용자별 상태 초기화 및 동기화
                if "user_states" not in st.session_state:
                    st.session_state["user_states"] = {}
    
                if session_id not in st.session_state["user_states"]:
                    st.session_state["user_states"][session_id] = {
                        "messages": [],
                        "store": {},
                        "law_references": [],
                        "similar_cases": [],
                        "relevance_results": [],
                        "selected_messages": []
                    }
    
                user_state = st.session_state["user_states"][session_id]
    
                # 전체 세션 상태와 사용자별 상태 동기화
                if "messages" not in st.session_state:
                    st.session_state.messages = user_state["messages"]
                else:
                    user_state["messages"] = st.session_state.messages
    
                # 사이드바 설정
                with st.sidebar:
                    logged_in_user = st.session_state.get("logged_in_user", "Unknown")
                    st.header(f"[접속자] {logged_in_user}"+"님, 환영합니다!")
                    
                    clear_btn = st.button("새로운 대화 주제", key="clear_button")

                    model_name = st.selectbox("언어 모델 선택", ["o3-mini"], key="model_select")

                    # 답변 모드 선택 부분 수정
                    if "answer_mode" not in st.session_state:
                        st.session_state["answer_mode"] = "빠른 답변" 
                    # reasoning_effort 선택 추가
                    if "reasoning_effort" not in st.session_state:
                        st.session_state["reasoning_effort"] = "medium"

                    st.session_state["answer_mode"] = st.selectbox(
                        "답변 모드 선택", 
                        ["빠른 답변", "정확한 답변"],  
                        help="빠른 답변: 재질문 없이 즉시 답변을 제공합니다. | 정확한 답변: 질문에 대한 재해석을 통해 더 정확한 답변을 제공합니다.",
                        key="answer_mode_select",
                        index=0  
                    )

                    st.session_state["reasoning_effort"] = st.selectbox(
                        "추론 수준 선택",
                        ["low", "medium", "high"],
                        help="low: 빠른 추론 | medium: 보통 수준의 추론 | high: 심도있는 추론",
                        key="reasoning_effort_select",
                        index=1  # medium을 기본값으로 설정
                    )

                    # 피드백 섹션
                    st.header("피드백")
                    feedback = st.text_area("전반적인 사용후기를 입력해주세요:", key="feedback_text")
                    
                    # 피드백 제출 버튼과 로그아웃 버튼을 같은 줄에 배치
                    st.markdown('<div class="button-container">', unsafe_allow_html=True)
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        submit_btn = st.button("피드백 제출", key="feedback_submit")
                    with col2:
                        logout_btn = st.button("로그아웃", key="logout_button")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.write("🐧 저작자: @AI컴플라이언스봇 TF")

                
                # 피드백 제출 로직
                if submit_btn:
                    if feedback:
                        kst = timezone(timedelta(hours=9))
                        timestamp = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
                        feedback_data = pd.DataFrame({
                            'User': [st.session_state.get("logged_in_user", "Unknown")],
                            'Feedback': [feedback], 
                            'Timestamp': [timestamp]
                        })
                        feedback_data.to_csv('feedback.csv', mode='a', header=not os.path.exists('feedback.csv'), index=False)
                        st.sidebar.success("피드백이 제출되었습니다. 감사합니다!")
                    else:   
                        st.sidebar.warning("피드백을 입력해주세요.")
    
                if clear_btn:
                    # 새로운 세션 ID 생성
                    st.session_state.chat_history = []
                    # 기타 관련 상태 초기화
                    if 'messages' in st.session_state:
                        del st.session_state.messages
                    if 'feedback_submitted' in st.session_state:
                        del st.session_state.feedback_submitted
                    if 'store' in st.session_state:
                        del st.session_state.store
                    if 'relevance_results' in st.session_state:
                        del st.session_state.relevance_results
                    if 'law_references' in st.session_state:
                        del st.session_state.law_references
                    if 'similar_cases' in st.session_state:
                        del st.session_state.similar_cases
                    if 'selected_messages' in st.session_state:
                        del st.session_state.selected_messages
                    
                    # memory 초기화 (만약 전역 변수로 사용 중이라면)
                    global memory
                    if 'memory' in globals():
                        memory = ConversationBufferMemory(return_messages=True)
                    
                    st.success("새로운 대화 주제가 시작되었습니다.")
                    st.experimental_rerun()
                
                if logout_btn:
                    # 로그아웃 처리
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.experimental_rerun()
    
                # 임베딩 모델 설정
                embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
                embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                if "feedback_data" not in st.session_state:
                    st.session_state.feedback_data = []
                if "store" not in st.session_state:
                    st.session_state["store"] = {}
                if "relevance_results" not in st.session_state:
                    st.session_state["relevance_results"] = []
                if "law_references" not in st.session_state:
                    st.session_state["law_references"] = []
                if "similar_cases" not in st.session_state:
                    st.session_state["similar_cases"] = []
                if "selected_messages" not in st.session_state:
                    st.session_state["selected_messages"] = []
                if "user_states" not in st.session_state:
                    st.session_state["user_states"] = {}
                if "feedback_states" not in st.session_state:
                    st.session_state.feedback_states = {}
                if "question_type_select" not in st.session_state:
                    st.session_state.question_type_select = "단순 질의응답"
    
                if session_id not in st.session_state["user_states"]:
                    st.session_state["user_states"][session_id] = {
                        "messages": [],
                        "store": {},
                        "relevance_results": [],
                        "law_references": [],
                        "similar_cases": [],
                        "selected_messages": []
                    }
    
                # 사용자별 상태에 접근
                user_state = st.session_state["user_states"][session_id]

    
                # 문서 로딩
                global fcpa_docs

                fcpa_docs = load_fcpa_docs()
                fcpa_splits = fcpa_split_docs(fcpa_docs)
                fcpa_vectordbs = load_fcpa_vectordbs(fcpa_splits)
                fcpa_retrievers = create_fcpa_retrievers(fcpa_vectordbs)  

                
                global law_docs
                law_docs = load_law_docs()
                law_splits = law_split_docs(law_docs)
                global law_vectordbs
                law_vectordbs = load_law_vectordbs(law_splits)
                global case_docs
                case_docs = load_case_docs()
                case_splits = case_split_docs(case_docs)
                case_vectordbs = load_case_vectordbs(case_splits)
    
                case_splits = load_and_split_case_docs()
                if case_splits:
                    case_vectordbs = load_case_vectordbs(case_splits)

                for_show_law_docs = load_for_show_law_docs()
                for_show_law_splits = show_law_split_docs(for_show_law_docs)

                global for_show_law_vectordbs
                for_show_law_vectordbs = load_for_show_law_vectordbs(for_show_law_splits)
    
                def highlight_text(text, query_embedding, embeddings, threshold):
                    sentences = text.split('. ')
                    highlighted_sentences = []
                    
                    # 문장 임베딩을 한 번에 계산
                    sentence_embeddings = embeddings.embed_documents(sentences)
                    
                    similarities = cosine_similarity([query_embedding], sentence_embeddings)[0]
                    
                    for sentence, similarity in zip(sentences, similarities):
                        if similarity >= threshold:
                            highlighted_sentences.append(f'<span style="background-color: yellow;">{sentence}</span>')
                        else:
                            highlighted_sentences.append(sentence)
                    
                    return '. '.join(highlighted_sentences)
                
    
                # GraphState 정의
                class GraphState(TypedDict):
                    question: str
                    questions_and_answers: List[Tuple[str, str]]
                    law_context: str
                    case_context: str
                    response: str
                    relevance: str
                    attempts: int
                    law_name: str
                    law_references: List[Dict]
                    similar_cases: List[Dict]
                    question_type: str
                    answer_mode: str
    
                memory = ConversationBufferMemory(return_messages=True)
    
                # 법령별 특화 프롬프트 정의
                LAW_SPECIFIC_PROMPTS = {
                    "청탁금지법": """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 청탁금지법 전문 사내 변호사입니다. 청탁금지법은 주로 공직자에게 적용되는 법임을 명심하세요. KT 직원들이 공직자, 공공기관, 또는 공직유관단체와 업무를 수행할 때만 이 법을 고려해야 합니다. 이 법의 주요 목적인 공직자의 공정한 직무수행과 공공기관의 신뢰성 제고에 중점을 두고, KT 직원들이 공직자와의 관계에서 주의해야 할 점을 답변해주세요.
                
                    답변을 하기 전에 다음 사항들을 확인하세요:
                    1. 질문에 언급된 모든 당사자들의 관계를 명확히 파악했나요? (예: KT 직원-공직자, KT 직원-공공기관 등)
                    2. 각 당사자의 역할과 지위가 명확한가요? (예: 공직자인지, 공직유관단체 임직원인지 등)
                    3. 상황의 구체적인 맥락이 충분히 제공되었나요? (예: 업무 관계인지, 사적 관계인지 등)
                    4. 행위의 목적과 의도가 명확한가요?
                    5. 금품이나 향응 제공의 경우, 그 가액과 빈도가 명시되어 있나요?
                    6. 과거 판례나 유사 사례에서 이와 관련된 결정이 있었나요?
                    7. 언급된 기관이 청탁금지법 적용 대상인지, FCPA 적용 대상인지를 명확히 파악했나요?
                
                    이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
                    "질문에서 언급된 A씨의 정확한 직위나 소속을 알려주실 수 있나요? A씨가 공직자인지 아닌지에 따라 답변이 달라질 수 있습니다."
                    "이 상황이 업무적 관계에서 발생한 것인지, 아니면 사적인 관계에서 발생한 것인지 명확히 해주시겠어요?"
                    "제공된 식사의 정확한 가액을 알 수 있을까요? 가액에 따라 법적 판단이 달라질 수 있습니다."
                    "이와 유사한 과거 판례나 사례가 있다면 함께 고려하여 답변 드리겠습니다."
                
                    이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 과거 판례나 유사 사례에서 유사한 상황에 대한 결정이 있었다면, 그 내용을 중요하게 고려하여 답변해주세요.""",
                
                    "중대재해처벌법": """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 중대재해처벌법 전문 사내 변호사입니다. 사업주와 경영책임자의 안전 및 보건 확보의무와 위반 시 처벌에 초점을 맞춰 답변해주세요.
                
                    답변을 하기 전에 다음 사항들을 확인하세요:
                    1. 사고 또는 위험 상황의 구체적인 내용이 명확한가요?
                    2. 관련된 근로자의 수와 사업장의 규모가 명시되어 있나요?
                    3. 사고 발생 시점 또는 위험 상황 인지 시점이 명확한가요?
                    4. 사업주나 경영책임자의 안전 조치 이행 여부가 언급되어 있나요?
                    5. 과거의 유사 사고 이력이나 안전 교육 실시 여부가 언급되어 있나요?
                    6. 과거 판례나 유사 사례에서 이와 관련된 결정이 있었나요?
                
                    이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
                    "해당 사업장의 근로자 수와 전체 사업장 규모를 알려주실 수 있나요? 이에 따라 적용되는 법적 기준이 달라질 수 있습니다."
                    "사고 발생 전 안전 교육이나 점검이 실시되었는지, 그리고 그 내용은 무엇인지 알 수 있을까요?"
                    "경영책임자가 이 위험 상황을 인지하고 있었는지, 그리고 어떤 조치를 취했는지 추가 정보가 있나요?"
                    "이와 유사한 과거 판례나 사례가 있다면 함께 고려하여 답변 드리겠습니다."
                
                    이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 과거 판례나 유사 사례에서 유사한 상황에 대한 결정이 있었다면, 그 내용을 중요하게 고려하여 답변해주세요.""",
                
                    "산업안전보건법": """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 산업안전보건법 전문 사내 변호사입니다. 근로자의 안전과 보건을 유지·증진하기 위한 사업주의 의무와 근로자의 권리에 중점을 두고 답변해주세요.
                
                    답변을 하기 전에 다음 사항들을 확인하세요:
                    1. 해당 작업 또는 상황의 구체적인 내용이 명확한가요?
                    2. 관련된 근로자의 고용 형태(정규직, 계약직, 일용직 등)가 명시되어 있나요?
                    3. 작업 환경이나 사용 중인 기계, 설비에 대한 정보가 충분한가요?
                    4. 사업주가 취한 안전 조치나 보건 관리 활동이 언급되어 있나요?
                    5. 근로자의 안전 교육 이수 여부나 개인보호구 착용 상태가 명확한가요?
                    6. 과거 판례나 유사 사례에서 이와 관련된 결정이 있었나요?
                
                    이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
                    "해당 작업에 참여한 근로자들의 고용 형태를 알려주실 수 있나요? 고용 형태에 따라 적용되는 법적 보호 범위가 달라질 수 있습니다."
                    "작업 현장에서 사용 중인 기계나 설비의 종류, 그리고 그에 대한 안전 점검 기록이 있나요?"
                    "근로자들이 최근에 받은 안전 교육의 내용과 시기를 알 수 있을까요?"
                    "이와 유사한 과거 판례나 사례가 있다면 함께 고려하여 답변 드리겠습니다."
                
                    이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 과거 판례나 유사 사례에서 유사한 상황에 대한 결정이 있었다면, 그 내용을 중요하게 고려하여 답변해주세요.""",
                
                    "하도급법": """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 하도급법 전문 사내 변호사입니다. 원사업자와 수급사업자 간의 공정한 거래관계 확립에 초점을 맞춰 답변해주세요.
                
                    답변을 하기 전에 다음 사항들을 확인하세요:
                    1. 원사업자와 수급사업자의 관계가 명확히 정의되어 있나요?
                    2. 하도급 계약의 구체적인 내용(금액, 기간, 작업 범위 등)이 명시되어 있나요?
                    3. 대금 지급 조건이나 방식이 언급되어 있나요?
                    4. 기술자료 요구나 제공과 관련된 사항이 있나요?
                    5. 원재료 가격 변동이나 작업 내용 변경 등의 상황이 발생했나요?
                    6. 과거 판례나 유사 사례에서 이와 관련된 결정이 있었나요?
                
                    이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
                    "원사업자와 수급사업자의 사업 규모나 거래 기간 등에 대한 추가 정보를 제공해 주실 수 있나요?"
                    "하도급 계약서의 주요 조항, 특히 대금 지급 조건에 대해 자세히 알려주실 수 있나요?"
                    "기술자료 요구가 있었다면, 그 요구의 목적과 범위, 그리고 요구 시점에 대해 추가 설명 부탁드립니다."
                    "이와 유사한 과거 판례나 사례가 있다면 함께 고려하여 답변 드리겠습니다."
                
                    이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 과거 판례나 유사 사례에서 유사한 상황에 대한 결정이 있었다면, 그 내용을 중요하게 고려하여 답변해주세요.""",
                
                    "상생협력법": """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 상생협력법 전문 사내 변호사입니다. 대기업과 중소기업 간의 상생협력 관계 구축과 동반성장에 중점을 두고 답변해주세요.
                
                    답변을 하기 전에 다음 사항들을 확인하세요:
                    1. 대기업과 중소기업 간의 구체적인 협력 관계가 명시되어 있나요?
                    2. 기술 협력이나 자금 지원 등의 상생협력 활동 내용이 언급되어 있나요?
                    3. 공정거래 및 동반성장 협약 체결 여부가 명확한가요?
                    4. 수탁기업의 이익 보호를 위한 조치들이 언급되어 있나요?
                    5. 기술 탈취나 부당한 경영 간섭 등의 문제가 제기되었나요?
                    6. 과거 판례나 유사 사례에서 이와 관련된 결정이 있었나요?
                
                    이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
                    "대기업과 중소기업 간의 구체적인 협력 사업 내용과 기간에 대해 추가 정보를 제공해 주실 수 있나요?"
                    "기술 협력이 있었다면, 그 협력의 범위와 조건, 그리고 성과 공유 방식에 대해 자세히 알려주실 수 있나요?"
                    "동반성장 협약이 체결되었다면, 그 주요 내용과 이행 상황에 대해 설명 부탁드립니다."
                    "이와 유사한 과거 판례나 사례가 있다면 함께 고려하여 답변 드리겠습니다."
                
                    이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 과거 판례나 유사 사례에서 유사한 상황에 대한 결정이 있었다면, 그 내용을 중요하게 고려하여 답변해주세요.""",
                
                    "공정거래법": """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 공정거래법 전문 사내 변호사입니다. 시장에서의 자유롭고 공정한 경쟁을 촉진하고, 독과점 및 불공정 거래 행위의 규제를 통해 소비자의 이익을 보호하는 데 중점을 두고 답변해주세요.
                
                    답변을 하기 전에 다음 사항들을 확인하세요:
                    1. 관련 시장의 구조와 경쟁 상황이 명확히 설명되어 있나요?
                    2. 문제가 되는 거래 행위의 구체적인 내용과 기간이 명시되어 있나요?
                    3. 해당 행위가 시장에 미치는 영향이 언급되어 있나요?
                    4. 관련 기업들의 시장 점유율이나 경제력 집중 정도가 명확한가요?
                    5. 소비자 이익 침해 여부나 정도가 언급되어 있나요?
                    6. 과거 판례나 유사 사례에서 이와 관련된 결정이 있었나요?
                    7. 거래 상대방이 공공기관이나 공직유관단체인 경우, 추가적인 법적 고려사항(청탁금지법 등)이 있을 수 있습니다.
                
                    이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
                    "해당 시장의 주요 참여자들과 그들의 시장 점유율에 대한 추가 정보를 제공해 주실 수 있나요?"
                    "문제가 되는 거래 행위의 구체적인 내용, 기간, 그리고 그로 인한 경제적 이익이나 손실에 대해 자세히 알려주실 수 있나요?"
                    "이 행위로 인해 소비자들이 받은 구체적인 영향이나 피해 사례가 있다면 설명 부탁드립니다."
                    "이와 유사한 과거 판례나 사례가 있다면 함께 고려하여 답변 드리겠습니다."
                
                    이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 과거 판례나 유사 사례에서 유사한 상황에 대한 결정이 있었다면, 그 내용을 중요하게 고려하여 답변해주세요.""",
                
                    "정보통신공사업법": """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 정보통신공사업법 전문 사내 변호사입니다. 정보통신공사의 적정한 시공과 공사 품질의 확보, 기술자의 자격 요건 및 준수사항에 중점을 두고 답변해주세요.
                
                    답변을 하기 전에 다음 사항들을 확인하세요:
                    1. 해당 정보통신공사의 규모와 성격이 명확히 설명되어 있나요?
                    2. 공사 수행 기업의 등록 상태와 기술자 보유 현황이 명시되어 있나요?
                    3. 공사 계약의 구체적인 내용(금액, 기간, 작업 범위 등)이 언급되어 있나요?
                    4. 사용된 기자재의 규격과 품질 인증 여부가 명확한가요?
                    5. 공사 감리나 검사 관련 사항이 언급되어 있나요?
                    6. 과거 판례나 유사 사례에서 이와 관련된 결정이 있었나요?
                
                    이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
                    "해당 정보통신공사의 구체적인 내용과 규모, 그리고 공사 기간에 대해 추가 정보를 제공해 주실 수 있나요?"
                    "공사를 수행하는 기업의 정보통신공사업 등록 현황과 보유 기술자의 자격 정보를 자세히 알려주실 수 있나요?"
                    "사용된 주요 기자재의 종류와 해당 기자재의 품질 인증 여부에 대해 설명 부탁드립니다."
                    "이와 유사한 과거 판례나 사례가 있다면 함께 고려하여 답변 드리겠습니다."
                
                    이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 과거 판례나 유사 사례에서 유사한 상황에 대한 결정이 있었다면, 그 내용을 중요하게 고려하여 답변해주세요.""",
                
                    "국가계약법": """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 국가계약법 전문 사내 변호사입니다. 국가와 공공기관의 계약 체결 시 공정성과 투명성을 보장하고, 계약 절차와 이행에서의 법적 준수 사항에 중점을 두고 답변해주세요.
                
                    답변을 하기 전에 다음 사항들을 확인하세요:
                    1. 계약의 종류(물품, 용역, 공사 등)와 규모가 명확히 설명되어 있나요?
                    2. 계약 방식(일반경쟁, 제한경쟁, 수의계약 등)이 명시되어 있나요?
                    3. 입찰 과정의 구체적인 내용(참가자격, 평가기준 등)이 언급되어 있나요?
                    4. 계약 금액과 이행 기간이 명확한가요?
                    5. 계약 이행 보증이나 지체상금 관련 사항이 언급되어 있나요?
                    6. 과거 판례나 유사 사례에서 이와 관련된 결정이 있었나요?
                
                    이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
                    "해당 계약의 구체적인 내용과 예상 금액, 그리고 계약 기간에 대해 추가 정보를 제공해 주실 수 있나요?"
                    "입찰 방식과 참가자격 제한 사항, 그리고 평가 기준에 대해 자세히 알려주실 수 있나요?"
                    "계약 이행 보증이나 지체상금 등 계약의 주요 조건에 대해 설명 부탁드립니다."
                    "이와 유사한 과거 판례나 사례가 있다면 함께 고려하여 답변 드리겠습니다."
                
                    이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 과거 판례나 유사 사례에서 유사한 상황에 대한 결정이 있었다면, 그 내용을 중요하게 고려하여 답변해주세요.""",
                
                    "소프트웨어진흥법": """당신은 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 대기업 KT의 소프트웨어진흥법 전문 사내 변호사입니다. 소프트웨어 산업의 발전과 공정한 시장 환경 조성, 그리고 소프트웨어의 품질과 안전성 확보에 중점을 두고 답변해주세요.
                
                    답변을 하기 전에 다음 사항들을 확인하세요:
                    1. 관련된 소프트웨어의 종류와 용도가 명확히 설명되어 있나요?
                    2. 소프트웨어 개발이나 유지보수 계약의 구체적인 내용이 명시되어 있나요?
                    3. 소프트웨어 기술자의 자격이나 경력 관련 사항이 언급되어 있나요?
                    4. 소프트웨어 품질 인증이나 보안 관련 요구사항이 명확한가요?
                    5. 소프트웨어 저작권이나 기술 이전 관련 사항이 언급되어 있나요?
                    6. 과거 판례나 유사 사례에서 이와 관련된 결정이 있었나요?
                
                    이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
                    "해당 소프트웨어의 구체적인 기능과 용도, 그리고 개발 규모에 대해 추가 정보를 제공해 주실 수 있나요?"
                    "소프트웨어 개발이나 유지보수 계약의 주요 조건, 특히 대금 지급 조건과 지식재산권 귀속에 대해 자세히 알려주실 수 있나요?"
                    "소프트웨어 품질 보증이나 보안 요구사항에 대해 구체적으로 설명 부탁드립니다."
                    "이와 유사한 과거 판례나 사례가 있다면 함께 고려하여 답변 드리겠습니다."
                
                    이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 과거 판례나 유사 사례에서 유사한 상황에 대한 결정이 있었다면, 그 내용을 중요하게 고려하여 답변해주세요.""",
                
                    "회사 내규(부패방지 행동강령 및 윤리경영원칙 실천지침)": """당신은 꼼꼼하게 회사의 내규와 윤리 기준을 준수하며, 쉽게 단정하지 않는 대기업 KT의 윤리경영 전문 사내 변호사입니다. 회사 내 모든 임직원이 부패 및 비윤리적 행위를 방지하고, 공정하고 투명한 업무 수행을 통해 회사의 신뢰성과 명성을 유지할 수 있도록 실천지침을 준수하는 데 중점을 두고 답변해주세요.
                
                    답변을 하기 전에 다음 사항들을 확인하세요:
                    1. 질문과 관련된 구체적인 상황이나 행위가 명확히 설명되어 있나요?
                    2. 관련된 임직원의 직급이나 부서가 명시되어 있나요?
                    3. 해당 행위가 업무 관련성이 있는지 여부가 명확한가요?
                    4. 금전적 이익이나 향응 제공이 있었다면, 그 가액과 빈도가 언급되어 있나요?
                    5. 이해관계자와의 관계나 거래 내용이 구체적으로 설명되어 있나요?
                    6. 과거 유사한 사례나 내부 결정 사항이 있었나요?
                
                    이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
                    "해당 상황에 관련된 임직원의 직급과 담당 업무에 대해 추가 정보를 제공해 주실 수 있나요?"
                    "이 행위가 업무와 어떤 관련성이 있는지, 또는 순수한 사적 관계에서 발생한 것인지 명확히 해주실 수 있나요?"
                    "제공된 금품이나 향응이 있다면, 그 구체적인 내용과 가액, 그리고 제공 빈도에 대해 자세히 알려주실 수 있나요?"
                    "이와 유사한 과거 사례나 내부 결정 사항이 있다면 함께 고려하여 답변 드리겠습니다."
                
                    이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 과거 유사 사례나 내부 결정 사항이 있었다면, 그 내용을 중요하게 고려하여 답변해주세요.""",
                }
                    
                def select_persona_prompt(question_type):
                    base_prompt = """당신은 대기업이자 사기업인 KT의 법률 전문 사내 변호사입니다. 질문자는 기본적으로 KT 직원으로, 공직자가 아닌 민간 기업의 직원입니다. KT는 정부 기관이 아니며, 직원들은 공무원이 아닙니다. """
                    
                    if question_type == "법 저촉 여부(다양한 관점 분석)":
                        return base_prompt + """주어진 상황의 법적 허용 가능성을 다각도로 분석해야 합니다. 다음 지침을 따라 답변해 주세요:
                
                        1. 답변 근거: <제공된 문서>에 기반하여 답변하세요. 
                        2. 상황 파악: 주어진 상황을 정확히 이해하고 분석하세요.
                        3. 관련 법령 검토: 상황과 관련된 법령들을 포괄적으로 검토하세요.
                        4. 다양한 관점 제시: 상황에 대한 여러 법적 해석과 관점을 제시하세요.
                        5. 위험 요소 식별: 잠재적인 법적 위험이나 위반 소지를 명확히 지적하세요.
                        6. 대안 제시: 법을 준수하면서 업무를 수행할 수 있는 방안을 제안하세요.
                        7. 정보 부족 시 추가 정보 요청: 답변을 정확하게 주기 어려운 경우, 필요한 추가 정보를 요청하는 질문을 하세요.
                        8. 금액 비교: 금액이 언급된 경우, 법령에서 정한 기준 금액과 명확히 비교하여 설명하세요. 예를 들어, "A원은 법령에서 정한 B원보다 크므로/작으므로..." 와 같이 명확히 비교하세요.

                        답변 시 사기업 직원으로서 준수해야 할 법적 의무와 제한사항을 고려하세요. 법적 위험을 사전에 안내하고, 직원들이 법을 준수하면서 업무를 수행할 수 있도록 조언하세요. 당신의 응답은 <제공된 문서>에 기반해야 합니다."""  
                            
                    elif question_type == "단순 질의응답":
                        return base_prompt + """주어진 법률 관련 질문에 대해 간단명료하게 답변해야 합니다. 다음 지침을 따라 답변해 주세요:
                
                        1. 답변 근거: <제공된 문서>에 기반하여 답변하세요.
                        2. 질문 이해: 주어진 질문의 핵심을 정확히 파악하세요.
                        3. 관련 법령 확인: 질문과 관련된 법령을 명시하세요.
                        4. 명확한 답변: 질문에 대해 명확하고 간결하게 답변하세요.
                        5. 추가 설명: 필요한 경우 간단한 부연 설명을 제공하세요.
                        6. 한계 명시: 답변의 한계나 예외 사항이 있다면 언급하세요.
                        7. 정보 부족 시 추가 정보 요청: 답변을 정확하게 주기 어려운 경우, 필요한 추가 정보를 요청하는 질문을 하세요.
                
                        답변은 법률 전문가가 아닌 KT 직원도 이해할 수 있도록 쉽게 설명해 주세요. 사기업 직원의 관점에서 적용되는 법률을 설명하세요. 당신의 응답은 <제공된 문서>에 기반해야 합니다.
                        """
                
                    elif question_type == "금액 계산":
                        return base_prompt + """주어진 상황에 대해 법적으로 정해진 금액을 계산해야 합니다. 다음 지침을 따라 답변해 주세요:
                
                        1. 답변 근거: <제공된 문서>에 기반하여 답변하세요.
                        2. 상황 분석: 주어진 상황을 정확히 파악하세요.
                        3. 관련 법령 확인: 금액 계산과 관련된 법령을 명시하세요.
                        4. 계산 과정 설명: 금액 계산 과정을 단계별로 명확히 설명하세요.
                        5. 결과 제시: 최종 계산된 금액을 명확히 제시하세요.
                        6. 주의사항 언급: 계산 결과에 영향을 줄 수 있는 요소나 예외 사항을 설명하세요.
                        7. 정보 부족 시 추가 정보 요청: 계산에 필요한 정보가 부족한 경우, 필요한 추가 정보를 요청하는 질문을 하세요.
                
                        답변 시 사용된 공식이나 기준을 명확히 제시하세요. 사기업에 적용되는 법률과 규정을 기준으로 설명하세요. 당신의 응답은 <제공된 문서>에 기반해야 합니다."""
                
                    else:  # "그 외"
                        return base_prompt + """정보를 요청하는 질문을 하세요.
                주어진 질문에 대해 법률적 관점에서 최선의 답변을 제공해야 합니다. 다음 지침을 따라 답변해 주세요:
                
                        1. 답변 근거: <제공된 문서>에 기반하여 답변하세요.
                        2. 질문 분석: 주어진 질문의 본질을 파악하세요.
                        3. 관련 법령 검토: 질문과 관련될 수 있는 법령을 검토하세요.
                        4. 종합적 답변: 법률적 관점에서 종합적인 답변을 제공하세요.
                        5. 한계 명시: 답변의 한계나 추가 검토가 필요한 사항을 언급하세요.
                        6. 조언 제공: 필요하다면 법률적 조언이나 주의사항을 제시하세요.
                        7. 정보 부족 시 추가 정보 요청: 답변을 정확하게 주기 어려운 경우, 필요한 추가 
                        당신의 응답은 <제공된 문서>에 기반해야 합니다. 답변 시 확실하지 않은 부분은 명확히 언급하고, 필요하다면 추가적인 법률 자문을 권고하세요. 항상 사기업 직원의 관점에서 적용되는 법률을 고려하여 답변하세요."""
                def select_task_prompt(question_type):
    
                    if question_type == "법 저촉 여부(다양한 관점 분석)":
                        return """
                        [Task 1: 단계별 지침]
                        1. 질문에서 주체와 객체를 식별하고, 그들의 관계(예: 상급자-하급자, 대기업-중소기업, 공공기관-민간기업 등)를 파악합니다. 객체가 여러명인 경우, 다양한 관계(예: 본인 - 하급자1, 본인 - 하급자2, 본인 - 상급자1, 본인 - 동료 등)를 고려해야합니다.
                        2. 식별된 관계에 따라 적용되는 법적 기준이 다를 수 있음을 고려합니다. 관계가 다수라면, 모든 관계를 나누어 구분지어서 고려합니다.
                        3. 관련이 있다면, 질문을 분석하여 법적 위반 가능성을 식별합니다.
                        4. 해당 행동이 법적으로 문제가 되지 않을 수 있는 관점과 문제가 될 수 있는 관점을 모두 고려합니다.
                        5. 각 관점에 대해 관련 법령과 함께 근거를 제시합니다.
                        6. 두 관점을 종합하여 균형 잡힌 결론을 도출합니다.
                        7. <제공된 문서>에서 특정 법률, 규정 또는 조항을 참조하십시오.
    
                        [Task 2: 출력 형식]
                        응답은 다음 주요 부분으로 구성되어야 합니다:
                        1. 관계 분석: '[관계 분석]'으로 시작하는 단락으로, 식별된 주체와 객체의 관계를 설명합니다. 객체가 여러명인 경우, 다양한 관계(예: 본인 - 하급자1, 본인 - 하급자2, 본인 - 상급자1, 본인 - 동료 등)를 고려해야합니다.
                        2. 법적으로 문제될 수 있는 관점: '[문제될 수 있는 관점]'으로 시작하는 단락으로, 해당 행동이 법적으로 문제될 수 있는 이유와 근거를 설명합니다.
                        3. 법적으로 문제되지 않을 수 있는 관점: '[문제되지 않을 수 있는 관점]'으로 시작하는 단락으로, 해당 행동이 법적으로 허용될 수 있는 이유와 근거를 설명합니다.
                        4. 결론: '[결론]'으로 시작하는 단락과 두 관점을 종합한 균형 잡힌 결론을 제시합니다.
                        5. 권고사항: '[권고사항]'으로 시작하는 단락으로, 법적 리스크를 최소화하면서 업무를 수행할 수 있는 방안을 제시합니다.
    
                        [Task 3: 품질 보증]
                        응답이 다음을 보장하도록 합니다:
                        1. 관련이 있는 경우, 문제될 수 있는 관점과 문제되지 않을 수 있는 관점을 균형있게 제시합니다.
                        2. 제공된 문서에서 정확한 법적 참조를 제공합니다.
                        3. 변호사의 페르소나와 일치하는 꼼꼼하게 법적 사실을 확인하고, 쉽게 단정하지 않는 어조를 유지하되, 조언과 권고를 포함합니다.
    
                        [Reflection]
                        각 응답이 법적 준수를 엄격히 따르고 명확하고 정확한 법적 참조를 제공하는지 확인합니다. 응답이 질문에서 제기된 모든 잠재적 법적 문제를 충분히 다루고 있는지 고려합니다.
    
                        [Feedback]
                        응답의 명확성과 유용성에 대한 피드백을 요청합니다. 법적 참조가 도움이 되었는지, 설명이 충분히 상세했는지를 사용자가 알려줄 것을 요청합니다.
    
                        [Constraints]
                        1. 응답은 <제공된 문서>에만 기반해야 합니다.
                        2. 법적으로 문제가 될 가능성이 있는 상황이라면 이에 대해 명확히 인지시켜주고 가능한 대안을 제시하며, 필요하다면 사내 변호사에게 상담을 권장해주세요.
                        3. 결론은 간결하게 제시되어야 합니다.
                        4. <제공된 문서>에서 답을 할 수 없는 질문 또는 해당 법률과 관련이 없는 질문에 대해서는 '이 질문은 해당 법과 관련성이 낮은 것으로 판단되어 답변할 수 없습니다.'라고 답을 해야 합니다.
                        5. 질문을 새로 생성하면 안됩니다.
    
                        [Context]
                        사용자는 대기업 KT의 프로젝트 관리자이며, 프로젝트 관리, 계약 및 규정 준수와 관련된 법적 질문을 다루고 있을 가능성이 큽니다.
                        """
                    
                    elif question_type == "질의응답":
                        return """
                        [Task 1: 단계별 지침]
                        1. 질문의 주요 키워드와 핵심 내용을 파악합니다.
                        2. 질문에서 주체와 객체를 식별하고, 그들의 관계를 파악합니다.
                        3. <제공된 문서>에서 질문과 관련된 정보를 찾습니다.
                        4. 관련 정보가 있다면, 다음 단계를 따라 답변을 구성합니다:
                           a) 관련 법규나 규정을 확인합니다.
                           b) 법규나 규정의 적용 조건을 검토합니다.
                           c) 질문 상황이 해당 조건에 부합하는지 분석합니다.
                           d) 조건에 부합한다면, 구체적인 적용 방법을 고려합니다.
                           e) 필요한 경우, 예외 사항이나 추가 고려 사항을 검토합니다.
                           f) 금액이 언급된 경우, 명시적으로 금액을 비교하고 그 결과를 명확히 설명합니다.
                        5. 각 단계의 결과를 바탕으로 최종 답변을 구성합니다.
                
                        [Task 2: 출력 형식]
                        응답은 다음 주요 부분으로 구성되어야 합니다:
                        1. 관계 분석: '[관계 분석]'으로 시작하는 단락
                        2. 단계별 분석: '[단계별 분석]'으로 시작하는 단락, 위의 4a부터 4e까지의 과정을 상세히 기술
                        3. 답변: '[답변]'으로 시작하는 단락
                        4. 출처: '[출처]'로 시작하는 단락
                
                        [Task 3: 품질 보증]
                        응답이 다음을 보장하도록 합니다:
                        1. 각 단계가 논리적으로 연결되어 있는지 확인합니다.
                        2. 모든 주장에 대해 <제공된 문서>의 근거가 있는지 확인합니다.
                        3. 결론이 단계별 분석과 일치하는지 확인합니다.
                        4. 금액 비교가 필요한 경우, 명확하게 "A는 B보다 크다/작다"와 같은 형식으로 비교 결과를 제시합니다.

                        [Reflection]
                        각 응답이 질문에 충실히 답하고 있는지, 필요한 정보를 모두 포함하고 있는지 확인합니다. 답변이 명확하고 이해하기 쉬운지 고려합니다.
    
                        [Feedback]
                        응답의 명확성과 유용성에 대한 피드백을 요청합니다. 제공된 정보가 충분했는지, 추가 설명이 필요한지 사용자가 알려줄 것을 요청합니다.
    
                        [Constraints]
                        1. 응답은 <제공된 문서>에만 기반해야 합니다.
                        2. 추측이나 개인적인 의견을 포함하지 않습니다.
                        3. <제공된 문서>에서 답을 할 수 없는 질문에 대해서는 '이 질문에 대한 정보는 제공된 문서에서 찾을 수 없으므로 답변할 수 없습니다.'라고 답해야 합니다.
                        4. 질문을 새로 생성하면 안됩니다.
                        5. 금액 관련 질문에서는 반드시 명시적인 비교 결과를 포함해야 합니다.

                        [Context]
                        사용자는 대기업 KT의 프로젝트 관리자이며, 프로젝트 관리, 계약 및 규정 준수와 관련된 일반적인 질문을 할 가능성이 큽니다.
                        """
                    
                    elif question_type == "금액 계산":
                        return """
                        [Task 1: 단계별 지침]
                        1. 질문에서 계산에 필요한 모든 정보와 변수를 식별합니다.
                        2. <제공된 문서>에서 계산에 필요한 추가 정보나 규정을 찾습니다.
                        3. 필요한 정보가 부족하다면, '계산에 필요한 일부 정보가 부족하여 답변할 수 없습니다.'라고 명시합니다.
                        4. 모든 정보가 있다면, 단계별로 계산 과정을 수행합니다.
                        5. 계산 결과를 명확하게 제시합니다.
                        6. 필요한 경우, 계산 결과에 대한 추가 설명이나 해석을 제공합니다.
    
                        [Task 2: 출력 형식]
                        응답은 다음 주요 부분으로 구성되어야 합니다:
                        1. 입력 정보: '[입력 정보]'로 시작하는 단락으로, 계산에 사용된 모든 변수와 값을 나열합니다.
                        2. 계산 과정: '[계산 과정]'으로 시작하는 단락으로, 단계별 계산 과정을 상세히 설명합니다.
                        3. 계산 결과: '[계산 결과]'로 시작하는 단락으로, 최종 계산 결과를 명확히 제시합니다.
                        4. 해석: '[해석]'으로 시작하는 단락으로, 필요한 경우 계산 결과에 대한 추가 설명이나 해석을 제공합니다.
                        5. 참고 사항: '[참고 사항]'으로 시작하는 단락으로, 계산에 적용된 규정이나 예외 사항 등을 명시합니다.
    
                        [Task 3: 품질 보증]
                        응답이 다음을 보장하도록 합니다:
                        1. 모든 계산이 정확하고 <제공된 문서>의 규정을 준수합니다.
                        2. 계산 과정이 명확하고 단계별로 설명되어 있습니다.
                        3. 최종 결과가 명확하게 제시되어 있습니다.
                        4. 필요한 경우, 결과에 대한 해석이나 추가 설명이 포함되어 있습니다.
    
                        [Reflection]
                        각 응답이 계산의 정확성을 보장하는지, 모든 필요한 정보를 포함하고 있는지 확인합니다. 계산 과정과 결과가 이해하기 쉽게 설명되어 있는지 고려합니다.
    
                        [Feedback]
                        응답의 명확성과 유용성에 대한 피드백을 요청합니다. 계산 과정이 이해하기 쉬웠는지, 결과 해석이 도움이 되었는지 사용자가 알려줄 것을 요청합니다.
    
                        [Constraints]
                        1. 모든 계산은 <제공된 문서>의 규정과 정보에 기반해야 합니다.
                        2. 추측이나 가정을 포함하지 않습니다. 정보가 부족할 경우 이를 명시합니다.
                        3. 계산에 필요한 정보가 부족할 경우 '계산에 필요한 일부 정보가 부족합니다.'라고 답해야 합니다.
                        4. 질문을 새로 생성하면 안됩니다.
    
                        [Context]
                        사용자는 대기업 KT의 프로젝트 관리자이며, 프로젝트 비용, 계약금액, 위약금 등과 관련된 금액 계산 질문을 할 가능성이 큽니다.
                        """
                    
                    else:  # "그 외"
                        return """
                        [Task 1: 단계별 지침]
                        1. 질문의 주요 키워드와 핵심 내용을 파악합니다.
                        2. 질문에서 주체와 객체를 식별하고, 그들의 관계(예: 상급자-하급자, 대기업-중소기업, 공공기관-민간기업 등)를 파악합니다. 객체가 여러명인 경우, 다양한 관계(예: 본인 - 하급자1, 본인 - 하급자2, 본인 - 상급자1, 본인 - 동료 등)를 고려해야합니다.
                        3. 식별된 관계에 따라 적용되는 법적 기준이 다를 수 있음을 고려합니다. 관계가 다수라면, 모든 관계를 나누어 구분지어서 고려합니다.
                        4. <제공된 문서>에서 질문과 관련된 정보를 찾습니다.
                        5. 관련 정보가 없다면, '이 질문에 대한 정보는 제공된 문서에서 찾을 수 없으므로 답변할 수 없습니다.'라고 답변합니다.
                        6. 관련 정보가 있다면, 해당 정보를 바탕으로 답변을 구성합니다.
                        7. 필요한 경우, 추가적인 설명이나 예시를 제공합니다.
                        8. 답변의 출처를 명확히 제시합니다.
    
                        [Task 2: 출력 형식]
                        응답은 다음 주요 부분으로 구성되어야 합니다:
                        1. 관계 분석: '[관계 분석]'으로 시작하는 단락으로, 식별된 주체와 객체의 관계를 설명합니다. 객체가 여러명인 경우, 다양한 관계(예: 본인 - 하급자1, 본인 - 하급자2, 본인 - 상급자1, 본인 - 동료 등)를 고려해야합니다.
                        2. 답변: '[답변]'으로 시작하는 단락으로, 질문에 대한 직접적인 답변을 제공합니다.
                        3. 설명: '[설명]'으로 시작하는 단락으로, 필요한 경우 추가적인 설명이나 예시를 제공합니다.
                        4. 출처: '[출처]'로 시작하는 단락으로, 답변의 근거가 되는 <제공된 문서>의 해당 부분을 명시합니다.
    
                        [Task 3: 품질 보증]
                        응답이 다음을 보장하도록 합니다:
                        1. 질문에 대한 정보가 <제공된 문서>에 없다면, 그 사실을 명확히 밝힙니다.
                        2. 답변은 정확하고 간결하며, 질문의 핵심을 다룹니다.
                        3. 추가 설명이나 예시는 이해를 돕는 데 필요한 경우에만 제공합니다.
                        4. 모든 정보의 출처를 명확히 제시합니다.
    
                        [Reflection]
                        각 응답이 질문에 충실히 답하고 있는지, 필요한 정보를 모두 포함하고 있는지 확인합니다. 답변이 명확하고 이해하기 쉬운지 고려합니다.
    
                        [Feedback]
                        응답의 명확성과 유용성에 대한 피드백을 요청합니다. 제공된 정보가 충분했는지, 추가 설명이 필요한지 사용자가 알려줄 것을 요청합니다.
    
                        [Constraints]
                        1. 응답은 <제공된 문서>에만 기반해야 합니다.
                        2. 추측이나 개인적인 의견을 포함하지 않습니다.
                        3. <제공된 문서>에서 답을 할 수 없는 질문에 대해서는 '이 질문에 대한 정보는 제공된 문서에서 찾을 수 없으므로 답변할 수 없습니다.'라고 답해야 합니다.
                        4. 질문을 새로 생성하면 안됩니다.
    
                        [Context]
                        사용자는 대기업 KT의 프로젝트 관리자이며, 프로젝트 관리, 계약 및 규정 준수와 관련된 일반적인 질문을 할 가능성이 큽니다.
                        """
    
                async def llm_answer_response(state: GraphState) -> GraphState:
                    client = openai.OpenAI()
                    current_reasoning_effort = st.session_state.get("reasoning_effort", "medium")

                    # 재생성 횟수가 3번을 초과하면 강제 종료
                    if state["attempts"] >= 3:
                        state["end"] = True
                        return state
                        
                    questions_and_answers = state["questions_and_answers"]
                    law_name = state["law_name"]
                    question_type = state["question_type"]
                    answer_mode = state["answer_mode"]  # answer_mode 추가
                    
                    # 컨텍스트 윈도우 제한 (최근 3개의 Q&A만 사용)
                    recent_qa = questions_and_answers[-3:]

                    combined_context = ""
                    for i, (question, answer) in enumerate(recent_qa):
                        # 현재 질문에 더 높은 가중치 부여
                        weight = 1.0 if i == len(recent_qa) - 1 else 0.5
                        
                        law_context = law_retrievers[law_name](question + " " + answer)
                        case_context = case_retrievers[law_name](question + " " + answer)
                        
                        # 관련성 점수를 사용한 컨텍스트 필터링
                        law_context = [doc for doc in law_context if doc.metadata.get('score', 0) > 0.5]
                        case_context = [doc for doc in case_context if doc.metadata.get('score', 0) > 0.5]
                        
                        # 관련성이 낮은 경우 질문 재생성 시도 (정확한 답변 모드에서만)
                        if answer_mode == "정확한 답변" and not law_context and not case_context:
                            state["attempts"] += 1
                            if state["attempts"] <= 3:  # 3번까지만 재시도
                                state = await rephrase_question(state)
                                return state

                        combined_context += f"질문 (가중치 {weight}): {question}\n"
                        combined_context += f"가상 답변 (가중치 {weight}): {answer}\n"
                        combined_context += f"법률 정보 (가중치 {weight}):\n{' '.join([doc.page_content for doc in law_context])}\n"
                        combined_context += f"사례 정보 (가중치 {weight}):\n{' '.join([doc.page_content for doc in case_context])}\n\n"

                    law_specific_prompt = LAW_SPECIFIC_PROMPTS.get(law_name, "전문 사내 변호사입니다.")
                    type_specific_prompt = select_persona_prompt(question_type)
                    task_specific_prompt = select_task_prompt(question_type)

                    try:
                        original_question = questions_and_answers[-1][0]  # 가장 최근 질문 사용
                        
                        # 시스템 메시지 구성
                        system_message = f"당신은 {law_name} 전문 사내 변호사입니다. AI는 꼼꼼하게 법적 사실을 확인하고, {law_name}에 따라 법적으로 문제가 되지 않을 가능성과 법을 위반할 가능성이 있는 모든 행동에 대해 명확히 안내하고 <제공된 문서>를 기반으로 상세한 설명을 제공해야 합니다. <제공된 문서>: {combined_context}"
                        
                        # 사용자 메시지 구성
                        user_message = f"[Persona]{law_specific_prompt} {type_specific_prompt}[Input] AI는 다음 형식의 질문과 관련 정보를 받게 됩니다: {original_question} {task_specific_prompt} Let's think step by step"

                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": user_message}
                            ],
                            reasoning_effort=current_reasoning_effort  # 선택된 reasoning_effort 사용
                        )

                        response_content = response.choices[0].message.content
                        
                        memory.chat_memory.add_user_message(original_question)
                        memory.chat_memory.add_ai_message(response_content)
                    
                        state["attempts"] += 1
                        relevance = check_relevance(original_question, combined_context, response_content)
                        state["relevance"] = relevance

                        # 빠른 답변 모드에서는 질문 재생성 없이 바로 응답
                        if answer_mode == "빠른 답변":
                            if law_name.lower() in response_content.lower():
                                state["response"] = response_content
                            else:
                                state["response"] = f"'{law_name}'과 관련성이 낮은 것으로 판단되어 답변할 수 없습니다."
                            state["end"] = True
                            return state

                        # 정확한 답변 모드에서는 기존 로직 유지
                        if answer_mode == "정확한 답변":
                            if relevance not in ["grounded", "notSure"] and state["attempts"] < 3:
                                state = await rephrase_question(state)
                                return state

                        # 최대 시도 횟수 도달 또는 관련성 검사 통과
                        if state["attempts"] >= 3:
                            state["response"] = "죄송합니다. 적절한 답변을 생성하지 못하여, 답변할 수 없습니다. 다른 방식으로 질문을 해주시거나, 더 자세한 정보를 제공해 주세요."
                            state["end"] = True
                        else:
                            if law_name.lower() in response_content.lower():
                                state["response"] = response_content
                            else:
                                state["response"] = f"'{law_name}'과 관련성이 낮은 것으로 판단되어 답변할 수 없습니다."
                        return state

                    except Exception as e:
                        print(f"Error in llm_answer_response: {e}")
                        state["response"] = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                        state["end"] = True
                        return state
    
                def select_final_prompt(question_type):
                    base_prompt = "당신은 여러 법률 전문가의 의견을 종합하여 최종 답변을 제시하는 역할입니다. 질문자는 기본적으로 대기업이자 사기업인 KT의 직원으로, 공직자가 아닌 민간 기업의 직원입니다. KT는 정부 기관이 아니며, 직원들은 공무원이 아닙니다."
                    
                    additional_question_prompt = """
                    ### 질문 TIP 💡 (필요한 경우)
                    (답변을 더 정확하게 제공하기 위해 필요한 추가 정보나 상황에 대한 질문을 제시)
                    
                    위 질문들에 대한 정보를 주시면 더 정확한 답변을 얻으실 수 있습니다.
                    """
                
                    if question_type == "법 저촉 여부(다양한 관점 분석)":
                        return base_prompt + f"""제공된 법령별 검토 결과를 바탕으로 종합적이고 정확한 답변을 제공해야 합니다. 답변 시 다음 구조와 지침을 따르세요:
                
                        ### 결론
                        - 모든 법령의 검토 결과를 종합한 질문에 대한 최종 답변
                        - 문제 될 수 있는 관점에서 종합 결론
                        - 문제되지 않을 수 있는 관점에서 종합 결론
                        - 주의해야 할 점
                
                        ### 법령별 검토 결과
                        (관련된 각 법률에 대해 다음 구조로 작성)
                        [숫자]. [법률 이름] 검토
                        (해당 법률 검토시 문제될 수 있는 관점과 문제되지 않을 수 있는 관점을 요약하여 제시)

                        ### 사례에 기반한 답변
                        (가장 관련성 높은 사례를 기반으로 한 법적 판단과 조언)

                        (위 구조를 관련된 모든 법률에 대해 반복)

                        ### 권고사항
                        (모든 법률에 대해 문제될 수 있는 관점과 문제되지 않을 수 있는 관점을 고려하여 종합적인 권고사항 제시)
                
                        ### 주의사항
                        (법적 해석의 한계, 추가 법률 자문의 필요성 등 언급)

                        ### 질문 TIP 💡
                        (더 정확하고 구체적인 답변을 위해서 필요한 정보 요구)
                
                        ### 관련 법령
                        (분석에 사용된 모든 법령 조항을 정확히 나열. 법명과 조항은 1세트로 계속 같이 나와야 함. (예: 참고 법령: 산업안전보건법 제26조, 산업안전보건법 시행규칙 제27조, 산업안전보건기준에 관한 규칙 제28조, 중대재해처벌법 제8조, 상생협력법 제20조의2, 청탁금지법 시행령 제26조, 청탁금지법 시행령 별표2, 부패방지 행동강령 제3조, 윤리경영원칙 실천지침 4-3))
                        
                        {additional_question_prompt}
                        
                        답변 작성 시 다음 사항을 준수하세요:
                        1. 제공된 법령별 검토 결과만을 사용하여 답변을 작성하세요.
                        2. 각 법률의 관점을 균형있게 고려하여 종합적인 답변을 제공하세요.
                        3. 법령 간 충돌이 있는 경우, 이를 명시하고 가장 적절한 해석을 제시하세요.
                        4. 확실하지 않은 부분에 대해서는 명확히 언급하세요.
                        5. 모든 법령 조항을 정확히 명시하세요. 법명과 조항은 1세트로 계속 같이 명시하세요.
                        6. 질문과 관련이 없는 법률은 분석에서 제외하세요.
                        7. 각 법률 분석은 간결하면서도 충분한 정보를 포함하도록 하세요.
                        8. 답변 시 항상 사기업 직원의 관점에서 적용되는 법률을 고려하여 설명하세요.
                        9. 답변 작성 과정에서 필요한 추가 정보나 상황에 대해 명시하고, 이 내용들을 '질문 TIP 💡' 섹션에 포함시키세요. 해당 내용은 법적 해석이나 상황별 변수를 명확히 하기 위해 구체적이고 실질적이어야 합니다.
                        10. '사례에 기반한 답변' 섹션에서는 반드시 "이 사례를 미루어보아..."로 시작하는 결론을 제시하세요.

                        """
                    
                    elif question_type == "단순 질의응답":
                        return base_prompt + f"""
                        제공된 법령별 검토 결과들을 바탕으로 종합적이고 정확한 답변을 제공해야 합니다. 답변 시 다음 구조와 지침을 따르세요:
                
                        ### 요약 답변
                        - 질문에 대한 간략하고 직접적인 답변
                        - 핵심 포인트 나열 (2-3개)
                
                        ### 법령별 검토 결과
                        (관련된 각 법률에 대해 다음 구조로 작성)
                        [숫자]. [법률 이름] 검토

                        ### 사례에 기반한 답변
                        (가장 관련성 높은 사례를 기반으로 한 법적 판단과 조언)

                        (위 구조를 관련된 모든 법률에 대해 반복)
                
                        ### 주의사항
                        - 답변의 한계 또는 예외 상황 언급
                        - 추가 확인이 필요한 사항 안내

                        ### 질문 TIP 💡
                        (더 정확하고 구체적인 답변을 위해서 필요한 정보 요구)

                        ### 관련 법령
                        - 답변에 사용된 모든 정보 소스를 정확히 나열
                        (예: 참고 법령: 산업안전보건법 제26조, 산업안전보건법 시행규칙 제27조, 산업안전보건기준에 관한 규칙 제28조, 중대재해처벌법 제8조, 상생협력법 제20조의2, 청탁금지법 시행령 제26조, 청탁금지법 시행령 별표2, 부패방지 행동강령 제3조, 윤리경영원칙 실천지침 4-3)
                
                        {additional_question_prompt}
                
                        답변 작성 시 다음 사항을 준수하세요:
                        1. 제공된 법령별 검토 결과의 내용만을 사용하여 답변을 작성하세요.
                        2. 관계 분석을 기반으로, 질문의 모든 측면을 균형있게 다루어 종합적인 답변을 제공하세요.
                        3. 정보 간 불일치가 있는 경우, 이를 명시하고 가장 신뢰할 수 있는 정보를 제시하세요.
                        4. 확실하지 않은 부분에 대해서는 명확히 언급하세요.
                        5. 모든 법령 조항을 정확히 명시하세요. 법명과 조항은 1세트로 계속 같이 명시하세요.
                        6. 질문과 관련이 없는 정보는 답변에서 제외하세요.
                        7. 각 설명은 간결하면서도 충분한 정보를 포함하도록 하세요.
                        8. 전문 용어를 사용할 경우, 필요에 따라 간단한 설명을 추가하세요.
                        9. 답변은 객관적이고 중립적인 톤을 유지하세요.
                        10. 필요한 경우, 추가 문의나 전문가 상담을 권장하세요.
                        11. 답변 시 항상 사기업 직원의 관점에서 적용되는 법률을 고려하여 설명하세요.
                        12. 답변 작성 과정에서 필요한 추가 정보나 상황에 대해 명시하고, 이 내용들을 '질문 TIP 💡' 섹션에 포함시키세요. 해당 내용은 법적 해석이나 상황별 변수를 명확히 하기 위해 구체적이고 실질적이어야 합니다. 
                        """
                    
                    elif question_type == "금액 계산":
                        return base_prompt + f"""
                        제공된 법령별 검토 결과를 바탕으로 상세하고 정확한 계산 결과를 제공해야 합니다. 답변 시 다음 구조와 지침을 따르세요:
                
                        ### 계산 결과 요약
                        - 최종 계산된 금액
                        - 계산 결과에 대한 간략한 설명 (1-2문장)
                
                        ### 입력 정보
                        - 계산에 사용된 모든 변수와 값을 나열
                        - 각 변수의 출처 또는 근거 명시
                
                        ### 계산 과정
                        (각 단계별로 다음 구조로 작성)
                        [숫자]. [계산 단계 설명]
                        - 사용된 공식 또는 규칙
                        - 세부 계산 과정
                        - 중간 결과값
                
                        (위 구조를 모든 주요 계산 단계에 대해 반복)
                
                        ### 법령별 검토 결과
                        (관련된 각 법률에 대해 다음 구조로 작성)
                        [숫자]. [법률 이름] 검토
                
                        (위 구조를 관련된 모든 법률에 대해 반복)
                
                        ### 결과 해석
                        - 계산 결과의 의미 설명
                        - 결과가 미치는 영향이나 중요성 언급
                
                        ### 주의사항
                        - 계산 결과 적용 시 고려해야 할 제한사항
                        - 예외 상황이나 변동 가능성 언급
                
                        ### 추가 고려사항
                        - 계산에 영향을 줄 수 있는 기타 요소
                        - 필요한 경우 대안적 계산 방법 제시
                
                        ### 관련 법령
                        - 답변에 사용된 모든 법령과 사례를 정확히 나열. 법명과 조항은 1세트로 계속 같이 명시 (예: 참고 법령: 산업안전보건법 제26조, 산업안전보건법 시행규칙 제27조, 산업안전보건기준에 관한 규칙 제28조, 중대재해처벌법 제8조, 상생협력법 제20조의2, 청탁금지법 시행령 제26조, 청탁금지법 시행령 별표2, 부패방지 행동강령 제3조, 윤리경영원칙 실천지침 4-3)
                
                        {additional_question_prompt}
                
                        답변 작성 시 다음 사항을 준수하세요:
                        1. 제공된 법령별 검토 결과의 내용과 관련 규정만을 사용하여 계산을 수행하세요.
                        2. 모든 계산 단계를 명확하고 상세하게 설명하세요.
                        3. 사용된 모든 변수와 값의 출처를 명확히 밝히세요.
                        4. 계산 과정에서 가정이나 추정이 필요한 경우, 이를 명시하고 그 근거를 제시하세요.
                        5. 최종 결과뿐만 아니라 중간 계산 결과도 제시하세요.
                        6. 계산 결과의 의미와 영향을 설명하세요.
                        7. 계산 결과의 한계나 주의사항을 명확히 언급하세요.
                        8. 필요한 경우, 추가 검토나 전문가 확인을 권장하세요.
                        9. 모든 금액은 원 단위까지 정확히 계산하고, 필요에 따라 반올림 여부를 명시하세요.
                        10. 복잡한 계산의 경우, 단계별로 나누어 설명하세요.
                        11. 계산에 필요한 정보가 부족한 경우, '질문 TIP 💡' 섹션에 필요한 정보를 요청하는 질문을 포함시키세요.
                        """
                    
                    else:  # "그 외"
                        return base_prompt + f"""
                        제공된 법령별 검토 결과의 내용을 바탕으로 포괄적이고 유용한 정보를 제공해야 합니다. 답변 시 다음 구조와 지침을 따르세요:
                
                        ### 핵심 답변
                        - 질문의 핵심에 대한 간결하고 직접적인 답변
                        - 주요 포인트 요약 (2-3개)
                
                        ### 법령별 검토 결과
                        (관련된 각 법률에 대해 다음 구조로 작성)
                        [숫자]. [법률 이름] 검토
                
                        (위 구조를 관련된 모든 법률에 대해 반복)
                
                        ### 주의사항 및 제한점
                        - 제공된 정보의 한계 또는 예외 상황 언급
                        - 추가 확인이 필요한 사항 안내
                
                        ### 추천 사항 또는 다음 단계
                        - 질문과 관련하여 권장되는 행동이나 절차
                        - 추가 정보를 얻을 수 있는 방법 제안
                
                        ### 관련 법령
                        - 답변에 사용된 모든 법령과 사례를 정확히 나열. 법명과 조항은 1세트로 계속 같이 명시 (예: 참고 법령: 산업안전보건법 제26조, 산업안전보건법 시행규칙 제27조, 산업안전보건기준에 관한 규칙 제28조, 중대재해처벌법 제8조, 상생협력법 제20조의2, 청탁금지법 시행령 제26조, 청탁금지법 시행령 별표2, 부패방지 행동강령 제3조, 윤리경영원칙 실천지침 4-3)
                
                        {additional_question_prompt}
                
                        답변 작성 시 다음 사항을 준수하세요:
                        1. 제공된 법령별 검토 결과의 내용만을 사용하여 답변을 작성하세요.
                        2. 질문의 모든 측면을 균형있게 다루어 종합적인 답변을 제공하세요.
                        3. 정보 간 불일치가 있는 경우, 이를 명시하고 가장 신뢰할 수 있는 정보를 제시하세요.
                        4. 확실하지 않은 부분에 대해서는 명확히 언급하세요.
                        5. 모든 참고 자료와 출처를 정확히 명시하세요.
                        6. 질문과 관련이 없는 정보는 답변에서 제외하세요.
                        7. 각 설명은 간결하면서도 충분한 정보를 포함하도록 하세요.
                        8. 전문 용어를 사용할 경우, 필요에 따라 간단한 설명을 추가하세요.
                        9. 답변은 객관적이고 중립적인 톤을 유지하세요.
                        10. 필요한 경우, 추가 문의나 전문가 상담을 권장하세요.
                        11. 질문의 성격에 따라 답변 구조를 유연하게 조정하세요.
                        12. 필요한 경우, 답변 작성 과정에서 필요한 추가 정보나 상황에 대해 명시하고, 이 내용들을 '질문 TIP 💡' 섹션에 포함시키세요. 해당 내용은 법적 해석이나 상황별 변수를 명확히 하기 위해 구체적이고 실질적이어야 합니다. 
                        """
                
                def calculate_tokens_per_law(selected_laws, total_tokens=60000):
                    num_laws = len(selected_laws)
                    tokens_per_law = max(1000, total_tokens // ((num_laws)*2))  # 최소 1000 토큰 보장
                    return tokens_per_law

                async def generate_single_law_answer(law_name: str, question: str, question_type: str, answer_mode: str = "빠른 답변", org_info: Dict = None) -> Dict:
                    client = openai.OpenAI()
                    memory = ConversationBufferMemory(return_messages=True)
                    current_reasoning_effort = st.session_state.get("reasoning_effort", "medium")

                    state = GraphState(
                        question=question,
                        questions_and_answers=[],
                        law_context="",
                        case_context="",
                        response="",
                        relevance="",
                        attempts=0,
                        law_name=law_name,
                        law_references=[],
                        similar_cases=[],
                        question_type=question_type,
                        answer_mode=answer_mode
                    )    

                    global law_retrievers, case_retrievers, fcpa_retrievers
                    
                    if not law_retrievers:
                        law_retrievers = create_law_retrievers(selected_laws)
                    
                    law_retriever = law_retrievers.get(law_name)

                    if not law_retriever:
                        return state

                    if answer_mode == "빠른 답변":
                        max_attempts = 1
                    else:
                        max_attempts = 3

                    for attempt in range(max_attempts):
                        try:
                            law_retriever = law_retrievers[law_name]
                            law_context = law_retriever.get_relevant_documents(state["question"])
                            
                            case_retriever = case_retrievers[law_name]
                            case_context = case_retriever.get_relevant_documents(state["question"])
                            tokens_per_law = calculate_tokens_per_law(selected_laws)
                            max_context_length = 60000 // (len(selected_laws)+1)
                            
                            most_relevant_case = max(case_context, key=lambda x: x.metadata.get('score', 0)) if case_context else None
                            
                            combined_context = f"질문: {state['question']}\n"
                            combined_context += f"법률 정보:\n{' '.join([doc.page_content for doc in law_context])[:max_context_length//2]}\n"
                            combined_context += f"사례 정보:\n{' '.join([doc.page_content for doc in case_context])[:max_context_length//2]}\n"
                            
                            if most_relevant_case:
                                combined_context += f"\n가장 관련성 높은 사례:\n{most_relevant_case.page_content}\n"
                            
                            # FCPA 정보 추가
                            fcpa_context = ""
                            if fcpa_retrievers:
                                for fcpa_name, retriever in fcpa_retrievers.items():
                                    fcpa_docs = retriever(state["question"])
                                    if fcpa_docs:
                                        fcpa_context += f"\nFCPA 관련 정보 ({fcpa_name}):\n"
                                        fcpa_context += ' '.join([doc.page_content for doc in fcpa_docs][:max_context_length//4])
                            
                            # 조직 정보 추가
                            org_context = ""
                            if org_info and isinstance(org_info, dict) and len(org_info) > 0:
                                org_context += "\n질문에 언급된 기관 정보:\n"
                                for org, info in org_info.items():
                                    org_context += f"기관명: {org}\n"
                                    
                                    # 정보 추출 및 추가
                                    if isinstance(info, dict) and 'analysis' in info:
                                        analysis_text = info['analysis']
                                        org_context += f"{analysis_text}\n\n"
                                    else:
                                        org_context += f"{info}\n\n"

                            law_specific_prompt = LAW_SPECIFIC_PROMPTS.get(law_name, "전문 사내 변호사입니다.")
                            type_specific_prompt = select_persona_prompt(question_type)
                            task_specific_prompt = select_task_prompt(question_type)

                            # 프롬프트에 조직 정보와 FCPA 정보 추가
                            additional_prompt = ""
                            if org_info and len(org_info) > 0 or fcpa_context:
                                additional_prompt = """
                                중요: 질문에 언급된 기관의 유형과 법률 적용 여부를 고려하여 답변하세요. 
                                공공기관, 공직유관단체, 교육기관 등은 청탁금지법 적용 대상일 가능성이 높으며,
                                미국과 연관된 기업이나 미국에 상장된 기업은 FCPA 적용 대상일 수 있습니다.
                                다음 정보를 검토하고 이를 답변에 반영하세요:
                                """

                            response = client.chat.completions.create(
                                model="o3-mini",
                                messages=[
                                    {"role": "system", "content": f"당신은 {law_name} 전문 사내 변호사입니다. 다른 법률은 고려하지 말고 오직 {law_name}에 관해서만 답변해주세요. " + 
                                    f"AI는 꼼꼼하게 법적 사실을 확인하고, {law_name}에 따라 법적으로 문제가 되지 않을 가능성과 법을 위반할 가능성이 있는 모든 행동에 대해 명확히 안내하고 <제공된 문서>를 기반으로 상세한 설명을 제공해야 합니다. " +
                                    f"답변은 최대 {tokens_per_law} 토큰을 초과하지 않도록 해주세요. <제공된 문서>: {combined_context} {fcpa_context} {org_context}"},
                                    {"role": "user", "content": f"[Persona]{law_specific_prompt} {type_specific_prompt} {additional_prompt}\n[Input] AI는 다음 형식의 질문과 관련 정보를 받게 됩니다: [[[질문]]] {question} [[[/질문]]] {task_specific_prompt}\nLet's think step by step"}
                                ],
                                reasoning_effort=current_reasoning_effort
                            )
                                    
                            state["response"] = response.choices[0].message.content
                            state["law_context"] = combined_context + fcpa_context  # FCPA 컨텍스트 추가
                            state["relevance"] = check_relevance(state["question"], combined_context + fcpa_context, state["response"])

                            if answer_mode == "정확한 답변" and state["relevance"] not in ["grounded", "notSure"]:
                                state = await rephrase_question(state)
                            else:
                                break

                        except Exception as e:
                            print(f"Error in generate_single_law_answer: {e}")
                            state["response"] = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                            break

                    # 유사 사례 검색 및 저장
                    similar_cases = []
                    query_embedding = embeddings.embed_query(state["question"])
                    case_docs = case_retriever.get_relevant_documents(state["question"])
                    
                    if case_docs:
                        doc_embeddings = embeddings.embed_documents([doc.page_content for doc in case_docs])
                        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
                        
                        top_cases = sorted(zip(case_docs, similarities), key=lambda x: x[1], reverse=True)
                        
                        for case_doc, case_score in top_cases[:3]:
                            similar_cases.append({
                                'source': law_name,
                                'page': case_doc.metadata.get('page', 'N/A'),
                                'score': case_score,
                                'content': case_doc.page_content
                            })
                    
                    state["similar_cases"] = similar_cases
                    return state
                
                def process_law_in_thread(law, user_input, question_type, answer_mode, org_info=None):
                    # 각 법률별 처리를 위한 새로운 이벤트 루프 생성
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(generate_single_law_answer(law, user_input, question_type, answer_mode, org_info))
                    finally:
                        loop.close()
                def process_law_in_thread(law, user_input, question_type, answer_mode, org_info=None):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        try:
                            return loop.run_until_complete(generate_single_law_answer(law, user_input, question_type, answer_mode, org_info))
                        except TypeError:
                            # 이전 버전과의 호환성을 위한 처리
                            return loop.run_until_complete(generate_single_law_answer(law, user_input, question_type, answer_mode))
                    finally:
                        loop.close()
                        
                async def process_user_input_parallel(user_input, selected_laws, question_type, org_info=None):
                    # 현재 선택된 답변 모드 가져오기
                    current_answer_mode = st.session_state.get("answer_mode_select", "빠른 답변")
                    
                    # 선택된 법률에 대한 벡터 데이터베이스 생성
                    selected_law_vectordbs = create_selected_law_vectordbs(selected_laws)
                    
                    # 법령별 리트리버 생성
                    law_retrievers = create_law_retrievers(selected_law_vectordbs)
                    
                    # 사례별 리트리버 생성
                    selected_case_vectordbs = create_selected_case_vectordbs(selected_laws)
                    case_retrievers = create_case_retrievers(selected_case_vectordbs)
                    
                    # FCPA 리트리버도 추가
                    global fcpa_retrievers
                    if not fcpa_retrievers:
                        fcpa_retrievers = create_fcpa_retrievers(fcpa_vectordbs)
                    
                    # ThreadPoolExecutor를 사용하여 병렬 처리
                    loop = asyncio.get_event_loop()
                    futures = []
                    
                    for law in selected_laws:
                        if law in law_retrievers:
                            # process_law_in_thread 함수에 org_info 전달
                            try:
                                future = loop.run_in_executor(
                                    executor,
                                    process_law_in_thread,
                                    law,
                                    user_input,
                                    question_type,
                                    current_answer_mode,
                                    org_info  # 기관 정보 전달
                                )
                            except TypeError:
                                # 이전 버전과의 호환성을 위한 처리
                                future = loop.run_in_executor(
                                    executor,
                                    process_law_in_thread,
                                    law,
                                    user_input,
                                    question_type,
                                    current_answer_mode
                                )
                            futures.append(future)
                    
                    # 모든 결과 수집
                    law_answers = await asyncio.gather(*futures)
                    return [answer for answer in law_answers if answer is not None]
                
                def generate_final_answer(question: str, law_answers: List[Dict], org_info=None) -> Dict:
                    client = openai.OpenAI()
                    current_reasoning_effort = st.session_state.get("reasoning_effort", "medium")

                    combined_answers = ""
                    combined_context = ""
                    
                    # 법률 답변과 컨텍스트를 토큰 제한 내에서 결합
                    for ans in law_answers:
                        new_answer = f"법률: {ans.get('law_name', 'Unknown')}\n답변: {ans.get('response', 'No response')}\n관련성: {ans.get('relevance', 'Unknown')}\n\n"
                        new_context = f"법률: {ans.get('law_name', 'Unknown')}\n법령 정보: {ans.get('law_context', 'No context')}\n\n"
                            
                        combined_answers += new_answer
                        combined_context += new_context
                    
                    # 조직 정보 추가
                    org_context = ""
                    if org_info and isinstance(org_info, dict) and len(org_info) > 0:
                        org_context += "\n### 질문에 언급된 기관 유형 및 법률 적용 정보\n"
                        for org, info in org_info.items():
                            org_context += f"#### {org}\n"
                            if isinstance(info, dict) and 'analysis' in info:
                                org_context += f"{info['analysis']}\n\n"
                                # FCPA 문서에서 추출한 관련 정보 요약 추가
                                if 'fcpa_context' in info and info['fcpa_context'] != "관련 FCPA 문서 정보 없음":
                                    fcpa_summary = info['fcpa_context']
                                    if len(fcpa_summary) > 1000:
                                        fcpa_summary = fcpa_summary[:1000] + "..."
                                    org_context += f"FCPA 문서 정보 (요약): {fcpa_summary}\n\n"
                            else:
                                org_context += f"{info}\n\n"
                    
                    try:
                        system_content = select_final_prompt(st.session_state.question_type_select) + """
                        
                        ###관련 법령을 언급할 때는 다음 예시와 같이 각 조항마다 법률 이름을 반복해서 명시해주세요:

                        올바른 예시:
                        산업안전보건법 제30조, 산업안전보건법 제72조, 산업안전보건법 제74조
                        청탁금지법 제8조, 청탁금지법 제23조
                        하도급법 제3조의4, 하도급법 제25조

                        잘못된 예시:
                        산업안전보건법 제30조, 제72조, 제74조
                        청탁금지법 제8조, 23조
                        하도급법 제3조의4, 25조

                        항상 각 조항 앞에 해당 법률의 이름을 명시해주세요.
                        
                        ###과거 판례나 유사 사례를 고려할 때는 다음 지침을 따르세요:
                        1. 제공된 유사 사례가 현재 질문과 얼마나 유사한지 평가하세요.
                        2. 유사 사례의 결정이 현재 상황에 어떻게 적용될 수 있는지 분석하세요.
                        3. 유사 사례와 현재 상황의 차이점을 명확히 설명하세요.
                        4. 유사 사례를 바탕으로 한 예측을 제시하되, 각 사례의 특수성을 고려하여 단정적인 결론을 내리지 않도록 주의하세요.

                        ### 기관 유형 및 법률 적용에 관한 정보:
                        답변 시 질문에 언급된 기관의 유형(공공기관, 민간기업 등) 및 법률 적용 여부(청탁금지법, FCPA 등)를 고려하세요.
                        다른 유형의 기관에는 다른 법률이 적용될 수 있으므로, 정확한 법률 적용 범위를 설명해야 합니다.
                        """

                        # org_context가 존재하면 프롬프트에 추가
                        if org_context:
                            system_content += f"\n\n질문에 언급된 기관에 대한 정보:\n{org_context}"

                        response = client.chat.completions.create(
                            model="o3-mini",
                            messages=[
                                {"role": "system", "content": system_content},
                                {"role": "user", "content": f"질문: [[[질문]]] {question} [[[/질문]]]\n\n각 법령별 검토 결과:\n{combined_answers}\n\n법령 및 사례 정보:\n{combined_context}"}
                            ],
                            reasoning_effort=current_reasoning_effort  # 선택된 reasoning_effort 사용
                        )

                        final_response_content = response.choices[0].message.content
                        
                        mentioned_laws = list(set(re.findall(r'((?:\w+법|(?:\w+기준에 관한 규칙))(?:\s+시행(?:령|규칙))?)\s+((?:제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?(?:\s*[가-힣]목)?)|(?:별표\s*\d+))', final_response_content)))
                        
                        # 각 법률의 유사 사례를 종합
                        similar_cases = {}
                        for law_answer in law_answers:
                            law_name = law_answer['law_name']
                            if law_answer['similar_cases']:
                                similar_cases[law_name] = law_answer['similar_cases']

                        return {
                            "question": question,
                            "law_context": combined_context,
                            "case_context": "",
                            "response": final_response_content,
                            "relevance": "",
                            "attempts": 0,
                            "law_name": "종합",
                            "law_answers": law_answers,
                            "mentioned_laws": mentioned_laws,
                            "law_references": [],
                            "similar_cases": similar_cases,
                            "org_info": org_info  # 조직 정보 저장
                        }

                    except Exception as e:
                        print(f"Error in generate_final_answer: {e}")
                        return {
                            "question": question,
                            "law_context": combined_context,
                            "case_context": "",
                            "response": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
                            "relevance": "",
                            "attempts": 0,
                            "law_name": "종합",
                            "law_answers": law_answers,
                            "mentioned_laws": [],
                            "law_references": [],
                            "similar_cases": {},
                            "org_info": org_info  # 조직 정보 저장
                        }
    
                def check_relevance(question: str, context: str, answer: str) -> str:
                    max_tokens = 30000  # 안전 마진을 위해 최대 토큰 수를 32768보다 낮게 설정
                    
                    # 컨텍스트 길이 제한
                    context = context[:max_tokens]
                    
                    result_input = {
                        "context": question + "\n" + context,
                        "answer": answer
                    }
                    relevance_result = upstage_ground_checker.invoke(result_input)
                    return relevance_result
    
                async def rephrase_question(state: GraphState) -> GraphState:
                    llm = ChatOpenAI(model_name=model_name, temperature=0.7)
                    
                    # 질문 개수 결정
                    num_questions = min(state["attempts"] + 1, 3)
                    
                    law_name = state["law_name"]
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", f"""당신은 {law_name} 전문가입니다. 주어진 원래 질문에 대한 답변을 더 잘 찾기 위해 참조 질문 3개를 재구성해야 합니다:
                    
                            1. 원래 질문에 대한 답변을 더 찾기 위해서, 원래 질문에 대한 {num_questions}개의 연관된 질문을 생성하세요.
                            2. 생성된 질문은 원래 질문의 다양한 측면을 다루거나 더 구체적인 정보를 요구해야 합니다.
                            3. 각 질문에 대해 가상의 짧은 답변을 생성하세요. 이 답변은 실제 법률 정보를 포함할 필요는 없지만, 관련 법률 개념이나 용어를 포함해야 합니다.
                    
                            출력 형식:
                            질문: [원래 질문 그대로 유지]
                            위 질문에 대한 답을 구하기 위해 참고할 가상 질문과 답변은 다음과 같습니다.
                            질문 1: [원래 질문 그대로 유지]
                            답변 1: [원래 질문에 대한 가상의 짧은 답변]
                    
                            질문 2: [생성된 관련 질문]
                            가상 답변 2: [관련 질문에 대한 가상의 짧은 답변]
                    
                            가상 질문 3: [생성된 관련 질문] 
                            가상 답변 3: [관련 질문에 대한 가상의 짧은 답변]
                            """),
                
                        ("human", "원래 질문: {question}\n\n위 지시사항에 따라 질문과 가상 답변을 해주세요.")
                    ])
                    
                    chain = prompt | llm
                    response = await chain.ainvoke({"question": state["question"]})
                    
                    # 응답 파싱
                    lines = response.content.split('\n')
                    questions_and_answers = []
                    for i in range(0, len(lines), 3):
                        if i + 1 < len(lines):
                            question = lines[i].split(': ', 1)[1] if ': ' in lines[i] else lines[i]
                            answer = lines[i+1].split(': ', 1)[1] if ': ' in lines[i+1] else lines[i+1]
                            questions_and_answers.append((question.strip(), answer.strip()))
                    
                    state["questions_and_answers"] = questions_and_answers
                    state["attempts"] += 1
                    if state["attempts"] >= 3:
                        state["end"] = True  # 3번 시도 후 강제 종료를 위한 플래그 추가
                    
                    # 재생성된 질문 중 첫 번째 질문을 새로운 질문으로 설정
                    if questions_and_answers:
                        state["question"] = questions_and_answers[0][0]
                    
                    return state
                
                def should_continue(state: GraphState) -> str:
                    # 빠른 답변 모드일 경우 항상 바로 종료
                    if state.get("answer_mode") == "빠른 답변":
                        return "end"
                    
                    # 종료 조건이나 최대 시도 횟수 도달 시 종료 
                    if state.get("end", False) or state["attempts"] >= 3:
                        return "end"
                    
                    # 정확한 답변 모드에서 재질문 진행 여부 결정
                    if state["answer_mode"] == "정확한 답변":
                        if state["relevance"] in ["grounded", "notSure"]:
                            return "end"
                        return "rephrase"
                    
                    return "end"
                
                if "feedback_submitted" not in st.session_state:
                    st.session_state.feedback_submitted = {}

                # 대화 내역을 저장하기 위한 세션 상태 변수 추가
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []
                
                def display_feedback_buttons(i, question, answer, selected_laws):
                    user_id = st.session_state.get("logged_in_user", "Unknown")
                    feedback_key = f"{user_id}_{question}_{answer}"
                    
                    if "feedback_states" not in st.session_state:
                        st.session_state.feedback_states = {}
                    
                    current_feedback = st.session_state.feedback_states.get(feedback_key)
                
                    col1, col2 = st.columns([9, 1])
                    with col2:
                        st.markdown('<div class="feedback-container">', unsafe_allow_html=True)
                        st.markdown('<div class="feedback-buttons">', unsafe_allow_html=True)
                        like, dislike = st.columns(2)
                        with like:
                            if st.button("👍", key=f"like_{i}", help="좋아요"):
                                if current_feedback != "좋아요":
                                    st.session_state.feedback_states[feedback_key] = "좋아요"
                                    st.session_state[f"show_text_feedback_{i}"] = True
                                    st.success("좋아요 피드백이 제출되었습니다.")
                                else:
                                    st.session_state.feedback_states[feedback_key] = None
                                    st.session_state[f"show_text_feedback_{i}"] = False
                                    st.info("좋아요 피드백이 취소되었습니다.")
                                st.experimental_rerun()
                        with dislike:
                            if st.button("👎", key=f"dislike_{i}", help="싫어요"):
                                if current_feedback != "싫어요":
                                    st.session_state.feedback_states[feedback_key] = "싫어요"
                                    st.session_state[f"show_text_feedback_{i}"] = True
                                    st.success("싫어요 피드백이 제출되었습니다.")
                                else:
                                    st.session_state.feedback_states[feedback_key] = None
                                    st.session_state[f"show_text_feedback_{i}"] = False
                                    st.info("싫어요 피드백이 취소되었습니다.")
                                st.experimental_rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col1:
                        if current_feedback:
                            st.markdown(f'<div class="feedback-message">{current_feedback} 피드백이 제출되었습니다. 감사합니다.</div>', unsafe_allow_html=True)
                        
                        if st.session_state.get(f"show_text_feedback_{i}", False):
                            feedback_prompt = "[좋았던 점이나 개선이 필요한 점을 자유롭게 입력해주세요.]" if current_feedback == "좋아요" else "아쉬웠던 의견을 말씀해주시면 반영하여 개선하겠습니다."
                            text_feedback = st.text_area(feedback_prompt, key=f"text_feedback_{i}")
                            if st.button("피드백 제출", key=f"submit_text_feedback_{i}"):
                                save_feedback(user_id, question, answer, current_feedback, selected_laws, st.session_state.question_type_select, text_feedback)
                                st.session_state[f"show_text_feedback_{i}"] = False
                                st.success("피드백이 제출되었습니다. 감사합니다.")
                                st.experimental_rerun()
                
                def save_feedback(user_id, question, answer, feedback_type, selected_laws, question_type, text_feedback):
                    filename = 'chatbot_feedback.csv'
                    kst = timezone(timedelta(hours=9))
                    timestamp = datetime.now(kst).strftime("%Y-%m-%d %H:%M:%S")
                    selected_laws_str = ", ".join(selected_laws)
                    new_feedback = pd.DataFrame([[user_id, timestamp, question, answer, feedback_type, selected_laws_str, question_type, text_feedback]], 
                                                columns=['User ID', 'Timestamp', 'Question', 'Answer', 'Feedback', 'Selected Laws', 'Question Type', 'Text Feedback'])
                    
                    if os.path.exists(filename):
                        df = pd.read_csv(filename)
                        df = pd.concat([df, new_feedback], ignore_index=True)
                    else:
                        df = new_feedback
                    
                    df.to_csv(filename, index=False, encoding='utf-8-sig')
                    
                def check_legal_relevance(question):
                    """
                    질문이 법률과 관련이 있는지만 확인하는 독립적인 함수
                    """
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                    
                    legal_relevance_prompt = ChatPromptTemplate.from_messages([
                        ("system", """당신은 법률 전문가입니다. 주어진 질문이 법률과 관련이 있는지 판단해야 합니다. 
                        특히 다음 법률들과의 관련성을 고려하세요: 청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 
                        정보통신공사업법, 국가계약법, 소프트웨어진흥법, 회사 내규(부패방지 행동강령 및 윤리경영원칙 실천지침).
                        
                        질문이 이 중 하나 이상의 법률과 관련이 있다면 '예'라고 답하고, 그렇지 않다면 '아니오'라고 답하세요."""),
                        ("human", "다음 질문이 법률과 관련이 있나요?: {question}")
                    ])
                    
                    legal_relevance_chain = legal_relevance_prompt | llm
                    legal_relevance_response = legal_relevance_chain.invoke({"question": question})
                    
                    # 응답에 '아니오'가 포함되어 있으면 False 반환
                    return '아니오' not in legal_relevance_response.content.lower()

                def filter_question(question):
                    """
                    질문이 법적으로 관련있는지 확인하고, 관련 법률 및 질문 유형 분류
                    """
                    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
                    
                    # 회사/기업/기관명 추출
                    company_extraction_prompt = ChatPromptTemplate.from_messages([
                        ("system", """당신은 텍스트에서 조직명(기업, 기관, 단체 등)을 추출하는 전문가입니다. 
                        주어진 질문에서 언급된 모든 회사, 기업, 기관, 단체의 이름을 추출하세요.
                        만약 질문에 기업이나 기관명이 없다면 "없음"이라고 답변하세요."""),
                        ("human", "{question}")
                    ])
                    
                    company_chain = company_extraction_prompt | llm
                    company_response = company_chain.invoke({"question": question})

                    global fcpa_retrievers
                    
                    if not fcpa_retrievers:
                        fcpa_retrievers = create_fcpa_retrievers(fcpa_vectordbs)

                    # 기업/기관명 추출 결과 처리
                    org_info = {}
                    if "없음" not in company_response.content.lower():
                        organizations = [org.strip() for org in company_response.content.split(',')]
                        
                        # 각 기관에 대한 유형 분석
                        for org in organizations:
                            if org and len(org) > 1:  # 의미 있는 기관명만 처리
                                try:
                                    # FCPA 문서 검색
                                    fcpa_docs = []
                                    if 'FCPA적용대상' in fcpa_retrievers:
                                        fcpa_docs = fcpa_retrievers['FCPA적용대상'](org)
                                    
                                    # 검색된 문서가 있으면 분석
                                    if fcpa_docs:
                                        # 모든 문서 내용 결합
                                        combined_content = "\n".join([doc.page_content for doc in fcpa_docs])
                                        
                                        # 기관 정보 추출 프롬프트
                                        analysis_prompt = ChatPromptTemplate.from_messages([
                                            ("system", """당신은 법률 문서 분석 전문가입니다. 제공된 문서 내용에서 특정 기관에 대한 다음 정보를 추출하세요:
                                            1. 기관 유형 (국가기관, 지방자치단체, 공공기관, 공직유관단체, 교육기관, 국제기구 등)
                                            2. 청탁금지법 적용 여부 (적용/미적용)
                                            3. FCPA 적용 여부 (적용/미적용)
                                            
                                            문서에 해당 정보가 명시적으로 포함되어 있는 경우에만 추출하고, 불명확한 경우 '불확실'로 표시하세요.
                                            
                                            출력 형식:
                                            기관유형: [추출된 유형 또는 '불확실']
                                            청탁금지법: [적용/미적용/불확실]
                                            FCPA: [적용/미적용/불확실]
                                            근거: [문서에서 찾은 관련 문장들]"""),
                                            ("human", f"기관명: {org}\n\n문서 내용:\n{combined_content}")
                                        ])
                                        
                                        analysis_chain = analysis_prompt | llm
                                        analysis_result = analysis_chain.invoke({})
                                        
                                        # 분석 결과 저장
                                        org_info[org] = {
                                            "analysis": analysis_result.content,
                                            "fcpa_context": combined_content,
                                            "has_fcpa_data": True
                                        }
                                    else:
                                        # FCPA 문서에서 정보를 찾지 못한 경우 일반적 분석
                                        general_analysis_prompt = ChatPromptTemplate.from_messages([
                                            ("system", """당신은 법률 전문가입니다. 조직의 이름만으로 해당 조직의 유형과 
                                            청탁금지법 및 FCPA 적용 여부를 추정해야 합니다.
                                            
                                            한국의 공공기관, 공직유관단체, 국가기관, 지방자치단체, 교육기관 등은 
                                            일반적으로 청탁금지법 적용 대상입니다.
                                            
                                            미국에 상장된 기업이나 미국과 관련된 사업을 수행하는 기업은 
                                            일반적으로 FCPA 적용 대상일 수 있습니다.
                                            
                                            답변 형식:
                                            기관유형: [유형]
                                            청탁금지법: [적용/미적용/불확실]
                                            FCPA: [적용/미적용/불확실]
                                            설명: [판단 근거]"""),
                                            ("human", f"조직명: {org}")
                                        ])
                                        
                                        general_analysis_chain = general_analysis_prompt | llm
                                        general_analysis_result = general_analysis_chain.invoke({})
                                        
                                        org_info[org] = {
                                            "analysis": general_analysis_result.content,
                                            "fcpa_context": "관련 FCPA 문서 정보 없음",
                                            "has_fcpa_data": False
                                        }
                                except Exception as e:
                                    print(f"Error analyzing organization {org}: {e}")
                                    org_info[org] = {
                                        "analysis": f"기관유형: 분석 실패\n청탁금지법: 불확실\nFCPA: 불확실\n설명: {org}에 대한 분석 중 오류가 발생했습니다: {str(e)}",
                                        "fcpa_context": "오류 발생",
                                        "has_fcpa_data": False
                                    }

                    # 질문 유형 및 관련 법률 분류
                    classification_prompt = ChatPromptTemplate.from_messages([
                        ("system", """당신은 법률 전문가입니다. 주어진 질문을 분석하여 다음 작업을 수행해야 합니다:
                        1. 질문의 유형을 다음 중 하나로 분류하세요:
                        a) 법 저촉 여부(다양한 관점 분석)
                        b) 단순 질의응답
                        c) 금액 계산
                        d) 그 외
                        2. 질문과 관련된 법률을 모두 선택하세요. 다음 법률들 중에서 선택하세요:
                        청탁금지법, 공정거래법, 중대재해처벌법, 산업안전보건법, 하도급법, 상생협력법, 
                        정보통신공사업법, 국가계약법, 소프트웨어진흥법, 회사 내규(부패방지 행동강령 및 윤리경영원칙 실천지침)

                        출력 형식:
                        질문 유형: a/b/c/d
                        관련 법률: 관련된 법률 목록, 쉼표로 구분"""),
                        ("human", "다음 질문을 분석해주세요: {question}")
                    ])
                    
                    classification_chain = classification_prompt | llm
                    classification_response = classification_chain.invoke({"question": question})
                    
                    # 응답 파싱
                    lines = classification_response.content.split('\n')
                    question_type = "d"
                    related_laws = []
                    
                    try:
                        if len(lines) >= 2:
                            question_type = lines[0].split(': ')[1].strip()
                            related_laws = [law.strip() for law in lines[1].split(': ')[1].split(',')]
                    except IndexError:
                        print(f"Unexpected response format: {classification_response.content}")
                    
                    # 질문 유형 매핑
                    type_mapping = {
                        'a': "법 저촉 여부(다양한 관점 분석)",
                        'b': "단순 질의응답",
                        'c': "금액 계산",
                        'd': "그 외"
                    }
                    question_type = type_mapping.get(question_type, "그 외")

                    if len(related_laws) == 1 and "회사 내규(부패방지 행동강령 및 윤리경영원칙 실천지침)" in related_laws:
                        return False, None, [], org_info
                    
                    # 회사 내규가 없는 경우 추가
                    if "회사 내규(부패방지 행동강령 및 윤리경영원칙 실천지침)" not in related_laws:
                        related_laws.append("회사 내규(부패방지 행동강령 및 윤리경영원칙 실천지침)")
                    
                    return True, question_type, related_laws, org_info
    
                def exact_match_search(vectordb, search_term, k=1):
                    docs = vectordb.docstore._dict.values()
                    matched_docs = []
                    
                    search_term_no_space = search_term.replace(" ", "")
                    
                    for doc in docs:
                        content_lines = doc.page_content.split('\n')
                        if content_lines:
                            first_line = content_lines[0].strip()
                            first_line_no_space = first_line.replace(" ", "")
                            
                            law_name, _, article = search_term_no_space.rpartition('제')
                            
                            if law_name in first_line_no_space and '시행령' not in first_line_no_space and '시행규칙' not in first_line_no_space:
                                article_pattern = re.escape(article).replace(r'\\d+', r'\d+')
                                if re.search(article_pattern, first_line_no_space):
                                    matched_docs.append((doc, 0))
                                elif article.split('조')[0] in first_line_no_space:
                                    matched_docs.append((doc, 1))
                            elif "부패방지행동강령" in first_line_no_space or "윤리경영원칙실천지침" in first_line_no_space:
                                if search_term_no_space in first_line_no_space:
                                    matched_docs.append((doc, 0))
                            else:
                                if search_term_no_space in first_line_no_space:
                                    matched_docs.append((doc, 2))
                    
                    return [doc for doc, _ in sorted(matched_docs, key=lambda x: x[1])[:k]]
        
    
                # 법령 이름과 조항 번호로 정렬
                def sort_key(doc):
                    content_lines = doc.page_content.split('\n')
                    if content_lines:
                        first_line = content_lines[0].strip()
                        match = re.search(law_pattern, first_line)
                        if match:
                            law_name, article = match.groups()
                            article_num = re.search(r'\d+', article)
                            return (law_name, int(article_num.group()) if article_num else 0)
                    return (doc.metadata['source'], 0)
    
                def simulate_typing(placeholder, text, speed=0.01):
                    """텍스트를 타이핑하는 것처럼 점진적으로 표시합니다."""
                    displayed_text = ""
                    for char in text:
                        displayed_text += char
                        placeholder.markdown(displayed_text + "▌")
                        time.sleep(speed)
                    placeholder.markdown(displayed_text)
          
                # LangGraph 워크플로우 정의
                workflow = StateGraph(GraphState)
                
                workflow.add_node("generate_answer", llm_answer_response)
                workflow.add_node("check_relevance", check_relevance)
                workflow.add_node("rephrase_question", rephrase_question)
                workflow.set_entry_point("generate_answer")
                workflow.add_edge("generate_answer", "check_relevance")
                workflow.add_conditional_edges(
                    "check_relevance",
                    should_continue,
                    {
                        "end": END,
                        "rephrase": "rephrase_question"
                    }
                )
                workflow.add_edge("rephrase_question", "generate_answer")
                
                app = workflow.compile()
    
                displayed_messages = set()  # 이미 표시된 메시지를 추적하기 위한 집합

                # 대화 내역 표시
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                for i, message in enumerate(st.session_state.chat_history):
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
                        if message["role"] == "assistant":
                            if "law_references" in message and message["law_references"]:
                                with st.expander("참고 법령", expanded=False):
                                    for law_ref in message["law_references"]:
                                        st.markdown(f"**출처**: {law_ref['source']}")
                                        st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;'>{law_ref['content']}</div>", unsafe_allow_html=True)
                        
                            if "similar_cases" in message and message["similar_cases"]:
                                with st.expander("유사 사례", expanded=False):
                                    for case in message["similar_cases"]:
                                        st.markdown(f"**출처**: {case['source']}, **페이지**: {case['page']}, **점수**: {case['score']:.2f}")
                                        st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;'>{case['content']}</div>", unsafe_allow_html=True)
                        
                            if i > 0 and st.session_state.chat_history[i-1]["role"] == "user":
                                display_feedback_buttons(i, st.session_state.chat_history[i-1]["content"], message["content"], selected_laws if 'selected_laws' in locals() else [])
 
                if check_password():
                    case_final_docs = []
                    # Streamlit 인터페이스
                    if user_input := st.chat_input("메세지를 입력해 주세요. "):
                        def add_message_to_history(role: str, content: str):
                            if "chat_history" not in st.session_state:
                                st.session_state.chat_history = []
                            st.session_state.chat_history.append({"role": role, "content": content})

                        # 먼저 법적 관련성만 체크
                        is_legally_relevant = check_legal_relevance(user_input)
                        
                        if not is_legally_relevant:
                            # 법적 관련성이 없으면 바로 경고 메시지 표시
                            st.chat_message("user").write(f"{user_input}")
                            st.warning("입력하신 질문은 법과 관련이 없어 답변할 수 없습니다. 법률 관련 질문을 입력해 주세요.")
                        else:
                            # 법적 관련성이 있으면 filter_question 함수를 호출하여 상세 분석
                            st.session_state.chat_history.append({"role": "user", "content": user_input})
                            is_valid, question_type, selected_laws, org_info = filter_question(user_input)
                            
                            if is_valid:
                                save_question(st.session_state.get("logged_in_user", "Unknown"), user_input, selected_laws)
                                
                                st.chat_message("user").write(f"{user_input}")
                                                            
                                with st.chat_message("assistant"):
                                    # 기관 정보가 있는 경우 표시
                                    if org_info and isinstance(org_info, dict) and len(org_info) > 0:
                                        org_info_expander = st.expander("📊 기관 유형 분석", expanded=True)
                                        with org_info_expander:
                                            for org, info in org_info.items():
                                                st.markdown(f"### {org}")
                                                
                                                # 정보 추출 및 표시
                                                analysis_text = info['analysis'] if isinstance(info, dict) and 'analysis' in info else info
                                                info_lines = analysis_text.split('\n')
                                                info_dict = {}
                                                
                                                for line in info_lines:
                                                    if ':' in line:
                                                        key, value = line.split(':', 1)
                                                        info_dict[key.strip()] = value.strip()
                                                
                                                # 표 생성
                                                if '기관유형' in info_dict:
                                                    data = {
                                                        '구분': ['기관유형', '청탁금지법', 'FCPA'],
                                                        '상태': [
                                                            info_dict.get('기관유형', '정보 없음'),
                                                            info_dict.get('청탁금지법', '정보 없음'),
                                                            info_dict.get('FCPA', '정보 없음')
                                                        ]
                                                    }
                                                    st.table(pd.DataFrame(data))
                                                
                                                # 설명 표시
                                                if '설명' in info_dict:
                                                    st.markdown(f"**설명**: {info_dict['설명']}")
                                                st.markdown("---")
                                    response_placeholder = st.empty()
                                    simulate_typing(response_placeholder, "답변을 생성하고 있습니다. 잠시만 기다려주세요...")
                                    
                                    # 리트리버 및 벡터 데이터베이스 생성
                                    create_retrievers(selected_laws)
                                    create_for_show_law_vectordbs(selected_laws)
                                    
                                    # org_info 전달 추가
                                    law_answers = asyncio.run(process_user_input_parallel(user_input, selected_laws, question_type, org_info))

                                    # 최종 답변 생성 - org_info 전달
                                    final_answer = generate_final_answer(user_input, law_answers, org_info)
                                    
                                    add_message_to_history("assistant", final_answer["response"])
                                    simulate_typing(response_placeholder, final_answer["response"])
                                    display_feedback_buttons(len(st.session_state.chat_history), user_input, final_answer["response"], selected_laws)

                                    # 참고 법령 표시
                                    with st.expander("참고 법령", expanded=False):
                                        all_law_docs = []
                                        first_law_doc = None
                                        mentioned_laws = final_answer["mentioned_laws"]
                                        law_pattern = r'((?:\w+법|(?:\w+기준에 관한 규칙)|부패방지 행동강령|윤리경영원칙 실천지침)(?:\s+시행(?:령|규칙))?)\s+((?:제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?(?:\s*[가-힣]목)?)|(?:별표\s*\d+)|(?:\d+-\d+))'
                                        query_embedding = embeddings.embed_query(user_input)
                                        
                                        for i, mentioned_law in enumerate(mentioned_laws):
                                            if isinstance(mentioned_law, tuple):
                                                full_law_name, law_reference = mentioned_law
                                            else:
                                                match = re.search(law_pattern, mentioned_law)
                                                if match:
                                                    full_law_name, law_reference = match.groups()
                                                else:
                                                    continue
                                            
                                            law_name = full_law_name.split()[0] if "시행령" in full_law_name or "시행규칙" in full_law_name else full_law_name
                                            
                                            if law_name in selected_laws or "부패방지 행동강령" in law_name or "윤리경영원칙 실천지침" in law_name:
                                                search_term = f"{full_law_name} {law_reference}"
                                                law_docs = exact_match_search(selected_for_show_law_vectordbs[law_name], search_term, k=1)
                                                if not law_docs:
                                                    broader_search_term = f"{full_law_name} {law_reference.split()[0]}"
                                                    law_docs = exact_match_search(selected_for_show_law_vectordbs[law_name], broader_search_term, k=1)
                                                
                                                if i == 0 and law_docs:
                                                    first_law_doc = law_docs[0]
                                                else:
                                                    all_law_docs.extend(law_docs)
        
                                        # 중복 제거 (원래 순서 유지)
                                        unique_law_docs = []
                                        seen = set()
                                        if first_law_doc:
                                            unique_law_docs.append(first_law_doc)
                                            seen.add((first_law_doc.page_content, first_law_doc.metadata['source'], first_law_doc.metadata['page']))
                                        
                                        for doc in all_law_docs:
                                            key = (doc.page_content, doc.metadata['source'], doc.metadata['page'])
                                            if key not in seen:
                                                seen.add(key)
                                                unique_law_docs.append(doc)
        
                                        unique_law_docs.sort(key=sort_key)
                                        
                                        law_refs = []
                                        for law_doc in unique_law_docs:
                                            law_name = law_doc.metadata['source']
                                            st.markdown(f"**출처**: {law_name}, **페이지**: {law_doc.metadata['page']}")
                                            content = law_doc.page_content.replace("[[", "\n<hr>")
                                            
                                            # 문서의 유사도 점수 계산
                                            doc_embedding = embeddings.embed_query(content)
                                            doc_similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                                            highlight_threshold = max(0, doc_similarity - 0.001)  # 최소값을 0으로 설정
                                            
                                            lines = content.split('\n')
                                            if lines:
                                                first_line = lines[0].strip()
                                                match = re.match(law_pattern, first_line)
                                                if match:
                                                    law_name, article = match.groups()
                                                    formatted_first_line = f"{law_name} {article.replace(' ', '')}"
                                                    formatted_content = f"<strong>{formatted_first_line}</strong><br><br>" + highlight_text('\n'.join(lines[1:]), query_embedding, embeddings, highlight_threshold)
                                                else:
                                                    formatted_content = highlight_text(content, query_embedding, embeddings, highlight_threshold)
                                            else:
                                                formatted_content = highlight_text(content, query_embedding, embeddings, highlight_threshold)
                                    
                                            st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;'>{formatted_content}</div>", unsafe_allow_html=True)                                        
                                            law_refs.append({
                                                'source': law_name,
                                                'page': law_doc.metadata['page'],
                                                'content': content
                                            })
                                    
                                        final_answer["law_references"] = law_refs
        
                                    # 유사 사례 처리 및 저장
                                    with st.expander("유사 사례", expanded=False):
                                        combined_query = " ".join([m.content for m in memory.chat_memory.messages])
                                        preprocessed_query = preprocess_text(combined_query)
                                        
                                        if not isinstance(preprocessed_query, str):
                                            preprocessed_query = str(preprocessed_query)

                                        similar_cases = {}

                                        for law in selected_laws:
                                            if any(ans['law_name'] in ans['response'] and "답변할 수 없습니다" not in ans['response'] for ans in final_answer["law_answers"] if ans['law_name'] == law):
                                                case_retriever = case_retrievers[law]
                                                if callable(case_retriever):
                                                    case_docs = case_retriever(preprocessed_query)
                                                    
                                                    if case_docs:
                                                        doc_embeddings = embeddings.embed_documents([doc.page_content for doc in case_docs])
                                                        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
                                                        
                                                        top_cases = sorted(zip(case_docs, similarities), key=lambda x: x[1], reverse=True)
                                                        
                                                        similar_cases[law] = []
                                                        for case_doc, case_score in top_cases:
                                                            content = case_doc.page_content.replace("[[", "\n<hr>")
                                                            similar_cases[law].append({
                                                                'source': law,
                                                                'page': case_doc.metadata.get('page', 'N/A'),
                                                                'score': case_score,
                                                                'content': content
                                                            })
                                                            if len(similar_cases[law]) >= 3:  # 각 법률당 최대 3개의 사례만 저장
                                                                break

                                        # 유사 사례를 전역 변수에 저장
                                        similar_cases_db[user_input] = similar_cases

                                    # 유사 사례 표시
                                    with st.expander("유사 사례", expanded=False):
                                        has_valid_cases = False  # 유효한 사례가 있는지 확인하는 플래그
                                        for law, cases in final_answer.get("similar_cases", {}).items():
                                            valid_cases = [case for case in cases if case['content'].strip() != "빈 문서"]
                                            if valid_cases:  # 유효한 사례가 있는 경우에만 표시
                                                has_valid_cases = True
                                                st.subheader(f"{law} 관련 유사 사례")
                                                for case in valid_cases:
                                                    st.markdown(f"**출처**: {case['source']}, **페이지**: {case['page']}, **점수**: {case['score']:.2f}")
                                                    content = case['content'].replace("[[", "\n<hr>")
                                                    st.markdown(f"<div style='padding: 10px; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 10px; white-space: pre-wrap;'>{content}</div>", unsafe_allow_html=True)
                                        
                                        if not has_valid_cases:
                                            st.write("관련된 유사 사례가 없습니다.")
                                            
                                    # AI 응답 추가 (참고법령과 유사사례 포함)
                                    ai_message = {
                                        "role": "assistant", 
                                        "content": final_answer["response"],
                                        "law_references": final_answer.get("law_references", []),
                                        "similar_cases": final_answer.get("similar_cases", [])
                                    }
                                    user_state["messages"].append(ai_message)
                                    st.session_state.messages.append(ai_message)
                                    
                                    memory.chat_memory.add_ai_message(final_answer["response"])
                                        
                                    st.write("---")
                                    st.write("아래는 가장 최근 질문에 대한 선택하신 법령별 관련성 검토 결과입니다.")
                                    # 법령별 검토 결과를 가장 마지막에 표시
                                    for law_answer in final_answer["law_answers"]:
                                        with st.expander(f"{law_answer['law_name']} 검토 결과", expanded=False):
                                            if "답변할 수 없습니다" in law_answer['response']:
                                                st.write(f"이 질문은 {law_answer['law_name']}과 관련성이 낮은 것으로 판단되어 답변할 수 없습니다.")
                                            else:
                                                st.write(law_answer["response"])
                                                if law_answer['relevance'] == "grounded":
                                                    st.write(f"관련성 검사: 🟢 (높음) (재생성 횟수: {law_answer['attempts']}번)")
                                                elif law_answer['relevance'] == "notSure":
                                                    st.write(f"관련성 검사: 🟡 (중간) (재생성 횟수: {law_answer['attempts']}번)")
                                                elif law_answer['relevance'] == "notGrounded":
                                                    st.write(f"관련성 검사: 🔴 (낮음) (재생성 횟수: {law_answer['attempts']}번)")
                                                elif law_answer['relevance'] == "Pass":
                                                    st.write(f"")
                                                else:
                                                    st.write(f"관련성 검사: 문서와의 연관성이 부족한 답변입니다. 참고하시기 바랍니다. (재생성 횟수: {law_answer['attempts']}번)")
                            else:
                                st.warning("입력하신 질문은 법과 관련이 없어 답변할 수 없습니다. 법률 관련 질문을 입력해 주세요.")
                    if "feedback_message" in st.session_state:
                        st.success(st.session_state["feedback_message"])
                        del st.session_state["feedback_message"]
    else:
        st.stop()   
        
if __name__ == "__main__":
    main()

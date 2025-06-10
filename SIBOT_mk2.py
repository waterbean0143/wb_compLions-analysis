# ─────────────────────────────────────────────────────
# 0) 라이브러리 및 선언 영역
# ─────────────────────────────────────────────────────
import streamlit as st
import requests
import tempfile
import os
import re
from datetime import datetime, timezone, timedelta
from typing import TypedDict, Dict, List, Tuple
from io import BytesIO
import uuid
import time
import threading
import asyncio
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kiwipiepy import Kiwi
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
)
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from langchain.chains import LLMChain

from collections import defaultdict
from langchain.callbacks import LangChainTracer
from langsmith import traceable

# ─────────────────────────────────────────────────────
# 0-1) PDF 다운로드 및 인덱스 추출
# ─────────────────────────────────────────────────────
def download_and_load(url: str) -> List[Document]:
    resp = requests.get(url)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(resp.content)
        tmp_path = tf.name
    try:
        docs = PyMuPDFLoader(tmp_path).load()
    except Exception:
        docs = []
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return docs

def extract_index_chunks(url: str) -> List[Document]:
    raw = download_and_load(url)
    if not raw: return []
    lines = raw[0].page_content.splitlines()
    start = next((i+1 for i,l in enumerate(lines) if l.strip().startswith("##")), 0)
    pattern = re.compile(r"^(\d+)\.\s*(.+)$")
    idxs: List[Document] = []
    for line in lines[start:]:
        m = pattern.match(line.strip())
        if not m: break
        num, title = m.groups()
        idxs.append(Document(page_content=f"{num}. {title}", metadata={"step":int(num),"title":title}))
    return idxs

# ─────────────────────────────────────────────────────
# 0-2) 질문 유형 분류 및 Persona
# ─────────────────────────────────────────────────────
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

class GraphState(TypedDict):
    question: str; step_name: str; sub_title: str
    question_type: str; context: str; response: str; attempts: int

INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """당신은 질문 의도 분류기입니다. 아래 6가지 유형 중 하나로 분류하세요.
- 정의 요청, 수행 절차 안내, 산출물·문서 요구 사항, 책임·역할 분담, 일정·마일스톤 확인, 일반 질문
질문: “{question}”
출력: 질문유형: <위 6가지 중 하나>"""
    )
])

def classify_with_llm(question: str) -> str:
    out = LLMChain(llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                   prompt=INTENT_CLASSIFICATION_PROMPT).predict(question=question)
    return out.split(":")[-1].strip()

QUESTION_TYPES = [
    "자유 질의", "정의 요청", "수행 절차 안내",
    "산출물·문서 요구 사항", "책임·역할 분담", "일정·마일스톤 확인"
]

def generate_prompt_by_phase_and_type(phase: str, qtype: str) -> str:
    # base_prompt, phase_prompts, question_type_prompts는 별도 모듈로 분리 가능
    base_prompt = """당신은 대기업이자 사기업인 KT의 이행 절차 전문 PM입니다. 질문자는 기본적으로 KT 직원으로, 공직자가 아닌 민간 기업의 직원입니다. KT는 정부 기관이 아니며, 직원들은 공무원이 아닙니다."""

    phase_prompts = {
        "제안/계약": """
당신은 대기업 KT의 제안/계약 전문 PM입니다. 제안/계약 단계는 프로젝트 수주를 위한 전단계와 계약 체결까지의 단계인 걸 명심하세요. KT 직원들이 자사, 협력사, 고객사와 함께 업무를 수행할 일정과 비용 그리고 영향 등을 고려해야 합니다. 이 절차의 주요 목적은 PM/PL/PMO 등의 공정한 직무수행과 사내 절차의 신뢰성 제고에 중점을 두고, KT 직원들이 고객사와 수행사의 관계에서 주의해야 할 점을 답변해주세요.

답변을 하기 전에 다음 사항들을 확인하세요:
1. 질문이 제안 전(VDC-A), 입찰 전(VDC-B), 계약 전(VDC-C) 중 어느 단계인지 파악했나요?
2. 질문에 등장하는 인물들의 역할(R&R)은 명확히 정의되었나요? (예: 영업대표, 제안PM, 이행PM 등)
3. 질문의 배경 상황이 VDC 관련 절차인지, 제안서 작성인지, 산출물 관련인지 구체적으로 언급되었나요?
4. 관련 시스템(KOS 등)과 연결된 요청사항인지 확인하셨나요?
5. 질문자가 필요로 하는 대상 자료나 문서의 종류는 명확히 언급되어 있나요? (예: 리스크 등급표, 심의자료 등)
6. QNA 또는 프로세스 문서에서 유사한 질문이 있는지 확인해보았나요?

이 중 하나라도 불명확한 점이 있다면, 답변 전에 다음과 같이 추가 정보를 요청하세요. 예를 들어:
- "질문하신 내용이 제안/계약 단계 중 어느 시점을 의미하는지 명확히 해주실 수 있나요? (예: VDC-A 단계인지, 계약 직전 단계인지)"
- "VDC 발의서나 심의자료가 필요한 상황인지, 아니면 단순 참고를 원하시는 건지 알려주시면 더 정확한 답변이 가능합니다."
- "KOS 시스템 상의 어떤 기능을 활용하고자 하시는 건가요? 예: 공조 요청, 영업기회 ID 등록 등"
- "이와 유사한 사전공고 또는 심의사례가 있다면 함께 참고하여 더 구체적인 안내가 가능합니다."

이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다.  
특히 단계별 프로세스 안내나 QnA 문서에 유사한 내역이 있다면, 그 내용을 중요하게 고려하여 답변해주세요.
""",
        "착수/계획": """당신은 대기업 KT의 착수/계획 단계 전문 PM입니다. 이 단계는 프로젝트 계약 이후 실행을 위한 기반을 구축하는 시점으로, 시스템 등록, 하도급 계약, 사무공간 확보, 품질지원 요청, 산출물 관리 등의 절차를 포함합니다. 이행 PM과 협력사, 고객사, 내부 조직 간 협업이 핵심인 만큼, 일정 준수와 사전 등록, 내부 승인 프로세스에 유의해야 합니다. PM/PL/PMO 등의 공정한 직무 수행과 사내 절차의 신뢰성 확보에 중점을 두고, KT 직원들이 착수/계획 단계에서 주의해야 할 점을 설명해주세요.

답변을 하기 전에 다음 사항들을 확인하세요:
1. 질문의 주체가 이행 PM인지, 영업대표인지, 협력사인지 명확히 확인하셨나요?
2. WBS 생성, 착수 PRB 등록, EasyERP 입력 등 시스템 관련 절차가 언급되어 있나요?
3. 하도급 계약 또는 수행사무실 확보 관련 이슈가 포함되어 있나요?
4. 품질 지원 요청 또는 전사 QA 협조 여부가 언급되어 있나요?
5. 법인공인인증서, 사용인감 날인, 공정률 확인서 등 사내 시스템/서류 처리와 관련된 내용인가요?
6. 질문 상황에서 '계약 이후', '착수 전', '초기 단계'라는 시점이 명확하게 식별되었나요?
7. 산출물 템플릿, 기술자료, 첨부메일 등 문서 관련 처리 조건이 언급되었나요?

이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
- "질문자께서 말씀하신 하도급 계약은 최초 체결인지, 추가 변경 건인지 명확히 알려주실 수 있나요?"
- "EasyERP 입력 주체가 영업대표인지 이행 PM인지 다시 확인해주실 수 있을까요?"
- "품질 지원을 말씀하셨는데, 해당 사업의 규모와 AICT 여부를 알 수 있을까요? 전사 QA 우선순위에 따라 지원 여부가 달라질 수 있습니다."
- "해당 PRB 등록일은 계약일로부터 며칠 경과된 시점인지 확인 부탁드립니다."

이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 <제공된 프로세스 문서>의 절차 흐름이나 <대표질문 문서>에 유사한 QnA가 존재하는 경우, 그 내용을 근거로 활용하여 답변해주세요.
""",
        "실행/통제": """당신은 꼼꼼하게 일정과 품질을 관리하고, 프로젝트 리스크에 선제적으로 대응하는 대기업 KT의 '실행/통제' 단계 전문 프로젝트 관리자(PM)입니다. 이 단계는 프로젝트 계획에 따라 실행하고, 일정을 통제하며, 품질·위험·성과 등을 관리하는 실질적인 수행 단계임을 명심하세요.  
PM, PL, PMO는 고객사와의 계약 조건, 변경요청, 외부감리 대응, PRB 발의 및 품질점검 등 다양한 실무 이슈에 직면하게 됩니다. KT 직원들이 프로젝트 실행 중 적절한 문서와 판단 기준을 바탕으로 안정적으로 사업을 수행할 수 있도록 안내해야 합니다.

---

💡 답변을 하기 전에 다음 사항들을 확인하세요:
1. 질문이 일정 관리, 품질관리, 요구사항 관리, 리스크 통제, 계약 변경 등 어느 카테고리에 속하는지 분류하셨나요?
2. 관련된 산출물(예: 요구사항 추적표, 검사기준서, 진척률 보고서 등)이 명확히 언급되었나요?
3. 문서 작성/관리의 책임 주체(PM/PL/본부/고객사 등)가 명확한가요?
4. 외부감리 또는 PRB와 같은 대외 절차에 대한 언급이 있는 경우, 준비 또는 승인을 위한 내부 프로세스가 충분히 설명되었나요?
5. 질문자가 요청하는 조치의 시점(예: 설계 단계 후, 이행 중, 종료 직전 등)이 명확하게 제공되었나요?
6. 유사한 문서, 예시 양식, 과거 작성 기준이 있는지 확인해보셨나요?

---

❗ 이 중 하나라도 불명확한 점이 있다면, 답변 전에 다음과 같은 추가 정보를 요청하세요:
- "요구사항 추적표와 관련된 시스템이나 사업 유형을 알려주실 수 있나요? 작성 시점이 프로젝트 성격에 따라 다를 수 있습니다."
- "검사기준서를 어느 단계에서 작성하려는 것인지 명확히 알려주세요. 설계 말기와 개발 중반이 기준입니다."
- "이행경비의 전용 계정이 사전에 지정되어 있는지, 아니면 변경 절차가 필요한지 알려주실 수 있나요?"
- "진행 PRB는 어떤 사유로 발의하는 것인지요? 자금 전용, 일정 변경, 리소스 증원 등 사유에 따라 양식과 흐름이 다를 수 있습니다."
- "외부감리 대상 프로젝트인지 먼저 확인해주시겠어요? 내부 QA 점검 절차가 달라집니다."

---

🧩 이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다. 특히 문서 작성 시점, 관련 기준, 사전 승인 절차 등은 프로세스 문서와 대표 QNA를 참고해 단계별로 안내해 주세요.
""",
        "종료/사후관리": """당신은 대기업 KT의 종료/사후관리 전문 PM입니다. 이 단계는 프로젝트의 공식 종료 처리, 고객 인계, 산출물 정리, 하자보수 및 사후 이슈 대응까지 포함된 단계임을 명심하세요. KT 직원들이 내부 프로세스를 완결 짓고 고객과의 관계를 긍정적으로 마무리하는 것이 핵심입니다. PM/PL/PMO로서의 책임 있는 절차 준수와 문서화를 중심으로 답변해주세요.

📌 답변을 하기 전에 다음 사항들을 확인하세요:
1. 질문의 내용이 "종료단계", "사후관리", "인수인계", "하자보수", "PRB", "잔금" 등 어떤 세부 프로세스를 지칭하는가요?
2. 종료 PRB 또는 사후 PRB 관련 질문인가요? → 해당 시점과 시스템 처리 여부가 명확한가요?
3. 고객 검수, 인수인계, 종료보고회와 관련된 문서/공식 승인이 언급되었나요?
4. 산출물 제출 및 반출 요청과 관련된 고객 협의 또는 공문 처리 내용이 있나요?
5. 하자보수 계획, 무상/유상 범위가 명확히 구분되어 언급되었나요?
6. 이슈 발생 후 추가 비용 처리(사후PRB 등) 관련 문의인가요?
7. 종료 후 안정화 기간과 대응 조직(이행팀 vs 하자보수팀) 구분이 필요한가요?

❗ 이 중 하나라도 불명확한 점이 있다면, 답변 전에 추가 정보를 요청하세요. 예를 들어:
- "말씀하신 PRB는 종료 PRB인지, 사후 PRB인지 명확히 말씀해주실 수 있나요?"
- "질문하신 비용 집행이 프로젝트 종료 이전인지 이후인지, 고객 검수는 마무리되었는지 알려주세요."
- "하자보수 대상 시스템/서비스 명과 범위를 조금 더 구체적으로 설명해주실 수 있나요?"
- "말씀하신 잔여 예산은 내부 정산 대상인지, 외부 수금/지급 관련 예산인지 구분이 필요합니다."

💡 이러한 추가 정보를 바탕으로 더 정확하고 구체적인 답변을 제공할 수 있습니다.  
특히 [종료 PRB 등록 처리], [사후 PRB 승인 후 비용 집행], [고객 검수와 종료보고회 관계], [무상하자보수 및 정기점검], [EasyERP 시스템 처리 흐름] 등은 `프로세스_종료+사후관리.pdf`, `QNA_종료+사후관리.pdf` 문서를 기반으로 판단하세요.  
"""
    }

    question_type_prompts = {
        "정의 요청": """
[Task 1: 단계별 지침]
    1. 질문에서 언급된 용어나 절차 항목을 식별하고, 그 정의 또는 구조를 명확히 설명해야 합니다.
    2. 해당 항목이 속한 절차 단계(예: 제안/계약, 착수/계획 등)를 명확히 지정하세요.
    3. <제공된 문서>에서 용어 정의, 절차 순서, 하위 단계 구조 등을 찾아 근거를 제시하세요.
    4. 사용자 질문이 일반 용어에 대한 정의인지, 특정 단계 또는 산출물에 대한 설명인지 구분하여 응답 방식을 조정하세요.
    5. 정보가 부족한 경우, 질문자가 명확하게 보완할 수 있도록 추가 정보를 요청하는 질문을 제시하세요.

    [Task 2: 출력 형식]
    응답은 다음 주요 부분으로 구성되어야 합니다:
    1. 정의 설명: '[정의 설명]' 단락으로 시작하며, 질문의 핵심 용어나 절차 항목을 설명합니다.
    2. 절차 구조: '[절차 구조]' 단락으로 시작하며, 해당 항목이 속하는 전체 절차 흐름, 위치, 관계된 산출물 등을 설명합니다.
    3. 관련 문서 근거: '[문서 근거]' 단락으로 시작하며, <제공된 문서>의 어떤 부분을 근거로 했는지 기술합니다.
    4. 권고사항: '[권고사항]' 단락으로 시작하며, 업무 적용 시 주의사항이나 참고할 문서를 제시합니다.
    
    [Task 3: 품질 보증]
    응답이 다음을 보장하도록 합니다:
    1. 용어 또는 절차 구조를 명확하게 정의합니다.
    2. <제공된 문서>에서 인용 또는 근거를 명시적으로 밝힙니다.
    3. 정의가 문맥에 맞고, 프로젝트 내에서 실제 활용 가능한 형태로 전달되도록 합니다.

    [Reflection]
    응답이 실제 SI 업무 현장에서 참조할 수 있는 문서 기반 정의 및 구조 설명으로 적절한지 확인합니다. 정의가 너무 단순하거나 문맥과 어긋나지 않는지 검토합니다.

    [Feedback]
    응답이 업무에 도움이 되었는지, 해당 정의가 충분히 설명되었는지 사용자의 피드백을 요청합니다.

    [Constraints]
    1. 응답은 <제공된 문서>에만 기반해야 합니다.
    2. 명확한 정의나 구조가 <제공된 문서>에 없는 경우, 이를 언급하고 '현재 문서 내에는 직접적인 정의가 확인되지 않습니다.'라고 명시해야 합니다.
    3. 질문에 포함된 내용 이외에 새로운 용어 정의나 절차를 임의 생성하지 않습니다.
    4. 결론은 간결하고 명확해야 하며, 필요시 사용자에게 추가 정보 요청을 안내해야 합니다.

    [Context]
    사용자는 대기업 KT의 프로젝트 관리자이며, 프로젝트 수행 단계에서 문서 기반의 정의나 절차적 설명을 필요로 하고 있습니다. 예를 들어 PRB, WBS, 검토위원회 등 용어나 '사전공고 분석' 등의 절차 항목에 대한 구조 이해를 요청하는 상황일 수 있습니다.
    """,
        "수행 절차 안내": """
[Task 1: 단계별 지침]
1. 질문에 언급된 업무 활동이나 상황을 정확히 파악하세요.
2. 해당 활동이 어떤 절차 단계(예: 제안/계약, 착수/계획 등)에 속하는지 매핑하세요.
3. 각 절차 단계에서의 주체(예: PM, PL, 고객사 등)와 책임(R&R)을 파악하세요.
4. 관련 문서나 산출물이 요구되는 경우 명시하세요.
5. 해당 절차의 선행 조건이나 후속 활동이 있다면 연결 관계를 설명하세요.
6. 유사한 QnA가 존재할 경우, 요약해 참고 내용으로 포함하세요.

[Task 2: 출력 형식]
응답은 다음 주요 부분으로 구성되어야 합니다:
1. 절차 개요: '[절차 개요]'로 시작하며, 질문과 관련된 절차의 목적과 흐름을 간단히 요약합니다.
2. 세부 수행 단계: '[세부 수행 단계]'로 시작하여 절차를 구체적으로 나열합니다. 각 단계는 "① 활동명 - 담당자 - 필요 산출물 - 관련 시스템" 형식으로 정리합니다.
3. 관련 산출물/시스템: '[관련 산출물 및 시스템]' 단락을 별도로 만들어 필요한 문서나 시스템을 명확히 나열합니다.
4. 유의사항: '[유의사항]'으로 시작하여 리스크 또는 실무상 유의점, 고객사 커뮤니케이션 주의사항 등을 안내합니다.

[Task 3: 품질 보증]
응답이 다음을 보장하도록 합니다:
1. 질문된 활동이 실제 SI 방법론 내 절차 흐름과 정확히 매핑되는지 확인합니다.
2. 실무자가 이해할 수 있는 문장 구성 및 업무 시스템을 고려한 언급을 포함합니다.
3. 관련 문서(프로세스 PDF, QNA 문서, 용어집)에서의 참조를 우선 적용합니다.

[Reflection]
각 응답이 절차 기준에 따라 정확히 안내되고, 문서에서 제공된 정보를 기반으로 하는지 확인하세요. 질문된 활동이 포함되는 전체 절차 흐름에서 어떤 위치인지 판단하고 종합적으로 설명되었는지 확인합니다.

[Feedback]
응답이 실제 업무에 도움이 되었는지, 절차 흐름이 명확하게 설명되었는지를 사용자에게 확인 요청합니다.

[Constraints]
1. 응답은 <제공된 문서>에만 기반해야 합니다.
2. 질문자가 특정 역할(PM, PL, 수행사 등)에 대해 명시하지 않은 경우, 업무 흐름상 추정 가능하면 명확히 밝히고 아니면 추가 정보 요청 문장을 추가해야 합니다.
3. 사용자 질문이 흐름의 중간 단계를 가리키는 경우, 앞뒤 단계 간 연결성을 보장합니다.
4. 명확하지 않은 절차나 문서가 포함된 경우, "<문서 참조 필요>" 또는 "<추가 확인 필요>"로 표시하여 판단 유보해야 합니다.
5. 질문을 새로 생성하면 안 됩니다.

[Context]
사용자는 대기업 KT의 프로젝트 관리자로, SI 사업 수행 전반에 걸쳐 절차적 흐름과 산출물 요구 사항, 관련 시스템 정보에 대한 정확한 안내를 필요로 합니다. 각 단계별 세부활동과 요구 문서를 통합적으로 안내하는 것이 중요합니다.
""",
        "산출물·문서 요구 사항": """
[Task 1: 단계별 지침]
1. 사용자의 질문에서 언급된 절차 단계 및 활동을 식별합니다.
2. 관련된 산출물 유형(예: 계획서, 회의록, 수지분석표, 요구사항 명세서 등)을 <제공된 문서>에서 추출하여 확인합니다.
3. 해당 산출물의 작성 주체(PM, PL, BD 등)와 작성 시점(착수 전, 수행 중, 종료 후 등)을 파악합니다.
4. 산출물의 제출 주체(내부 승인용/고객 제출용)를 구분합니다.
5. 형상관리(버전관리/기준일자 통제 등) 여부를 체크합니다.
6. 관련 문서의 제출 형식(예: 양식, 주요 항목, 서명 필요 여부 등)을 확인합니다.
7. <제공된 문서>에서 산출물 관련 책임 분담 구조가 명시된 경우, 그에 따라 각 주체의 책임 범위를 식별합니다.

[Task 2: 출력 형식]
응답은 다음 주요 부분으로 구성되어야 합니다:
1. [산출물 요약]: 질문에서 요청된 항목이 포함된 절차의 산출물 개요와 정의
2. [작성 및 제출 조건]: 산출물의 작성 시점, 작성자, 제출 주체, 제출 기한 등
3. [형상관리 여부]: 버전 관리, 기준일 통제, 변경 이력 요구 여부 등
4. [주의사항]: 해당 산출물의 누락/지연/불일치가 가져올 리스크 또는 내부/외부 감사 시 문제 소지
5. [예시 문서 명]: 문서 내 존재하는 유사 문서 양식 예시 (있는 경우에 한함)

[Task 3: 품질 보증]
응답이 다음을 보장하도록 합니다:
1. 산출물의 정의 및 사용 목적이 <제공된 문서> 기준과 일치합니다.
2. 작성 주체, 제출 주체, 활용 목적이 혼동 없이 구분되어 설명됩니다.
3. 산출물의 사내 문서 관리 기준 (형상관리, 버전 명시 등)을 따릅니다.

[Reflection]
각 응답이 절차에 따른 정확한 문서 작성 기준과 책임 구분을 충실히 반영하는지 확인합니다. 특히 수행 중 제출되는 문서의 승인 흐름이나 외부 제출 여부에 대해 명확히 설명되었는지 고려합니다.

[Feedback]
응답의 명확성과 실무 적용 가능성에 대한 피드백을 요청합니다. 제시된 문서 양식이나 절차 흐름이 현업 수행에 실질적으로 도움이 되었는지 확인합니다.

[Constraints]
1. 응답은 <제공된 문서>에만 기반해야 합니다.
2. 표준 양식이 명시되지 않았거나 문서 정의가 모호한 경우, 명확하게 '문서 양식이 문서에 명시되어 있지 않음'을 표기해야 합니다.
3. 질문을 새로 생성하거나 불명확한 가정을 바탕으로 문서를 임의 정의해서는 안 됩니다.

[Context]
사용자는 KT의 프로젝트 실행 조직 소속 실무자 또는 관리자로서, 특정 절차에 필요한 문서를 어떤 기준으로 작성하고 제출해야 하는지를 묻고 있습니다. 산출물의 정확한 정의와 작성 조건, 책임 주체에 대한 명확한 안내가 필요합니다.
""",
        "책임·역할 분담": """
[Task 1: 단계별 지침]
1. 질문 내에서 언급된 역할(R&R) 또는 조직/직무(PM, PL, PMO 등)를 식별합니다.
2. 질문자가 속한 역할이 명확하지 않은 경우, "사용자의 역할이 무엇인지 명확히 해주실 수 있나요?"와 같은 추가 정보 요청 문장을 제안합니다.
3. 프로젝트 단계에 따라 책임 분담이 어떻게 정의되는지 <제공된 문서> 기준으로 분석합니다.
4. 의사결정 권한(VDC, PRB 등)과 실행 권한(PM, PL, QA 등) 간 차이를 구분하여 설명합니다.
5. 역할 간 충돌 가능성(예: PM vs PRB 결정, QA vs 개발팀 등)이 있을 경우, 우선순위 판단 기준과 그에 따른 업무 분담 예시를 제시합니다.
6. 역할별 보고 체계나 승인 체계(예: PRB 승인 필요 여부 등)를 명확히 안내합니다.

[Task 2: 출력 형식]
응답은 다음 주요 부분으로 구성되어야 합니다:
1. [식별된 역할]로 시작하여, 질문 내에 언급된 책임 주체와 그 역할을 정리합니다.
2. [책임 범위]로 시작하여, 각 주체의 책임과 권한이 <제공된 문서>에 따라 어떻게 분담되는지 설명합니다.
3. [조정 및 협의 필요 영역]으로 시작하여, 역할 간 경계 모호성 또는 분쟁 발생 가능성이 있는 영역을 식별하고 조정 권한자(예: PRB, CCB 등)를 명시합니다.
4. [결론 및 권고사항]으로 마무리하며, 질문자가 해당 상황에서 어떻게 판단하거나 조치할지 방향성을 제시합니다.

[Task 3: 품질 보증]
응답이 다음을 보장하도록 합니다:
1. 제공된 문서의 절차, 용어정의, 책임주체 구분에 기반하여 작성됩니다.
2. R&R이 명확히 식별되지 않을 경우, 사용자에게 명확한 추가 정보 요청 질문을 포함합니다.
3. 시스템/절차적으로 결정 권한이 있는 주체와 실행 주체가 구분되었는지 확인합니다.

[Reflection]
각 응답이 KT 프로젝트 관리 체계(PM/PL/PMO/VDC 등)의 구조를 반영하고, 내부 승인 체계 및 책임 소재를 명확히 구분하였는지 검토합니다. 다수 역할이 얽힌 경우, 책임-보고-결정 체계를 계층적으로 설명했는지도 확인합니다.

[Feedback]
응답의 명확성과 구조화에 대해 사용자에게 피드백을 요청합니다. "이 설명이 역할 간 책임 구분을 이해하는 데 도움이 되었나요?", "PMO와 PRB의 차이를 충분히 설명했나요?"와 같이 구체적인 항목을 제안합니다.

[Constraints]
1. 응답은 <제공된 문서>에만 기반해야 합니다.
2. 책임소재가 불분명하거나 문서 상 명시되지 않은 경우, '해당 문서에서는 명확히 구분되지 않습니다'라고 표시해야 하며, 가능한 권한자에게 질의할 것을 권장합니다.
3. 질문에 없는 추가 시나리오를 생성하지 않습니다.
4. 역할 간 충돌이 있는 경우, 해당 조정 체계(CCB, PRB 등)가 문서에 있는지 먼저 확인하고 없을 경우 경고해야 합니다.

[Context]
사용자는 KT의 PM 또는 PL이며, 프로젝트 내에서 업무 분장, 책임 범위, 협의 체계에 대해 실무적으로 검토 중일 가능성이 큽니다.
""",
        "일정·마일스톤 확인": """
[Task 1: 단계별 지침]
1. 질문에서 특정된 절차 단계(예: 제안/계약, 착수/계획 등)와 그에 연관된 주요 일정/마일스톤 여부를 식별합니다.
2. <제공된 문서>에서 해당 단계의 수행 주체, 요구 일정, 필수 산출물, 연계 의사결정(예: PRB, VDC) 등을 검토합니다.
3. 전체 프로젝트 일정 중 현재 질문과 관련된 세부 일정 또는 마일스톤의 위치를 파악합니다.
4. 예상 일정 지연, 병목 구간, 완료 조건 미충족 등 리스크 요인을 식별합니다.
5. 마일스톤이 내부 의사결정과 외부 승인(고객, 협력사 등)에 의해 종속되는 경우 이를 분리하여 설명합니다.
6. 문서 내 기준 일정과 사용자 질문의 현재 상태 간 차이가 있는 경우, 명확히 비교하고 그 영향도 분석을 포함합니다.
7. 필요한 경우 사용자에게 다음과 같은 추가 정보를 요청합니다:
    - "해당 단계의 시작일과 완료일을 알려주실 수 있나요?"
    - "이 일정 지연이 고객 승인 지연인지, 내부 산출물 준비 부족인지 명확히 해주실 수 있나요?"
    - "VDC 또는 PRB 일정과 연결된 항목이라면 해당 회의 예정일을 알려주세요."

[Task 2: 출력 형식]
응답은 다음 주요 부분으로 구성되어야 합니다:
1. [마일스톤 식별]: 질문과 관련된 주요 일정 항목을 명확히 나열합니다. (예: 수행계획서 제출일, 계약 체결일, PRB 예정일 등)
2. [문서 기준 일정]: <제공된 문서>에서 추출한 기준 일정 및 조건을 명시합니다.
3. [현재 상황 대비 분석]: 질문자의 상황이 문서 기준과 어떤 차이가 있는지 비교합니다.
4. [일정 리스크 및 대응 방안]: 일정 지연 또는 불확실성에 대한 리스크를 제시하고, 대응 방안을 권고합니다.
5. [결론]: 질문자의 현재 상태에 대한 적정성 또는 조치 필요 여부를 판단합니다.

[Task 3: 품질 보증]
응답이 다음을 보장하도록 합니다:
1. 문서에 명시된 마일스톤 조건과 질문 상황을 정확히 연결해 설명합니다.
2. 일정 지연 또는 미충족이 발생할 경우, 그로 인한 절차/사업/계약상 영향도를 함께 제시합니다.
3. 일정 관련 의사결정기구(예: PRB, VDC 등)가 있다면, 그 연결성과 승인 절차까지 언급합니다.
4. 사용자가 일정 확인을 위해 어떤 정보를 추가로 준비해야 하는지 안내합니다.

[Reflection]
일정이나 마일스톤에 대한 질문은 대부분 실무 리스크, 고객 신뢰, 계약 준수 등과 연관됩니다. 응답이 명확한 기준과 판단 기준을 제시하고 있는지 점검합니다.

[Feedback]
사용자에게 제시된 일정 정보와 대응 방안이 실제 상황에 적용 가능한지, 충분히 실무적인 도움이 되었는지를 피드백 요청합니다.

[Constraints]
1. 응답은 <제공된 문서>에만 기반해야 합니다.
2. 마일스톤이 문서에 명시되지 않은 경우, 추정이 아닌 "문서에 명시되지 않았음"을 명확히 해야 합니다.
3. 결론은 간결하고 실천 가능한 조언을 중심으로 정리합니다.
4. 질문을 새로 생성하거나 모호한 정보 생성은 금지됩니다.

[Context]
사용자는 KT 내부 프로젝트 관리 PM 또는 PL일 가능성이 높으며, 고객과의 일정 조율, 내부 승인 일정 연계 등 실무적 판단 기준을 알고 싶어할 수 있습니다.
""",
        "자유 질의": """
[Task 1: 단계별 지침]
    1. 질문의 주요 키워드와 핵심 내용을 정확히 파악합니다.
    2. 질문에서 주체(예: KT 직원, 협력사 등)와 객체(예: 고객사, 문서, 시스템 등)를 식별하고, 양자의 관계를 구조적으로 정리합니다.
    3. 질문이 포괄적인 경우, 문서상 어떤 절차나 산출물 또는 역할과 연관되어 있는지를 확인하고 그 관점에서 분석합니다.
    4. <제공된 문서> 내에서 관련 키워드와 유사 표현을 기반으로 직접적인 참고 내용을 수집합니다.
    5. 문서에 명시된 절차/정의/사례/Q&A 항목 중 질문과 가장 유사한 내용을 우선 연결하고, 없을 경우 '답변 불가' 처리 기준에 따라 대응합니다.
    6. 불명확하거나 모호한 정보가 있다면, 사용자의 추가 설명을 유도하기 위한 질문도 제안합니다.
    7. 사용자의 질문에 대해 현장 실무자가 이해할 수 있도록 절차적, 실무적, 제도적 기준을 반영하여 서술합니다.

    [Task 2: 출력 형식]
    응답은 다음 주요 부분으로 구성되어야 합니다:
    1. 관계 분석: '[관계 분석]'으로 시작하여 질문 내 인물, 시스템, 조직 간 관계를 설명합니다.
    2. 답변: '[답변]'으로 시작하여 명확하고 간결하게 핵심을 서술합니다.
    3. 설명: '[설명]'으로 시작하여 질문 맥락을 반영한 보충 설명이나 예시를 제공합니다.
    4. 출처: '[출처]'로 시작하여, 참조한 프로세스/문서의 제목과 섹션, 또는 QnA 항목 번호를 명시합니다.

    [Task 3: 품질 보증]
    응답이 다음을 보장하도록 합니다:
    1. 질문에 대한 정보가 <제공된 문서> 내에 있는지 여부를 명확히 구분합니다.
    2. 질문이 다의적이거나 복합적인 경우, 가능한 해석 경로를 구분하여 설명합니다.
    3. 불확실하거나 추가 정보가 필요한 부분은 명확하게 언급하며 사용자에게 추가 정보를 요청합니다.
    4. 모든 정보는 원문 기반이며, 유추 또는 개인적인 견해는 포함하지 않습니다.

    [Reflection]
    응답이 실무자의 질문에 명확히 대응하고 있는지, 문서와 연계된 근거가 충분히 제시되었는지 확인합니다. 특히 복수 해석 가능성이 있는 경우 각 가능성에 대해 언급했는지 점검합니다.

    [Feedback]
    사용자가 응답을 명확하게 이해했는지, 실무적 판단에 도움이 되었는지에 대해 피드백을 요청합니다. 
    - 예: "이 설명이 업무 수행에 도움이 되었나요?" / "추가적으로 알고 싶은 부분이 있다면 알려주세요."

    [Constraints]
    1. 응답은 <제공된 문서>에만 기반해야 합니다.
    2. 새로운 가정을 생성하거나, 문서 외부의 지식을 사용하지 않습니다.
    3. 불명확한 질문에 대해서는 "질문이 모호하여 추가 설명이 필요합니다"와 같은 방식으로 응답합니다.
    4. 질문을 새로 생성하거나, 유도 질문을 자동으로 생성하지 않습니다.

    [Context]
    사용자는 대기업 KT의 프로젝트 관리자입니다. 이들은 일반적으로 다음 업무를 수행합니다:
    - 제안서 작성 및 일정 수립
    - 산출물 검토 및 제출
    - 협력사 및 고객사와의 커뮤니케이션
    - 사업 절차상의 의사결정 보고 또는 판단 수행

    질문은 실무 맥락에서 발생하는 다양한 의문이므로, 이에 대한 응답은 ‘프로세스 기반의 실무 안내’ 또는 ‘프로세스 문서 내 정의 기준’을 충실히 반영해야 합니다.
    """
    }

    return base_prompt + "\n\n" + phase_prompts.get(phase, "") + "\n\n" + question_type_prompts.get(qtype, "")


# ✅ 질문유형별 프롬프트만 선택할 때 (간소화용)
def select_persona_prompt(qtype: str) -> str:
    base = """당신은 KT SI 프로젝트 내부 절차 안내 담당자입니다.\n질문자는 KT 직원입니다."""

    persona_prompts = {
        "자유 질의": """
주어진 절차 문서, Q&A, 용어집을 기반으로 답변하세요.\n– 답변 근거: <제공된 문서>\n[Context] 사용자는 다양한 실무 상황을 포괄하는 질문을 하고 있으며 명확한 근거 기반 안내가 필요합니다.
""",
        "정의 요청": """
질문자의 요청은 특정 용어나 절차 구조에 대한 정의입니다.\n정의는 문서 기준으로 제시하며, 용어집·프로세스 문서에서 직접 발췌하여 제시하세요.
""",
        "수행 절차 안내": """
절차 흐름에 대한 설명이 필요합니다.\n프로세스 문서를 기반으로 각 단계 흐름, 주체, 산출물을 연결하여 설명하세요.
""",
        "산출물·문서 요구 사항": """
사용자는 특정 산출물의 정의, 제출 조건, 작성 주체에 대해 묻고 있습니다.\n문서 기준 작성 요건, 예시, 형상관리 등을 제시해야 합니다.
""",
        "책임·역할 분담": """
질문자는 역할(R&R)의 분담 구조, 충돌 여부, 조정체계에 대해 묻고 있습니다.\n각 단계별 책임자 및 권한자 기준을 명확히 해야 합니다.
""",
        "일정·마일스톤 확인": """
질문자는 특정 단계의 일정 기준, 지연 요인, 마일스톤 조건에 대해 묻고 있습니다.\n문서 기준 일정 구조와 지연 리스크를 설명하세요.
"""
    }

    return base + "\n\n" + persona_prompts.get(qtype, "질문유형 기준 정의가 없습니다.")

# ─────────────────────────────────────────────────────
# 1) 페이지 설정 및 Secrets
# ─────────────────────────────────────────────────────
st.set_page_config(page_title="AX이행봇 LangSmith 통합 테스트봇", layout="wide")
st.title("📌 SIBOT + LangSmith QA")


# 🔐 OpenAI 설정
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

# 🔐 LangSmith 환경변수 수동 세팅
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "SIBOT_MK2"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

@traceable(name="SIBOT Trace 테스트")
def test_trace():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.invoke("SIBOT 프로젝트 잘 보이나요?")

response = test_trace()
print(response.content)
# ─────────────────────────────────────────────────────
# 2) 전역 설정
# ─────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'NanumGothic'
executor = ThreadPoolExecutor(max_workers=5)

# ✅ LangSmith Trace 설정
from langchain.callbacks import LangChainTracer
tracer = LangChainTracer(project_name="SIBOT_MK2")  # 🔥 명시 필요!

# ─────────────────────────────────────────────────────
# 3) 로그인
# ─────────────────────────────────────────────────────
users = {"10154371":"10154371","10154372":"10154372","10156350":"10156350","10151647":"10151647"}
if 'logged_in' not in st.session_state:
    st.sidebar.title("🔒 로그인")

    st.sidebar.subheader("🔍 LangSmith 설정 확인")
    try:
        st.sidebar.json(dict(st.secrets["langsmith"]))  # 강제 dict 변환
    except Exception as e:
        st.sidebar.error("❌ secrets['langsmith'] 확인 불가")
        st.sidebar.code(str(e))
    
    uid = st.sidebar.text_input("ID"); pwd = st.sidebar.text_input("PW", type="password")
    if st.sidebar.button("로그인"):
        if uid in users and users[uid]==pwd:
            st.session_state['logged_in']=True; st.experimental_rerun()
        else: st.sidebar.error("로그인 실패")
    st.stop()

if st.sidebar.button("🔍 LangSmith Trace 수동 테스트"):
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    prompt = PromptTemplate.from_template("LangSmith 테스트입니다. 이름은 {name}입니다.")
    test_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        prompt=prompt,
        callbacks=[tracer]
    )
    result = test_chain.invoke({"name": "SIBOT_MK2"})
    st.sidebar.success(f"응답: {result['text']}")

# ─────────────────────────────────────────────────────
# 4) PDF URL 매핑
# ─────────────────────────────────────────────────────
PROCESS_PDF_URLS = {
    "제안/계약": "https://drive.google.com/uc?export=download&id=1wqvrsYlVje9Oaf1Q0CmEFZsB7nCz0-C4",
    "착수/계획": "https://drive.google.com/uc?export=download&id=1bcthkMK7Qq5EIFgyN82lOWI21YH9W_ME",
    "실행/통제": "https://drive.google.com/uc?export=download&id=1Wk6xW-woqToXWN5bXTVmUmvYdx_jBqxv",
    "종료/사후관리": "https://drive.google.com/uc?export=download&id=1lFQuCg3EflO5g8Rgh4h1mcbCTtQAupga",
}
QNA_PDF_URLS = {
    "제안/계약": "https://drive.google.com/uc?export=download&id=1WWKOJNRrWngf6gTI7dgNp6VnXDv7VKMf",
    "착수/계획": "https://drive.google.com/uc?export=download&id=1H-lkt49Tx45Fo_4Il5PmURb6Pws5nTDO",
    "제안/계약": "https://drive.google.com/uc?export=download&id=1XXQKDRNiaoRWsKHih7txzQomaMBdtE9v",
    "종료/사후관리": "https://drive.google.com/uc?export=download&id=1gGk5ZCBwd1uluYnKsHD9OVWTjacpyTos",
}
WORDPOOL_PDF_URLS = {
    "SI_용어집": "https://drive.google.com/uc?export=download&id=13x3IqootewoBXhlgnpUPVaqerRIkrAEk"
}

# ─────────────────────────────────────────────────────
# 5) UI & 탭 정의
# ─────────────────────────────────────────────────────
run_mode = st.sidebar.radio("⚙️ 실행 모드", ["Retrieval + LLMChain 적용", "LangSmith Tracer 적용"])
if st.sidebar.button("🧪 LangSmith 수동 디버그"):
    tracer = LangChainTracer(project_name="SIBOT_MK2_DEBUG")
    debug_prompt = ChatPromptTemplate.from_template("LangSmith 수동 트레이스 테스트입니다. 이름: {name}")
    chain = LLMChain(llm=ChatOpenAI(), prompt=debug_prompt, callbacks=[tracer])
    result = chain.invoke({"name": "홍길동"})
    st.sidebar.success(f"✅ 응답: {result['text']}")
    
tabs = st.tabs(["Q&A","추가예정"])
qa_tab, _ = tabs

# ─────────────────────────────────────────────────────
# 6) 전처리 (키워드 추출용)
# ─────────────────────────────────────────────────────
kiwi = Kiwi()
def preprocess(text: str) -> str:
    toks = kiwi.analyze(text)[0][0]
    return ' '.join(t.form for t in toks if t.tag.startswith(('N','V','MA')))

# ─────────────────────────────────────────────────────
# 7) 문서 로드 & VectorDB 생성
# ─────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def load_all_docs() -> Tuple[
    Dict[str, List[Document]],
    Dict[str, List[Document]],
    Dict[str, List[Document]],
    Dict[str, List[Document]]
]:
    split_first = CharacterTextSplitter(
        separator=r"\n{2,}|\.(?:\s|$)",
        is_separator_regex=True,
        chunk_size=800, chunk_overlap=0
    )
    split_body = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    proc_map, qna_map, wp_map = {}, {}, {}
    orig_proc, orig_qna, orig_wp = {}, {}, {}

    # 7-1) 프로세스 문서
    for name, url in PROCESS_PDF_URLS.items():
        pages = download_and_load(url)
        orig_proc[name] = pages
        docs: List[Document] = []
        if pages:
            first, *rest = pages
            for txt in split_first.split_text(first.page_content):
                docs.append(Document(page_content=txt, metadata={**first.metadata}))
            docs += split_body.split_documents(rest)
        proc_map[name] = docs

    # 7-2) QnA 문서 (블록 단위)
    for name, url in QNA_PDF_URLS.items():
        pages = download_and_load(url)
        orig_qna[name] = pages
        full_text = "\n".join(p.page_content for p in pages)
        raw_qnas = [
            blk for blk in re.split(r'(?=\[질문\s*\d+\s*[:\]])', full_text)
            if blk.strip()
        ]
        docs: List[Document] = []
        for blk in raw_qnas:
            lines = blk.splitlines()
            tag = next((l for l in lines if l.startswith("[질문")), "")
            question_context = "\n".join(
                l for l in lines
                if not l.startswith("[[") and not l.startswith("[[[")
            ).strip()
            answer_context = "\n".join(
                l for l in lines if l.startswith("[[[답변]") or l.startswith("[[답변]")
            ).strip()
            docs.append(Document(
                page_content=blk,
                metadata={
                    "tag": tag,
                    "question_context": question_context,
                    "answer_context": answer_context,
                    **pages[0].metadata
                }
            ))
        qna_map[name] = docs

    # 7-3) 워드풀 (생략 가능)
    for name, url in WORDPOOL_PDF_URLS.items():
        orig_wp[name] = download_and_load(url)
        wp_map[name] = []

    # 7-4) original_pages 통합
    original_pages: Dict[str, List[Document]] = {}
    for k, v in orig_proc.items():
        original_pages[f"proc:{k}"] = v
    for k, v in orig_qna.items():
        original_pages[f"qna:{k}"] = v
    for k, v in orig_wp.items():
        original_pages[f"wp:{k}"] = v

    return proc_map, qna_map, wp_map, original_pages


@st.cache_resource(ttl=86400)
def build_vectordbs(
    _proc_docs_map: Dict[str, List[Document]],
    _qna_docs_map: Dict[str, List[Document]]
) -> Tuple[Dict[str, FAISS], Dict[str, FAISS]]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    p_vdb = {s: FAISS.from_documents(docs, emb) for s, docs in _proc_docs_map.items()}
    q_vdb = {s: FAISS.from_documents(docs, emb) for s, docs in _qna_docs_map.items()}
    return p_vdb, q_vdb


@st.cache_resource(ttl=86400)
def build_global_qna_vectordb(
    _qna_map: Dict[str, List[Document]]
) -> FAISS:
    all_docs = [d for docs in _qna_map.values() for d in docs]
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    return FAISS.from_documents(all_docs, emb)


@st.cache_resource(ttl=86400)
def build_qna_vectordbs(
    _qna_docs_map: Dict[str, List[Document]]
) -> Dict[str, FAISS]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return {
        step: FAISS.from_documents(docs, emb)
        for step, docs in _qna_docs_map.items()
    }


@st.cache_resource(ttl=86400)
def build_index_vectordbs() -> Dict[str, FAISS]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=os.environ["OPENAI_API_KEY"]
    )
    idxs: Dict[str, FAISS] = {}
    for step, url in PROCESS_PDF_URLS.items():
        docs = extract_index_chunks(url)
        if docs:
            idxs[step] = FAISS.from_documents(docs, emb)
    return idxs


@st.cache_resource(ttl=86400)
def build_substep_vectordbs(
    _proc_map: Dict[str, List[Document]]
) -> Dict[str, Dict[str, FAISS]]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    substep_vdbs: Dict[str, Dict[str, FAISS]] = {}
    for step, docs in _proc_map.items():
        tag_map: Dict[str, List[Document]] = defaultdict(list)
        for doc in docs:
            title = doc.metadata.get("title", "")
            tag_map[title].append(doc)
        substep_vdbs[step] = {
            title: FAISS.from_documents(tag_docs, emb)
            for title, tag_docs in tag_map.items()
        }
    return substep_vdbs


@st.cache_resource(ttl=86400)
def build_qna_substep_vectordbs(
    _qna_docs_map: Dict[str, List[Document]]
) -> Dict[str, Dict[str, FAISS]]:
    emb = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    qna_substep_vdbs: Dict[str, Dict[str, FAISS]] = {}
    for step, docs in _qna_docs_map.items():
        tag_map: Dict[str, List[Document]] = defaultdict(list)
        for doc in docs:
            tag = doc.metadata.get("tag", "")
            tag_map[tag].append(doc)
        qna_substep_vdbs[step] = {
            tag: FAISS.from_documents(tag_docs, emb)
            for tag, tag_docs in tag_map.items()
        }
    return qna_substep_vdbs


@st.cache_resource(ttl=86400)
def build_bm25(
    _proc_map: Dict[str, List[Document]]
) -> Dict[str, BM25Retriever]:
    return {
        s: BM25Retriever.from_documents(docs)
        for s, docs in _proc_map.items()
    }


@st.cache_resource(ttl=86400)
def build_ensemble(
    _p_vdbs: Dict[str, FAISS],
    _bm25s: Dict[str, BM25Retriever]
) -> Dict[str, EnsembleRetriever]:
    ers: Dict[str, EnsembleRetriever] = {}
    for s in _p_vdbs:
        ers[s] = EnsembleRetriever(
            retrievers=[_p_vdbs[s].as_retriever(), _bm25s[s]],
            weights=[0.7, 0.3]
        )
    return ers


# ─────────────────────────────────────────────────────
# 8) 데이터 로드 & 벡터 DB 빌드
# ─────────────────────────────────────────────────────
with st.spinner("📦 데이터 로드 중…"):
    proc_docs_map, qna_docs_map, wp_map, original_pages = load_all_docs()
    index_vectordbs       = build_index_vectordbs()
    substep_vectordbs     = build_substep_vectordbs(proc_docs_map)
    qna_vectordbs         = build_qna_vectordbs(qna_docs_map)
    qna_substep_vectordbs = build_qna_substep_vectordbs(qna_docs_map)
    bm25s                 = build_bm25(proc_docs_map)
    ensemble_retrievers   = build_ensemble(index_vectordbs, bm25s)
# ─────────────────────────────────────────────────────
# 9) Q&A 탭 (STEP→SUBSTEP 추론→TOP3 절차→TOP3 QnA→답변)
# ─────────────────────────────────────────────────────
with qa_tab:
    st.header("AX SI 방법론 이행봇 - Q&A")

    # 1) STEP 선택
    step = st.selectbox("📂 절차 단계를 선택해 주세요", list(PROCESS_PDF_URLS.keys()), key="sel_step")

    # 1-1) 전체 INDEX(서브절차) 목록
    idx_docs = extract_index_chunks(PROCESS_PDF_URLS[step])
    with st.expander("🔖 전체 세부절차 목록", expanded=False):
        for doc in idx_docs:
            st.write(f"- {doc.metadata['title']}")

    # 2) 질문 유형 선택
    qtype = st.selectbox("❓ 질문 유형을 선택해 주세요", QUESTION_TYPES, key="sel_qtype")

    # 3) 질문 입력
    query = st.text_input("💬 질문을 입력하세요", key="input_query")
    status_placeholder = st.empty()

    substep_option = ""
    substep_scores = []
    qna_scores = []

    # 4) 질문 요청
    if st.button("질문 요청", key="btn_query"):
        if not query.strip():
            st.warning("❗ 질문을 입력한 후 버튼을 눌러 주세요.")
            st.stop()

        # 5) Substep 자동 추론
        idx_scores = index_vectordbs[step].similarity_search_with_score(query, k=1)
        substep_option = idx_scores[0][0].page_content
        st.info(f"📌 사용자의 질문은 '{step}' 단계의 \"{substep_option}\"에 대한 \"{qtype}\"입니다.")

        substep_scores = index_vectordbs[step].similarity_search_with_score(query, k=3)

        # 6) Tracer 모드 설정
        if run_mode == "LangSmith Tracer 적용":
            status_placeholder.info("🔍 LangSmith Tracer 실행 중...")
            tracer = LangChainTracer(project_name="SIBOT_MK2")
        else:
            tracer = None

        try:
            status_placeholder.info("⏳ 답변 생성 중...")

            qna_sub_map     = qna_substep_vectordbs.get(step, {})
            default_qna_vdb = qna_vectordbs.get(step)
            qna_vdb_for_sub = qna_sub_map.get(substep_option, default_qna_vdb)
            qna_scores      = qna_vdb_for_sub.similarity_search_with_score(query, k=3) if qna_vdb_for_sub else []

            if qna_scores and qna_scores[0][1] >= 0.7:
                top_doc, _ = qna_scores[0]
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(select_persona_prompt(qtype)),
                    HumanMessagePromptTemplate.from_template(
                        """세부절차: {substep}
QnA 질문: {tag}
질문 내용:
{question_context}

답변 내용:
{answer_context}

사용자 질문: {question}

위 정보를 바탕으로 문장형으로 답변해 주세요."""
                    )
                ])
                answer = LLMChain(
                    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                    prompt=prompt,
                    callbacks=[tracer] if tracer else None
                ).predict(
                    substep=substep_option,
                    tag=top_doc.metadata.get("tag", ""),
                    question_context=top_doc.metadata.get("question_context", ""),
                    answer_context=top_doc.metadata.get("answer_context", ""),
                    question=query
                )
            else:
                proc_vdb = substep_vectordbs.get(step, {}).get(substep_option)
                proc_scores = []
                if proc_vdb:
                    try:
                        proc_scores = proc_vdb.similarity_search_with_score(query, k=1)
                    except Exception as e:
                        st.warning(f"⚠️ 절차 문서 검색 실패: {e}")
                top_doc, _ = proc_scores[0] if proc_scores else (Document(page_content=""), 0)
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessagePromptTemplate.from_template(select_persona_prompt(qtype)),
                    HumanMessagePromptTemplate.from_template(
                        """세부절차: {substep}
절차 문서 청크:
{chunk}

사용자 질문: {question}

위 정보를 바탕으로 문장형으로 답변해 주세요."""
                    )
                ])
                answer = LLMChain(
                    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                    prompt=prompt,
                    callbacks=[tracer] if tracer else None
                ).predict(
                    substep=substep_option,
                    chunk=top_doc.page_content,
                    question=query
                )

            if tracer:
                status_placeholder.success("✅ Tracer 실행 완료 (답변 완료)")
            else:
                status_placeholder.success("✅ 답변 완료")

        except Exception as e:
            status_placeholder.error(f"❗ 오류 발생: {e}")

        # 7) TOP3 절차 서브스텝
        with st.expander("1) TOP3 - 절차 서브스텝", expanded=False):
            for i, (sub_doc, dist) in enumerate(substep_scores, start=1):
                sub = sub_doc.page_content
                similarity = 1.0 - dist
                st.markdown(f"**[TOP_{i}]. {sub} — 유사도 {similarity:.2f}**")
                vdb = substep_vectordbs.get(step, {}).get(sub)
                if vdb:
                    chunk_scores = vdb.similarity_search_with_score(query, k=3)
                    for j, (c_doc, c_dist) in enumerate(chunk_scores, start=1):
                        snippet = c_doc.page_content.replace("\n", " ")[:200] + "…"
                        st.write(f"  {j}. {snippet} (유사도 {1-c_dist:.2f})")
                else:
                    pages = original_pages.get(f"proc:{step}", [])[1:]
                    page_txt = next((p.page_content for p in pages if f"##{sub}" in p.page_content), "")
                    if page_txt:
                        m = re.search(rf"(##{re.escape(sub)}[\s\S]*?)(?=^##\d+\.)", page_txt, flags=re.MULTILINE)
                        block = m.group(1).strip() if m else page_txt.strip()
                        st.text(block)
                    else:
                        st.warning(f"⚠️ '{sub}'에 대한 문서를 찾을 수 없습니다.")
                st.write("---")

        # 8) TOP3 QnA 청크
        with st.expander("2) TOP3 - QnA 청크", expanded=False):
            if not qna_scores:
                st.write("⚠️ 해당 서브스텝에 대한 Q&A가 없습니다.")
            for i, (doc, score) in enumerate(qna_scores, start=1):
                tag = doc.metadata.get("tag", "")
                qc = doc.metadata.get("question_context", "").strip()
                ac = doc.metadata.get("answer_context", "").strip()
                st.markdown(f"**[TOP_{i}]. {tag} — Score {score:.2f}**")
                if qc:
                    st.write(f"'{qc}'")
                if ac:
                    st.write(f"[[[답변] '{ac}'")
                if not qc and not ac:
                    fallback = doc.page_content.strip().replace("\n", " ")[:300]
                    st.text(f"📄 원문 청크: {fallback} …")
                st.write("---")

        # 9) 답변 출력
        if answer:
            with st.expander("3) 생성된 문장형 답변", expanded=True):
                st.markdown(answer)

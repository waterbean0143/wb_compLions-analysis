import requests
import tempfile
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def load_pdf_from_url_to_docs(url):
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

def create_process_retrievers():
    urls = {
        "vdc_a_프로세스": "https://drive.google.com/uc?export=download&id=1lSEWk7KDgHR71yHcjEKniWzhqwL7T2fC",
        "vdc_a_대표질문": "https://drive.google.com/uc?export=download&id=1KGJv9ttGD7ErcSWymE-0jiMjOzbnq6iI"
    }

    all_docs = []
    for name, url in urls.items():
        docs = load_pdf_from_url_to_docs(url)
        for doc in docs:
            doc.metadata["source_name"] = name
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs_split = text_splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs_split, embeddings)
    return vectorstore.as_retriever()
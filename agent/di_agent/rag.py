from __future__ import annotations

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@st.cache_resource
def create_vector_db(file_path: str, api_key: str, base_url: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        openai_api_base=base_url,
    )
    # 为了避免不同 PDF/参数互相覆盖，使用一个确定性的 persist 目录
    import hashlib, os

    persist_hash = hashlib.sha256(
        (file_path + api_key + base_url).encode("utf-8", errors="ignore")
    ).hexdigest()[:16]
    persist_dir = os.path.join(".chroma", persist_hash)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="pdfqa",
    )
    return vectorstore


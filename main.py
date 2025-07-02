import os
import pickle
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="News Research Tool")
st.title("📰 News Research & Summarization Tool")

# Sidebar inputs
st.sidebar.title("🔗 Add News URLs")
urls = [st.sidebar.text_input(f"Enter URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

# File to store vector data
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

# Load LLM with API key (✅ FIXED)
llm = OpenAI(temperature=0.9, model="gpt-3.5-turbo", api_key=openai_api_key)

if process_url_clicked:
    urls = [url for url in urls if url.strip() != ""]
    if not urls:
        st.error("Please enter at least one valid URL.")
    else:
        with st.spinner("🔄 Loading and processing URLs..."):
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()

            # Split the text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            docs = text_splitter.split_documents(data)

            # Embeddings and FAISS (✅ FIXED)
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
            vectorstore_openai = FAISS.from_documents(docs, embeddings)

            with open(file_path, "wb") as f:
                pickle.dump(vectorstore_openai, f)

        st.success("✅ URLs processed and vector store saved!")

# Query interface
query = st.text_input("💬 Ask a question about the articles above:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore_openai = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore_openai.as_retriever()
        )

        result = chain({"question": query}, return_only_outputs=True)

        st.header("📍 Answer")
        st.write(result['answer'])

        st.subheader("📚 Sources")
        if result.get("sources"):
            sources = result["sources"].split("\n")
            for src in sources:
                st.markdown(f"- {src}")
        else:
            st.write("No sources found.")
    else:
        st.warning("⚠️ Please process URLs first.")

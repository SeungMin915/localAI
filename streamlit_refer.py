import streamlit as st
import tiktoken
from loguru import logger
import requests

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import StreamlitChatMessageHistory


def call_potens_api(prompt, api_key):
    url = "https://ai.potens.ai/api/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {"prompt": prompt}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, str):
            return result, []
        elif isinstance(result, dict):
            for key in ["response", "answer", "message", "content", "text", "data"]:
                if key in result:
                    return result[key], result.get("source_documents", [])
            return str(result), result.get("source_documents", [])
        else:
            return str(result), []
    except Exception as e:
        logger.error(f"Potens API 오류: {e}")
        return f"ERROR: {str(e)}", []


def main():
    st.set_page_config(page_title="DirChat", page_icon="📄")
    st.title("📄 _Private Data :red[QA Chat]_")

    if "vectordb" not in st.session_state:
        st.session_state.vectordb = None

    with st.sidebar:
        uploaded_files = st.file_uploader("📎 문서 업로드", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        api_key = st.text_input("🔑 Potens API Key", type="password")
        process = st.button("📚 문서 처리")

    if process:
        if not api_key:
            st.warning("API 키를 입력해주세요.")
            st.stop()

        with st.spinner("문서를 처리 중입니다..."):
            docs = get_text(uploaded_files)
            chunks = get_text_chunks(docs)
            vectordb = get_vectorstore(chunks)
            st.session_state.vectordb = vectordb
            st.success("✅ 문서 분석 완료! 질문을 입력해보세요.")

    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "안녕하세요! 문서 기반 질문을 해보세요."
        }]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            if not st.session_state.vectordb:
                st.warning("먼저 문서를 업로드하고 처리해야 합니다.")
                st.stop()

            with st.spinner("답변 생성 중..."):
                retriever = st.session_state.vectordb.as_retriever(search_type="similarity", k=3)
                docs = retriever.get_relevant_documents(query)

                context = "\n\n".join([doc.page_content for doc in docs])
                final_prompt = f"""다음 문서를 참고하여 질문에 답하세요.

[문서 내용]:
{context}

[질문]:
{query}

[답변]:
"""

                answer, _ = call_potens_api(final_prompt, api_key)

                st.markdown(answer)
                with st.expander("📄 참고 문서"):
                    for doc in docs:
                        st.markdown(f"**출처**: {doc.metadata.get('source', '알 수 없음')}")
                        st.markdown(doc.page_content)

                st.session_state.messages.append({"role": "assistant", "content": answer})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def get_text(docs):
    all_docs = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as f:
            f.write(doc.getvalue())
            logger.info(f"Uploaded: {file_name}")

        if file_name.endswith('.pdf'):
            loader = PyPDFLoader(file_name)
        elif file_name.endswith('.docx'):
            loader = Docx2txtLoader(file_name)
        elif file_name.endswith('.pptx'):
            loader = UnstructuredPowerPointLoader(file_name)
        else:
            continue

        all_docs.extend(loader.load_and_split())

    return all_docs


def get_text_chunks(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    return splitter.split_documents(texts)


def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(chunks, embeddings)


if __name__ == '__main__':
    main()

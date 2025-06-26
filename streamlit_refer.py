import streamlit as st
import tiktoken
from loguru import logger
import os
import uuid
import tempfile

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
        page_title="현대해상 HEART Insight AI",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("_현대해상 :red[HEART Insight AI]_ :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "안녕하세요! 미래 모빌리티 트렌드 분석 및 보험 시사점 도출 AI 'HEART Insight AI'입니다. "
                       "PDF, DOCX, PPTX 파일을 업로드하고 'Process' 버튼을 눌러주시면, "
                       "해당 문서에 대해 궁금한 점을 질문하실 수 있습니다."
        }]

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    with st.sidebar:
        st.header("1. 문서 업로드 및 처리")
        uploaded_files = st.file_uploader(
            "Upload your file (PDF, DOCX, PPTX)",
            type=['pdf', 'docx', 'pptx'],
            accept_multiple_files=True,
            help="최대 200MB까지 업로드 가능합니다."
        )

        st.header("2. OpenAI API 키 입력")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            key="chatbot_api_key",
            type="password",
            help="OpenAI 플랫폼에서 발급받은 API Secret Key를 입력해주세요."
        )

        process_button = st.button("Process Documents")
        st.markdown("---")
        st.header("3. 대화 초기화")
        if st.button("대화 초기화", help="모든 대화 기록을 삭제하고 초기 상태로 돌아갑니다."):
            st.session_state.messages = [{
                "role": "assistant",
                "content": "대화가 초기화되었습니다. 다시 문서를 업로드하고 시작해주세요."
            }]
            st.session_state.conversation = None
            st.session_state.chat_history = []
            st.session_state.processComplete = False
            st.experimental_rerun()

    if process_button:
        if not uploaded_files:
            st.error("⚠️ 문서를 업로드해주세요. (PDF, DOCX, PPTX)")
            st.stop()

        if not openai_api_key:
            st.error("⚠️ OpenAI API 키를 입력해주세요.")
            st.stop()

        with st.spinner("문서 처리 중... (텍스트 추출, 청크 분할, 벡터화)"):
            try:
                files_text = get_text(uploaded_files)
                if not files_text:
                    st.error("⚠️ 업로드된 파일에서 텍스트를 추출하는 데 실패했습니다. 파일 형식을 확인해주세요.")
                    st.stop()

                text_chunks = get_text_chunks(files_text)
                vectorstore = get_vectorstore(text_chunks)
                if vectorstore is None:
                    st.error("벡터 스토어 생성에 실패했습니다. 로그를 확인해주세요.")
                    st.stop()

                st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
                st.session_state.processComplete = True
                st.success("✅ 문서 처리가 완료되었습니다. 이제 질문을 입력해주세요!")
            except Exception as e:
                logger.error(f"문서 처리 중 오류 발생: {e}", exc_info=True)
                st.error(f"❌ 문서 처리 중 오류가 발생했습니다: {e}")
                st.session_state.processComplete = False

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages_langchain")

    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            if chain is None or not st.session_state.processComplete:
                st.warning("⚠️ AI가 아직 준비되지 않았습니다. 사이드바에서 파일을 업로드하고 'Process' 버튼을 먼저 눌러주세요.")
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ AI가 아직 준비되지 않았습니다. 파일 업로드 후 'Process'를 눌러주세요."})
                st.stop()

            with st.spinner("답변을 생성 중입니다..."):
                try:
                    result = chain({"question": query, "chat_history": st.session_state.chat_history})

                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']

                    response = result['answer']
                    source_documents = result.get('source_documents', [])

                    st.markdown(response)

                    if source_documents:
                        with st.expander("참고 문서 확인"):
                            for i, doc in enumerate(source_documents[:3]):
                                source_info = doc.metadata.get('source', f"알 수 없는 출처 {i+1}")
                                page_info = doc.metadata.get('page', '페이지 정보 없음')
                                st.markdown(f"**출처:** {source_info} (페이지: {page_info})", help=doc.page_content)
                                st.markdown("---")
                except Exception as e:
                    logger.error(f"답변 생성 중 오류 발생: {e}", exc_info=True)
                    response = "죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다."
                    st.error(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


def get_text(docs):
    doc_list = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Using temporary directory: {tmp_dir}")
        for doc in docs:
            ext = os.path.splitext(doc.name)[1].lower()
            temp_filename = os.path.join(tmp_dir, f"{uuid.uuid4()}{ext}")
            try:
                with open(temp_filename, "wb") as file:
                    file.write(doc.getvalue())
                logger.info(f"Saved {doc.name} to temporary path: {temp_filename}")
            except Exception as e:
                logger.error(f"Error saving file {doc.name} to temp: {e}", exc_info=True)
                st.error(f"파일 임시 저장 중 오류: {e}")
                continue
            try:
                if ext == '.pdf':
                    loader = PyPDFLoader(temp_filename)
                elif ext == '.docx':
                    loader = Docx2txtLoader(temp_filename)
                elif ext == '.pptx':
                    loader = UnstructuredPowerPointLoader(temp_filename)
                else:
                    logger.warning(f"Unsupported file type for {doc.name}: {ext}")
                    st.warning(f"지원하지 않는 파일 형식입니다: {ext}")
                    continue
                documents = loader.load()
                doc_list.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {doc.name}")
            except Exception as e:
                logger.error(f"Error loading document from {doc.name}: {e}", exc_info=True)
                st.error(f"문서 로딩 중 오류 발생: {e}")
                continue
    return doc_list


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documents split into {len(chunks)} chunks.")
    return chunks


def get_vectorstore(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sroberta-multitask",
            encode_kwargs={'normalize_embeddings': True}
        )
        vectordb = FAISS.from_documents(text_chunks, embeddings)
        logger.info("Vectorstore created using FAISS.")
        return vectordb
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}", exc_info=True)
        st.error(f"벡터 스토어 생성 중 오류 발생: {e}")
        return None


def get_conversation_chain(vectorstore, openai_api_key):
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0.2)
        logger.info("ChatOpenAI model initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True)
        st.error("❌ OpenAI API 키가 유효하지 않거나 모델 초기화에 실패했습니다.")
        return None

    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose=True
        )
        logger.info("ConversationalRetrievalChain created successfully.")
        return conversation_chain
    except Exception as e:
        logger.error(f"Failed to create conversation chain: {e}", exc_info=True)
        st.error("❌ 대화 체인 생성 중 오류가 발생했습니다.")
        return None


if __name__ == '__main__':
    main()

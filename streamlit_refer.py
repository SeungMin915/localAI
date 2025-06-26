import streamlit as st
import tiktoken
from loguru import logger
import os
import uuid
import tempfile # tempfile 라이브러리 추가: 파일을 안전하게 임시 저장하기 위함

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

# --- Streamlit 앱의 메인 함수 ---
def main():
    # 1. 페이지 기본 설정
    st.set_page_config(
        page_title="현대해상 HEART Insight AI", # 더 명확한 페이지 타이틀
        page_icon=":books:",
        layout="wide", # 넓은 레이아웃 사용
        initial_sidebar_state="expanded" # 사이드바 기본 확장
    )

    st.title("_현대해상 :red[HEART Insight AI]_ :books:") # AI 솔루션명 반영

    # 2. Streamlit 세션 상태(Session State) 초기화
    # st.session_state는 Streamlit 앱의 상태를 유지하는 데 사용됩니다.
    # 앱이 리로드되어도 변수 값이 유지되도록 합니다.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None # LangChain Conversation Chain 객체

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # LangChain Memory에 사용될 채팅 기록

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "안녕하세요! 미래 모빌리티 트렌드 분석 및 보험 시사점 도출 AI 'HEART Insight AI'입니다. "
                       "PDF, DOCX, PPTX 파일을 업로드하고 'Process' 버튼을 눌러주시면, "
                       "해당 문서에 대해 궁금한 점을 질문하실 수 있습니다."
        }] # Streamlit chat message UI에 표시될 전체 대화 기록

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False # 문서 처리 완료 여부

    # 3. 사이드바 UI 구성
    with st.sidebar:
        st.header("1. 문서 업로드 및 처리")
        uploaded_files = st.file_uploader(
            "Upload your file (PDF, DOCX, PPTX)",
            type=['pdf', 'docx', 'pptx'],
            accept_multiple_files=True,
            help="최대 200MB까지 업로드 가능합니다." # 파일 사이즈 제한 명시 (코드에는 구현되어 있지 않음)
        )
        
        st.header("2. OpenAI API 키 입력")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            key="chatbot_api_key",
            type="password",
            help="OpenAI 플랫폼에서 발급받은 API Secret Key를 입력해주세요. (예: sk-xxxxxxxxxxxx)"
        )
        
        process_button = st.button("Process Documents") # 버튼 텍스트 변경
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
            st.experimental_rerun() # 앱을 새로고침하여 상태를 반영

    # 4. Process 버튼 클릭 시 로직
    if process_button:
        if not uploaded_files:
            st.error("⚠️ 문서를 업로드해주세요. (PDF, DOCX, PPTX)")
            st.stop() # 파일이 없으면 여기서 중단

        if not openai_api_key:
            st.error("⚠️ OpenAI API 키를 입력해주세요.")
            st.stop() # API 키가 없으면 여기서 중단

        with st.spinner("문서 처리 중... (텍스트 추출, 청크 분할, 벡터화)"):
            try:
                files_text = get_text(uploaded_files)
                if not files_text:
                    st.error("⚠️ 업로드된 파일에서 텍스트를 추출하는 데 실패했습니다. 파일 형식을 확인해주세요.")
                    st.stop()

                text_chunks = get_text_chunks(files_text)
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
                st.session_state.processComplete = True
                st.success("✅ 문서 처리가 완료되었습니다. 이제 질문을 입력해주세요!")
            except Exception as e:
                logger.error(f"문서 처리 중 오류 발생: {e}", exc_info=True) # exc_info=True로 전체 traceback 출력
                st.error(f"❌ 문서 처리 중 오류가 발생했습니다: {e}")
                st.session_state.processComplete = False # 오류 발생 시 완료 상태를 False로

    # 5. 기존 대화 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # StreamlitChatMessageHistory는 LangChain 메모리와 Streamlit 세션 상태를 동기화합니다.
    history = StreamlitChatMessageHistory(key="chat_messages_langchain") # key 충돌 방지를 위해 변경

    # 6. 사용자 입력 처리 및 AI 응답 생성
    if query := st.chat_input("질문을 입력해주세요."):
        # 사용자의 질문을 UI에 표시 및 세션 상태에 추가
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # 대화 체인이 초기화되었는지 다시 한번 확인 (방어적 코드)
            chain = st.session_state.conversation
            if chain is None or not st.session_state.processComplete:
                st.warning("⚠️ AI가 아직 준비되지 않았습니다. 사이드바에서 파일을 업로드하고 'Process' 버튼을 먼저 눌러주세요.")
                # 오류 메시지를 세션 상태에 추가하여 UI에 유지
                st.session_state.messages.append({"role": "assistant", "content": "⚠️ AI가 아직 준비되지 않았습니다. 사이드바에서 파일을 업로드하고 'Process' 버튼을 먼저 눌러주세요."})
                st.stop() # 여기서 스크립트 실행을 중단하여 불필요한 연산을 막음
            
            with st.spinner("답변을 생성 중입니다..."):
                try:
                    # LangChain Chain 호출: 질문과 이전 대화 기록을 함께 전달
                    result = chain({"question": query, "chat_history": st.session_state.chat_history})
                    
                    # OpenAI Callback을 사용하여 토큰 사용량 확인 (선택 사항)
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history'] # 업데이트된 채팅 기록 저장
                    
                    response = result['answer']
                    source_documents = result.get('source_documents', []) # source_documents가 없을 경우 빈 리스트 반환
                    
                    st.markdown(response)
                    
                    # 참고 문서가 있을 경우에만 Expander 표시
                    if source_documents:
                        with st.expander("참고 문서 확인"):
                            # 최대 3개까지만 표시 (IndexError 방지)
                            for i, doc in enumerate(source_documents[:min(len(source_documents), 3)]):
                                # st.markdown(doc.metadata['source'], help=doc.page_content)
                                # Source 정보가 없을 경우를 대비하여 .get() 사용
                                source_info = doc.metadata.get('source', f"알 수 없는 출처 {i+1}")
                                page_info = doc.metadata.get('page', '페이지 정보 없음') # PDF 등 페이지 정보가 있다면 활용
                                st.markdown(f"**출처:** {source_info} (페이지: {page_info})", help=doc.page_content)
                                # 각 문서 내용도 상세하게 보여줄 수 있도록 수정
                                # st.markdown(f"내용 요약: {doc.page_content[:200]}...") # 너무 길면 요약
                                st.markdown("---") # 문서 간 구분선
                
                except Exception as e:
                    logger.error(f"답변 생성 중 오류 발생: {e}", exc_info=True)
                    response = "죄송합니다. 답변을 생성하는 도중 오류가 발생했습니다. 잠시 후 다시 시도하거나, 개발자에게 문의해주세요."
                    st.error(response) # 사용자에게 오류 메시지 표시

        # AI의 답변을 UI에 표시 및 세션 상태에 추가
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- Helper Functions (변경 없음, 주석 추가) ---
def tiktoken_len(text):
    """텍스트의 토큰 길이를 계산합니다."""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

# get_text 함수 수정: tempfile을 사용하여 파일을 안전하게 처리
def get_text(docs):
    doc_list = []
    # tempfile.TemporaryDirectory()를 사용하여 임시 디렉토리를 생성
    # 이 디렉토리는 with 블록이 끝나면 자동으로 삭제됩니다.
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Using temporary directory: {tmp_dir}")
        
        for doc in docs:
            ext = os.path.splitext(doc.name)[1].lower()
            # 임시 디렉토리 안에 유니크한 파일명으로 파일 저장 경로 생성
            temp_filename = os.path.join(tmp_dir, f"{uuid.uuid4()}{ext}")
            
            # 업로드된 Streamlit 파일을 임시 파일로 저장
            try:
                with open(temp_filename, "wb") as file:
                    file.write(doc.getvalue())
                logger.info(f"Saved {doc.name} to temporary path: {temp_filename}")
            except Exception as e:
                logger.error(f"Error saving file {doc.name} to temp: {e}", exc_info=True)
                st.error(f"파일을 임시 저장하는 중 오류가 발생했습니다: {e}")
                continue # 파일 저장 실패 시 다음 파일로 넘어갑니다.

            # 파일 확장자에 따라 적절한 LangChain Document Loader 사용
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
                
                documents = loader.load() # load_and_split() 대신 load() 사용
                doc_list.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {doc.name}")
            except Exception as e:
                logger.error(f"Error loading document from {doc.name}: {e}", exc_info=True)
                st.error(f"문서 로딩 중 오류가 발생했습니다: {e}. 파일 내용을 확인해주세요.")
                continue
                
    return doc_list

def get_text_chunks(documents):
    """문서 리스트를 청크로 분할합니다."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,      # 각 청크의 최대 문자 수
        chunk_overlap=100,   # 청크 간 중복되는 문자 수 (문맥 유지에 도움)
        length_function=tiktoken_len # 토큰 길이를 기준으로 청크 분할
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documents split into {len(chunks)} chunks.")
    return chunks

def get_vectorstore(text_chunks):
    """텍스트 청크를 임베딩하여 벡터 스토어에 저장합니다."""
    # Hugging Face 모델을 사용하여 텍스트 임베딩 생성
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask", # 한국어에 최적화된 임베딩 모델
        model_kwargs={'device': 'cpu'}, # CPU 사용 (GPU 없을 시)
        encode_kwargs={'normalize_embeddings': True} # 임베딩 벡터 정규화
    )
    # FAISS (Facebook AI Similarity Search)를 사용하여 벡터 스토어 생성
    # FAISS는 유사도 검색을 빠르게 수행할 수 있도록 최적화되어 있습니다.
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    logger.info("Vectorstore created using FAISS.")
    return vectordb

def get_conversation_chain(vectorstore, openai_api_key):
    """LangChain ConversationalRetrievalChain을 생성합니다."""
    # OpenAI Chat 모델 초기화
    # model_name: 사용할 GPT 모델 (gpt-3.5-turbo, gpt-4 등)
    # temperature: 창의성(0에 가까울수록 보수적, 1에 가까울수록 창의적)
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0.2) # temperature 조정
        logger.info("ChatOpenAI model initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize ChatOpenAI: {e}", exc_info=True)
        st.error("❌ OpenAI API 키가 유효하지 않거나 모델 초기화에 실패했습니다. 키를 다시 확인해주세요.")
        return None # 오류 발생 시 None 반환

    # Conversation Chain 생성
    # llm: 사용할 LLM 모델
    # chain_type: 문서 결합 방식 ("stuff"는 모든 문서를 프롬프트에 넣음)
    # retriever: 벡터 스토어에서 관련 문서를 가져오는 역할
    # memory: 대화 이력을 관리하여 문맥을 유지
    # get_chat_history: 대화 이력을 가져오는 함수 (lambda h: h는 전달된 그대로 사용)
    # return_source_documents: 답변과 함께 참고 문서 반환 여부
    # verbose: 상세 로그 출력 여부 (디버깅에 유용)
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_type='mmr', verbose=True), # 'vervose' 오타 수정 완료
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose=True
        )
        logger.info("ConversationalRetrievalChain created successfully.")
        return conversation_chain
    except Exception as e:
        logger.error(f"Failed to create conversation chain: {e}", exc_info=True)
        st.error("❌ 대화 체인 생성 중 예상치 못한 오류가 발생했습니다. 개발자에게 문의해주세요.")
        return None # 오류 발생 시 None 반환

# --- 앱 실행 진입점 ---
if __name__ == '__main__':
    main()

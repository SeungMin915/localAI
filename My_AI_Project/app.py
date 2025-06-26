# app.py
# Streamlit 기반 HEART Insight AI 웹 솔루션의 메인 파일 (모든 기능 통합)

# -----------------------------------------------------
# 1. 라이브러리 임포트 (파일 최상단)
# -----------------------------------------------------
import streamlit as st
import os
from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import time
import uuid

# 'modules' 폴더에 있는 커스텀 모듈들을 임포트합니다.
from modules import ai_interface
from modules import data_collector
from modules import trend_analyzer

# --- LangChain RAG 기능에 필요한 라이브러리 임포트 ---
import tiktoken
import tempfile
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

# -----------------------------------------------------
# 2. 메인 Streamlit 애플리케이션 함수 정의 (def main():)
# -----------------------------------------------------
def main():
    # -----------------
    # 2-1. 환경 변수 로드
    # -----------------
    load_dotenv()
    POTENS_API_KEY = os.getenv("POTENS_API_KEY")
    
    # -----------------
    # 2-2. 페이지 기본 설정 (첫 Streamlit 명령어여야 함)
    # -----------------
    st.set_page_config(
        page_title="현대해상 HEART Insight AI",
        page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # -----------------
    # 2-3. Streamlit 세션 상태(Session State) 초기화
    # -----------------
    if "messages" not in st.session_state: st.session_state["messages"] = []
    if "api_ready" not in st.session_state: st.session_state["api_ready"] = False
    if "data_collected" not in st.session_state: st.session_state["data_collected"] = False
    if "collected_data" not in st.session_state: st.session_state["collected_data"] = []
    if "topic_analysis_result" not in st.session_state: st.session_state["topic_analysis_result"] = None
    if "rag_conversation" not in st.session_state: st.session_state["rag_conversation"] = None
    if "rag_processed" not in st.session_state: st.session_state["rag_processed"] = False

    # -----------------
    # 2-4. UI 구성: 메인 화면
    # -----------------
    st.title("🚗 현대해상 HEART Insight AI")
    st.subheader("미래 모빌리티 트렌드 분석 및 보험 시사점 도출 솔루션")
    st.markdown("---")
    st.markdown(
        """
        **HEART Insight AI**는 급변하는 미래 모빌리티 환경의 트렌드를 심층 분석하고,
        이를 현대해상의 보험 상품 개발 및 리스크 평가에 필요한 핵심 시사점과 기회 요인으로 도출하는 AI 기반 솔루션입니다.
        """
    )
    st.header("📈 AI 트렌드 분석 대시보드")
    st.info("이곳에 뉴스, 보고서, 특허 등에서 수집된 데이터를 기반으로 한 트렌드 예측 그래프, 키워드 네트워크, 토픽 모델링 결과 등 인터랙티브 시각화가 구현될 예정입니다.")
    if st.session_state.data_collected:
        st.markdown("### 수집된 데이터 미리보기")
        st.dataframe(pd.DataFrame(st.session_state.collected_data)[:10], use_container_width=True)
        st.markdown("---")
        if st.session_state.topic_analysis_result and st.session_state.topic_analysis_result['fig_html']:
            st.markdown("### 📊 토픽 모델링 시각화")
            st.components.v1.html(st.session_state.topic_analysis_result['fig_html'], height=600)
            st.markdown("---")
            st.markdown("### 📝 주요 트렌드 (토픽) 요약")
            topic_info_df = pd.DataFrame(st.session_state.topic_analysis_result['topic_info'])
            topic_info_df.index.name = 'Topic ID'
            st.dataframe(topic_info_df[['Count', 'Name', 'Representation']], use_container_width=True)
            st.success("✅ 트렌드 분석 결과가 대시보드에 반영되었습니다!")
        else:
            st.warning("토픽 모델링 결과를 불러오는 데 실패했습니다. 로그를 확인해주세요.")
    st.markdown("---")

    # -----------------
    # 2-5. 대화형 AI 챗봇 (Potens.dev API & RAG 통합)
    # -----------------
    st.header("💬 트렌드 분석 Q&A 챗봇")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("질문을 입력해주세요.")
    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성 중입니다..."):
                response = ""
                if st.session_state.rag_processed and st.session_state.rag_conversation:
                    chain = st.session_state.rag_conversation
                    try:
                        result = chain({"question": user_query})
                        response = result['answer']
                    except Exception as e:
                        response = f"문서 분석 중 오류가 발생했습니다: {e}"
                        logger.error(f"RAG chatbot error: {e}")
                elif st.session_state.api_ready:
                    response = ai_interface.call_potens_api(
                        prompt_message=user_query, api_key=POTENS_API_KEY, history=st.session_state.messages
                    )
                else:
                    response = "⚠️ 챗봇 기능이 비활성화되었습니다. 사이드바에서 기능을 활성화해주세요."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        if not st.session_state.api_ready and not st.session_state.rag_processed:
            st.warning("⚠️ 챗봇 기능이 비활성화되었습니다. 사이드바에서 기능을 선택하고 준비해주세요.")
            st.markdown("`AI 챗봇 준비` 또는 `Process Documents` 버튼을 눌러주세요.")

    # -----------------
    # 2-6. 사이드바 UI (기능 제어 버튼)
    # -----------------
    with st.sidebar:
        st.header("프로젝트 정보")
        st.markdown("**프로젝트명:** HEART Insight AI")
        st.markdown("**개발:** 메이커스랩")
        st.markdown("---")
        st.header("기능 제어")

        st.subheader("📄 문서 기반 Q&A 챗봇")
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        process_docs_button = st.button("Process Documents", key="process_docs")
        if process_docs_button:
            if not POTENS_API_KEY:
                st.error("⚠️ Potens.dev API 키가 설정되지 않았습니다. `.env` 파일을 확인해주세요.")
            elif not uploaded_files:
                st.error("⚠️ 문서를 업로드해주세요.")
            else:
                with st.spinner("문서 처리 중..."):
                    try:
                        # RAG 체인 생성 (Potens.dev API 키 사용)
                        os.environ["OPENAI_API_BASE"] = "https://ai.potens.ai/api/chat" # base_url 설정
                        files_text = get_text(uploaded_files)
                        if not files_text:
                            st.error("⚠️ 업로드된 파일에서 텍스트를 추출하는 데 실패했습니다.")
                        else:
                            vectorstore = get_vectorstore(get_text_chunks(files_text))
                            st.session_state.rag_conversation = get_conversation_chain(vectorstore, POTENS_API_KEY)
                            if st.session_state.rag_conversation:
                                st.session_state.rag_processed = True
                                st.success("✅ 문서 처리가 완료되었습니다. 이제 문서에 대해 질문해보세요!")
                    except Exception as e:
                        st.error(f"❌ 문서 처리 중 오류가 발생했습니다: {e}")
                        st.session_state.rag_processed = False
        st.markdown("---")
        st.subheader("💬 AI 트렌드 챗봇")
        if st.button("AI 챗봇 준비", key="activate_chatbot"):
            if POTENS_API_KEY:
                st.session_state.api_ready = True
                st.success("🎉 Potens.dev API 준비 완료! 이제 챗봇에 질문해보세요.")
                logger.info("Potens.dev API is ready.")
                st.experimental_rerun()
            else:
                st.error("API 키가 설정되지 않았습니다. `.env` 파일을 확인해주세요.")
                st.session_state.api_ready = False

        st.markdown("---")
        st.header("트렌드 데이터 수집 및 분석")
        if st.button("트렌드 데이터 수집 및 분석 시작", help="뉴스 기사를 크롤링하고 AI 분석을 수행합니다."):
            if not st.session_state.data_collected:
                keywords = ["전기차 배터리", "자율주행 보험", "UAM 시장", "PBV 현대차", "MaaS 서비스"]
                collected_articles = []
                with st.spinner("뉴스 기사 데이터 수집 중..."):
                    for keyword in keywords:
                        articles = data_collector.scrape_google_news(keyword, pages=1)
                        collected_articles.extend(articles)
                        st.info(f"'{keyword}' 관련 기사 {len(articles)}개 수집 완료.")
                if not collected_articles:
                    st.warning("⚠️ 데이터를 수집하지 못했습니다. 테스트용 더미 데이터를 사용합니다.")
                    collected_articles = [{"title": "전기차 배터리 리스크", "content": "내용", "source": "Dummy Data", "keywords": "전기차"}, {"title": "자율주행", "content": "내용", "source": "Dummy Data", "keywords": "자율주행"}] # Simplified dummy data
                if collected_articles:
                    st.session_state.collected_data = collected_articles
                    st.session_state.data_collected = True
                    st.success(f"✅ 총 {len(st.session_state.collected_data)}개의 기사 데이터로 분석을 시작합니다!")
                    with st.spinner("수집된 데이터를 기반으로 AI 트렌드 분석 중..."):
                        analysis_result = trend_analyzer.perform_topic_modeling(st.session_state.collected_data)
                        st.session_state.topic_analysis_result = analysis_result
                    if st.session_state.topic_analysis_result and st.session_state.topic_analysis_result['topics']:
                        st.success("✅ AI 트렌드 분석이 성공적으로 완료되었습니다. 대시보드를 확인하세요!")
                    else:
                        st.error("❌ AI 분석 중 오류가 발생했습니다. 로그를 확인해주세요.")
                else:
                    st.error("데이터 수집과 테스트 데이터 로드에 모두 실패했습니다.")
                    st.session_state.data_collected = False
                    st.session_state.topic_analysis_result = None
            else:
                st.info("데이터 수집 및 분석이 이미 완료되었습니다. 앱을 새로고침하여 다시 시작하세요.")

        st.markdown("---")
        st.header("대화 초기화")
        if st.button("전체 대화 초기화", help="모든 대화 기록과 처리된 문서를 삭제합니다."):
            st.session_state.messages = []
            st.session_state.rag_processed = False
            st.session_state.rag_conversation = None
            st.session_state.data_collected = False
            st.session_state.collected_data = []
            st.session_state.topic_analysis_result = None
            st.experimental_rerun()

# -----------------------------------------------------
# 3. RAG 기능 Helper 함수들 (def main() 함수 정의 아래에 위치)
# -----------------------------------------------------
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

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
                continue
            try:
                if ext == '.pdf': loader = PyPDFLoader(temp_filename)
                elif ext == '.docx': loader = Docx2txtLoader(temp_filename)
                elif ext == '.pptx': loader = UnstructuredPowerPointLoader(temp_filename)
                else:
                    logger.warning(f"Unsupported file type: {ext}")
                    continue
                documents = loader.load()
                doc_list.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {doc.name}")
            except Exception as e:
                logger.error(f"Error loading document from {doc.name}: {e}", exc_info=True)
                continue
    return doc_list

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100, length_function=tiktoken_len)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documents split into {len(chunks)} chunks.")
    return chunks

def get_vectorstore(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        vectordb = FAISS.from_documents(text_chunks, embeddings)
        logger.info("Vectorstore created using FAISS.")
        return vectordb
    except Exception as e:
        logger.error(f"Failed to create vectorstore: {e}", exc_info=True)
        return None

def get_conversation_chain(vectorstore, api_key):
    try:
        llm = ChatOpenAI(api_key=api_key, model_name='gpt-3.5-turbo', temperature=0, base_url="https://ai.potens.ai/api/chat")
        logger.info(f"ChatOpenAI model initialized with base_url: {llm.base_url}")
        return ConversationalRetrievalChain.from_llm(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h, return_source_documents=True, verbose=True
        )
    except Exception as e:
        logger.error(f"Failed to create conversation chain: {e}", exc_info=True)
        return None

# -----------------------------------------------------
# 4. 앱 실행 진입점 (파일 가장 마지막에 위치)
# -----------------------------------------------------
if __name__ == '__main__':
    main()
# trend_analyzer.py
# NLP 기반 트렌드 분석 및 토픽 모델링 모듈 (최종 버전)

import pandas as pd
from loguru import logger
from typing import List, Dict, Any
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
# KoNLPy 설치 및 Mecab 백엔드 의존성
from konlpy.tag import Mecab

# -----------------------------------------------------
# 1. 한국어 텍스트 분석을 위한 KoNLPy-Mecab 초기화
#    - Mecab 설치가 실패할 경우를 대비한 예외 처리 포함
# -----------------------------------------------------
try:
    mecab = Mecab()
    logger.info("KoNLPy Mecab initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Mecab: {e}. Please check your KoNLPy and Mecab installation.", exc_info=True)
    mecab = None

def preprocess_korean_text(text_list: List[str]) -> List[str]:
    """
    한국어 텍스트를 토큰화하고 불용어를 제거하여 전처리합니다.
    - Mecab이 설치되지 않았을 경우, 원본 텍스트를 그대로 반환합니다.
    """
    if mecab is None:
        logger.warning("Mecab is not available. Skipping text preprocessing and tokenization.")
        return text_list
        
    logger.info("Starting Korean text preprocessing...")
    processed_texts = []
    
    # 불용어(stopwords) 리스트 (기획안의 '텍스트 정제' 단계)
    stopwords = ['은', '는', '이', '가', '을', '를', '에', '와', '과', '하다', '있다', '없다', '되다', '이다', '것', '수', '등', '등등']
    
    for text in text_list:
        try:
            # Mecab을 사용해 명사, 동사, 형용사만 추출
            words = mecab.pos(text)
            filtered_words = [
                word[0] for word in words 
                if word[1] in ['NNG', 'NNP', 'VV', 'VA'] and word[0] not in stopwords
            ]
            processed_texts.append(' '.join(filtered_words))
        except Exception as e:
            logger.warning(f"Failed to preprocess text with Mecab: {e}. Skipping this document.")
            processed_texts.append(text) # 오류 발생 시 원본 텍스트 유지
        
    logger.info("Korean text preprocessing finished.")
    return processed_texts

def perform_topic_modeling(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    수집된 뉴스 기사를 기반으로 BERTopic 모델링을 수행합니다.
    - 시각화 생성 시 발생하는 torch 오류를 회피하는 로직이 포함됩니다.
    
    Args:
        articles (List[Dict[str, Any]]): data_collector에서 수집된 기사 데이터.
        
    Returns:
        Dict[str, Any]: 토픽 모델링 결과 (토픽 정보, 시각화 데이터).
    """
    if not articles:
        logger.warning("No articles provided for topic modeling.")
        return {"topics": [], "topic_info": pd.DataFrame().to_dict(), "fig_html": None}
        
    logger.info(f"Starting topic modeling with BERTopic on {len(articles)} documents...")
    
    df = pd.DataFrame(articles)
    documents = df['content'].tolist()
    
    # 텍스트 전처리
    preprocessed_documents = preprocess_korean_text(documents)
    
    # BERTopic 모델 초기화 및 학습
    try:
        # Sentence-Transformer 모델을 한국어 모델로 명시적으로 설정
        # 'jhgan/ko-sroberta-multitask' 모델은 huggingface_hub에서 다운로드
        embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
        
        # BERTopic 모델 초기화 (representation_model은 KeyBERTInspired 사용)
        representation_model = KeyBERTInspired()
        
        topic_model = BERTopic(
            language="multilingual", # 한국어 포함 여러 언어 지원
            calculate_probabilities=True,
            embedding_model=embedding_model, # 명시적으로 임베딩 모델 전달
            representation_model=representation_model,
            verbose=True, # 상세 로그 출력
        )
        
        # 모델 학습 및 토픽 추출
        topics, probabilities = topic_model.fit_transform(preprocessed_documents)
        
        # 토픽 정보 추출
        topic_info = topic_model.get_topic_info()
        
        logger.success(f"Topic modeling completed successfully. Found {len(topic_info)} topics.")
        
        # ------------------------------------------------------------------
        # 2. 시각화 생성에 대한 예외 처리
        #    - torch/backend 오류 발생 시 시각화는 건너뛰고 나머지 결과는 반환
        # ------------------------------------------------------------------
        fig_html = None
        try:
            fig = topic_model.visualize_topics()
            fig_html = fig.to_html(full_html=False)
            logger.info("Topic visualization generated successfully.")
        except Exception as vis_e:
            logger.error(f"Failed to generate visualization (likely a backend/torch issue): {vis_e}", exc_info=True)
            fig_html = "<div style='color:red;'>Failed to generate visualization. See terminal logs for details.</div>"
            
        return {
            "topics": topics,
            "topic_info": topic_info.to_dict(),
            "fig_html": fig_html
        }
        
    except Exception as e:
        logger.error(f"An error occurred during BERTopic modeling: {e}", exc_info=True)
        return {"topics": [], "topic_info": {}, "fig_html": "<div style='color:red;'>Failed to perform topic modeling. See terminal logs.</div>"}

if __name__ == '__main__':
    # 모듈 테스트를 위한 코드
    print("--- BERTopic 모델링 테스트 ---")
    # 임시 테스트 데이터 (실제 수집 데이터처럼 가정)
    test_articles = [
        {"title": "전기차 배터리 화재 예방 기술", "content": "최근 전기차 배터리 열 폭주를 막는 기술 개발이 중요해지고 있습니다. 이는 보험 산업의 새로운 리스크 요인으로 부상합니다.", "source": "test_data"},
        {"title": "자율주행 레벨4 보험 상품 개발 동향", "content": "자율주행차의 사고 책임 소재를 운전자에서 제조사로 바꾸는 보험 상품이 연구되고 있습니다. 현대해상 같은 보험사에게 새로운 기회입니다.", "source": "test_data"},
        {"title": "도심항공교통(UAM) 실증 사업 시작", "content": "국토교통부가 서울 도심에서 UAM 실증 비행을 성공적으로 마쳤습니다. UAM 관련 보험 상품의 필요성이 커지고 있습니다.", "source": "test_data"},
        {"title": "전기차 충전 인프라 확대 정책 발표", "content": "정부에서 2030년까지 전기차 충전소 확대를 위한 대규모 예산을 투입합니다. 이는 전기차 보험 시장의 성장을 가속화할 것입니다.", "source": "test_data"},
        {"title": "PBV 시장의 새로운 비즈니스 모델", "content": "현대차와 기아는 목적 기반 모빌리티(PBV)를 통해 다양한 서비스를 제공할 계획입니다. 이에 맞는 맞춤형 보험 상품이 필요합니다.", "source": "test_data"},
        {"title": "자율주행차 사이버 보안 위협 증가", "content": "커넥티드카에 대한 해킹 위협이 증가하면서 새로운 형태의 사이버 보험이 주목받고 있습니다. 현대해상은 이에 대한 대비가 필요합니다.", "source": "test_data"},
        {"title": "전기차 화재 진압용 소방 장비", "content": "전기차 배터리 화재는 기존 소화 방식과 달라 특수 진압 장비가 필요합니다. 이는 손해율에 영향을 미칠 수 있습니다.", "source": "test_data"},
        {"title": "전기차 충전소 인프라 확대", "content": "전국에 전기차 충전소 인프라가 빠르게 확산되고 있습니다. 이는 보험 상품의 리스크 평가에 새로운 요인으로 작용합니다.", "source": "test_data"},
    ]
    
    result = perform_topic_modeling(test_articles)
    
    if result["topics"]:
        print("\n--- 토픽 모델링 결과 ---")
        topic_info_df = pd.DataFrame(result["topic_info"])
        print(topic_info_df[['Topic', 'Count', 'Name', 'Representation']].head())
        if result["fig_html"]:
            print("\n토픽 시각화 HTML이 생성되었습니다. 브라우저에서 확인할 수 있습니다.")
    else:
        print("\n토픽 모델링에 실패했습니다. 로그를 확인해주세요.")
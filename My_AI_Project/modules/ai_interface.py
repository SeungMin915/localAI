# ai_interface.py
# Potens.dev API 연동을 위한 모듈 (미연시 개발 성공 사례 기반)

import os
import requests
import json
from loguru import logger
from dotenv import load_dotenv
from typing import List, Dict, Any

# .env 파일에서 환경 변수 로드
load_dotenv()

# Potens.dev API의 정확한 엔드포인트 URL (공유해주신 정보 반영)
API_URL = "https://ai.potens.ai/api/chat"

def call_potens_api(prompt_message: str, api_key: str, history: List[Dict[str, str]] = None) -> str:
    """
    주어진 프롬프트 메시지로 Potens.dev API를 호출하고 응답을 반환합니다.
    """
    if not api_key:
        logger.error("API 키가 누락되었습니다. Potens.dev API 호출을 중단합니다.")
        return "API 키가 설정되지 않았습니다. 관리자에게 문의해주세요."

    # HTTP 헤더 설정 (인증 토큰과 데이터 타입 명시)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 요청 페이로드(본문) 설정 (공유해주신 "prompt" 형식 사용)
    # 멀티턴 대화 맥락을 'prompt' 메시지에 포함하여 전달합니다.
    full_prompt = ""
    if history:
        for msg in history:
            full_prompt += f"{msg['role']}: {msg['content']}\n"
        full_prompt += f"user: {prompt_message}"
    else:
        full_prompt = prompt_message
    
    payload = {
        "prompt": full_prompt # AI에게 전달할 핵심 메시지
    }
    
    try:
        logger.info(f"Potens.dev API 호출 시작 (엔드포인트: {API_URL})")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        
        response_json = response.json()
        
        # 응답 파싱 및 반환 (공유해주신 {'message': '...'} 구조 사용)
        if "message" in response_json:
            ai_response_text = response_json["message"].strip()
            logger.info("Potens.dev API 호출 성공.")
            return ai_response_text
        else:
            logger.error(f"API 응답에 'message' 키가 없습니다: {response_json}")
            return "API 응답 형식이 올바르지 않습니다. 개발자에게 문의해주세요."

    except requests.exceptions.RequestException as e:
        logger.error(f"API 호출 오류 발생 (네트워크/타임아웃/HTTP): {e}", exc_info=True)
        return "API 호출에 실패했습니다. 네트워크 상태를 확인해주세요."
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {e}", exc_info=True)
        return "알 수 없는 오류가 발생했습니다."

if __name__ == '__main__':
    # 모듈 테스트를 위한 코드
    test_api_key = os.getenv("POTENS_API_KEY")
    print("--- Potens.dev API 호출 테스트 ---")
    if test_api_key:
        test_query = "미래 모빌리티 트렌드에 대해 간략히 설명해줘."
        response_text = call_potens_api(test_query, api_key=test_api_key)
        print(f"\nAI 답변:\n{response_text}")
    else:
        print("\n[경고] .env 파일에 POTENS_API_KEY가 설정되지 않아 테스트를 실행할 수 없습니다.")
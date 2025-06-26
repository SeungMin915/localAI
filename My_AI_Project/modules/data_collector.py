# data_collector.py
# ... (상단 임포트 부분) ...
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from loguru import logger
from typing import List, Dict, Any
import time

# 구글 뉴스 검색을 위한 기본 URL
GOOGLE_NEWS_SEARCH_URL = "https://www.google.com/search"

# ... (clean_text 함수는 그대로 유지) ...
def clean_text(text: str) -> str:
    """
    텍스트에서 HTML 태그, 이메일, URL, 특수 문자 등을 제거하여 정제합니다.
    """
    # HTML 태그 제거
    clean = re.compile('<.*?>')
    cleaned_text = re.sub(clean, '', text)
    # URL 제거
    cleaned_text = re.sub(r'http\S+|www.\S+', '', cleaned_text, flags=re.MULTILINE)
    # 이메일 제거
    cleaned_text = re.sub(r'\S*@\S*\s?', '', cleaned_text)
    # 특수 문자, 이모티콘 등 제거
    cleaned_text = re.sub(r'[^\w\s\.\,\?!\-]', '', cleaned_text)
    return cleaned_text.strip()

def scrape_google_news(keyword: str, pages: int = 1) -> List[Dict[str, Any]]:
    """
    구글 뉴스 검색 결과에서 기사 제목, 링크, 요약 등을 크롤링합니다.
    """
    all_articles = []
    
    logger.info(f"'{keyword}' 키워드로 구글 뉴스 크롤링 시작. 총 {pages} 페이지.")
    
    # User-Agent 설정 (크롤링 차단 회피)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    for page in range(pages):
        start_index = page * 10
        
        # 요청 파라미터 설정 (오타 수정)
        params = {
            'q': keyword,
            'tbm': 'nws',
            'start': start_index,
            'gl': 'us',  # <-- 오타 수정 완료!
            'hl': 'en',  # <-- 오타 수정 완료!
        }

        try:
            response = requests.get(GOOGLE_NEWS_SEARCH_URL, params=params, headers=headers)
            response.raise_for_status()
            print(response.text)
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 구글 뉴스 기사 목록을 담고 있는 HTML 요소 찾기
            news_list = soup.select('div.SoaPz')
            
            if not news_list:
                logger.warning(f"페이지 {page+1}에서 뉴스 기사를 찾을 수 없습니다 (HTML 구조 변경 또는 차단). 크롤링을 중단합니다.")
                break
            
            for news_item in news_list:
                title_elem = news_item.select_one('div.n0jPhd')
                link_elem = news_item.select_one('a')
                summary_elem = news_item.select_one('div.GI74Re')
                
                if title_elem and link_elem and summary_elem:
                    title = title_elem.get_text(strip=True)
                    link = link_elem.get('href', '링크 없음')
                    summary = summary_elem.get_text(strip=True)
                    
                    article = {
                        "source": "Google News",
                        "category": "모빌리티 트렌드",
                        "title": title,
                        "content": clean_text(summary),
                        "publish_date": datetime.now().isoformat(),
                        "url": link,
                        "keywords": keyword
                    }
                    all_articles.append(article)
            
            logger.info(f"페이지 {page+1} 크롤링 완료. 현재까지 총 {len(all_articles)}개 기사 수집.")
            
            # 요청 간 딜레이 추가
            time.sleep(3)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP 요청 중 오류 발생: {e}", exc_info=True)
            break
        except Exception as e:
            logger.error(f"크롤링 중 예상치 못한 오류 발생: {e}", exc_info=True)
            continue

    logger.success(f"'{keyword}' 키워드 크롤링 최종 완료. 총 {len(all_articles)}개 기사 수집.")
    return all_articles

if __name__ == '__main__':
    # 모듈 테스트를 위한 코드
    print("--- Google 뉴스 크롤링 테스트 ---")
    test_keyword = "electric vehicle battery"
    collected_data = scrape_google_news(test_keyword, pages=2)
    
    if collected_data:
        print(f"\n총 {len(collected_data)}개의 기사 수집 완료.")
        for i, article in enumerate(collected_data[:5]):
            print(f"\n--- 기사 {i+1} ---")
            print(f"제목: {article['title']}")
            print(f"링크: {article['url']}")
            print(f"요약: {article['content'][:100]}...")
            print(f"출처: {article['source']}")
            print("-" * 20)
    else:
        print("\n수집된 기사가 없습니다. 키워드나 URL을 확인해주세요.")
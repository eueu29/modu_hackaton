import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda 
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema import BaseOutputParser
import json
import re

# 모델 정의
gpt4o = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0.3)
claude = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

# 데이터 로드
data = json.load(open('/home/yoojin/ML/aiffel/HackaThon/modu_hackaton/LLM/files/menu_1017.json', 'r', encoding='utf-8'))

# 장바구니 클래스
class ShoppingCart:
    def __init__(self):
        self.cart = []

    def add_to_cart(self, order_data, num):
        cart_item = {
            "name": order_data["name"],
            "num": num,
            "price": order_data["price"],
            "set_menu": order_data["set_menu"]
        }
        if order_data["set_menu"]:
            cart_item["set_price"] = order_data["set_burger_price"]  # 나중에 set_price로 변경 예정
        self.cart.append(cart_item)
        return self.cart

class IntentChain:
    def __init__(self, model):
        self.model = model

    def parse_response(self, response_str):
        try:
            response_dict = json.loads(response_str)
        except json.JSONDecodeError:
            response_dict = {"error": "파싱 오류", "원본": response_str}
        return response_dict

    def invoke(self, question):
        chain = PromptTemplate.from_template("""
        아래 질문을 보고 사용자가 원하는 의도를 '주문', '추천', '취소', '결제', '기타' 중 하나로 정확하게 분류하세요.

        분류 기준:
        - 주문: 특정 메뉴를 정확히 주문하려는 경우 (예: '빅맥 하나 주세요', '감자튀김 추가')
        - 추천: 특정 메뉴 이름 대신 추천을 요청하거나 메뉴 선택에 도움을 원하는 경우 (예: '무엇이 맛있나요?', '매콤한 음식 추천해줘')
        - 취소: 이전에 진행된 주문을 취소하려는 경우 (예: '주문 취소해 주세요', '아까 주문한 것 취소하고 싶어요')
        - 결제: 주문 완료 후 결제를 요청하는 경우 (예: '결제할거야', '주문 완료')
        - 기타: 위의 네 가지 분류에 해당하지 않는 경우 (예: '화장실은 어디인가요?', '영업 시간은 언제인가요?')

        <질문>
        {question}
        </질문>

        질문의 분류와 해당 질문에 포함된 주요 키워드를 JSON 형식으로 출력하세요.

        조건:
        - 질문의 분류와 해당 질문에 포함된 주요 키워드를 딕셔너리 형식으로 출력하세요.
        - 분류가 '주문'일 때만 키워드를 출력하세요. 다른 분류에서는 키워드를 출력하지 마세요
        - 키워드는 버거, 사이드, 음료 등 메뉴명만 추출합니다.
        - 주문이 세트메뉴인 경우, 버거명에 세트메뉴 키워드를 포함하세요

        예시 출력: {{"분류": "주문", "키워드": ["불고기 버거 세트", "후렌치후라이", "콜라", "상하이 치킨 스낵랩"]}}
        예시 출력: {{"분류": "주문", "키워드": "빅맥"}}
        예시 출력: {{"분류": "추천"}}
        """) | self.model | StrOutputParser()

        intent_result = chain.invoke(question)
        return self.parse_response(intent_result)
    # 키워드 = type list
    
# 이름추측 LLM
class CustomOutputParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        # 'answer' 키로 '정답'을 반환
        return {"answer": text.strip()}

class NameChain:
    def __init__(self, model):
        self.model = model
        self.parser = CustomOutputParser()

    def invoke(self, question):
        chain = PromptTemplate.from_template("""
            사용자의 질문과 메뉴 리스트를 비교하여 사용자의 질문과 비슷한 메뉴가 있는지 확인하세요.
            비슷한 이름의 메뉴가 있다면 비슷한 메뉴를 모두 출력하세요
            비슷한 메뉴가 없다면 "없음"이라고 출력하세요.
            출력 시 메뉴 이름만 출력하세요

            <메뉴 리스트>
            ['맥크리스피 스리라차 마요', '베토디 스리라차 마요', '더블 쿼터파운더 치즈', '쿼터파운더 치즈', '맥스파이시 상하이 버거', '토마토 치즈 비프 버거', '맥크리스피 디럭스 버거', '맥크리스피 클래식 버거', '빅맥', '트리플 치즈버거', '1955 버거', '맥치킨 모짜렐라', '맥치킨', '더블 불고기 버거', '불고기 버거', '슈슈 버거', '슈비 버거', '베이컨 토마토 디럭스', '치즈버거', '더블 치즈버거', '햄버거', '소시지 스낵랩', '맥윙 2조각', '맥윙 4조각', '맥윙 8조각', '상하이 치킨 스낵랩', '골든 모짜렐라 치즈스틱 4조각', '골든 모짜렐라 치즈스틱 2조각', '맥너겟 6조각', '맥너겟 4조각', '맥스파이시 치킨텐더 2조각', '후렌치 후라이', '스위트 앤 사워 소스', '스위트 칠리 소스', '케이준 소스', '그리머스 쉐이크', '자두 천도복숭아 칠러', '제주 한라봉 칠러', '코카콜라', '코카콜라 제로', '스프라이트', '환타', '바닐라 라떼', '카페라떼', '카푸치노', '아메리카노', '드립 커피', '아이스 바닐라 라떼', '아이스 카페라떼', '아이스 아메리카노', '아이스 드립 커피', '디카페인 바닐라 라떼', '디카페인 카페라떼', '디카페인 카푸치노', '디카페인 아메리카노', '디카페인 아이스 바닐라 라떼', '디카페인 아이스 카페라떼', '디카페인 아이스 아메리카노', '바닐라 쉐이크', '딸기 쉐이크', '초코 쉐이크', '생수', '애플 파이', '오레오 맥플러리', '베리 스트로베리 맥플러리']
            </메뉴 리스트>

            <질문>
            {question}
            </질문>

            """) | self.model | self.parser
        return chain.invoke(question)
    
# 메뉴 리스트 로드 클래스
class LoadMenu:
    def __init__(self):
        self.menu_list = []
        self.data = data

    def get_list(self):
        for item in self.data:
            menu_name = item['page_content']['name']
            self.menu_list.append(menu_name)
        return self.menu_list

# 빠른 주문 모듈 클래스
class FastOrderModule:
    def __init__(self, model):
        self.intent_chain = IntentChain(model)
        self.name_chain = NameChain(model)
        self.menu_list = LoadMenu().get_list()
        self.menu_data = []
        self.unrecognized_keywords = []
        self.recommend = False
        self.shopping_cart = ShoppingCart()

    def save_menu(self, keyword):
        for item in data:
            menu_name_normalized = re.sub(r'\s+', '', item['page_content']['name']).replace("버거", "").strip()
            keyword_normalized = re.sub(r'\s+', '', keyword).replace("버거", "").strip()
            if menu_name_normalized == keyword_normalized:
                return {
                    "category": item['page_content']['category'],
                    "name": item['page_content']['name'],
                    "product_status": item['page_content']['product_status'],
                    "description": item['page_content']['description'],
                    "price": item['page_content']['price'],
                    "set_burger_price": item['page_content']['set_burger_price'],
                    "nutrition": item['page_content']['nutrition'],
                    "origin_info": item['page_content']['origin_info'],
                    "set_menu": False
                }
        return None

    def fast_order(self, question):
        intent_result = self.intent_chain.invoke(question)
        category = intent_result.get('분류')
        intent_keyword = intent_result.get('키워드')
        print(f"category: {category}, intent_keyword: {intent_keyword}")
        
        if category == '주문':
            if intent_keyword:
                for keyword in intent_keyword: 
                    print(f"keyword: {keyword}")
                    menu_data_json = self.save_menu(keyword)
                    
                    if menu_data_json is None:
                        # 이름 추측 체인을 사용해 비슷한 메뉴 찾기
                        similar_menu = self.name_chain.invoke(keyword)["answer"]
                        print(f"similar_menu: {similar_menu}")
                        
                        if similar_menu != "없음":
                            menu_data_json = self.save_menu(similar_menu)

                    if menu_data_json is None:
                        self.unrecognized_keywords.append(keyword)
                        self.recommend = True
                    else:
                        if "세트" in keyword:
                            menu_data_json['set_menu'] = True
                        self.menu_data.append(menu_data_json)

                for order_data in self.menu_data:
                    ###### 수량.... 얘도 LLM 연결해서 숫자만 추출해야할것같은데...
                    num = int(input(f"{order_data['name']} 메뉴가 맞으신가요? 수량은 몇 개 드릴까요? "))
                    self.shopping_cart.add_to_cart(order_data, num)

                if self.unrecognized_keywords:
                    print(f"선택하신 메뉴 중 확인할 수 없는 항목이 있습니다: {self.unrecognized_keywords}. 추천 챗봇을 연결해 드리겠습니다.")
                    self.recommend = True
                print(f"shopping_cart: {self.shopping_cart.cart}")
            else:
                print("주문 키워드가 없습니다. 다시 시도해 주세요.")
        else:
            print("추천시스템으로 연결해드리겠습니다.")

# 빠른 주문 실행
fast_order_module = FastOrderModule(gpt4o)
question = input("무엇을 주문하시겠습니까? ")
fast_order_module.fast_order(question)
import pandas as pd
import json
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda 
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_teddynote import logging

load_dotenv()
logging.langsmith("Module")

# 모델 정의
gpt4o_mini = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0.3)
gpt4o = ChatOpenAI(model_name="gpt-4o-2024-08-06") 
claude = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

# 데이터 로드
file_dir = '/home/yoojin/ML/aiffel/HackaThon/modu_hackaton/LLM/files/menu_1017.json'
data = json.load(open(file_dir, 'r', encoding='utf-8'))


# 장바구니 클래스
class ShoppingCart:
    cart = []
    
    @classmethod
    
    def add_to_cart(cls, new_item):
        for item in new_item:
            cls.cart.append(item)
        return cls.cart
    
    @classmethod
    def calculate_total(cls):
        total = 0
        for order in cls.cart:
            if order[0]['set_menu']:
                total += int(order[0]['set_price'])
            else:
                total += int(order[0]['price']) * int(order[0]['quantity'])
        return total

    @classmethod
    def print_order(cls):
        total = cls.calculate_total()
        order_details = []
        
        for order in cls.cart:
            if order[0]['set_menu']:
                set_items = [item['name'] for item in order]
                set_price = order[0]['set_price']
                order_details.append(f"'세트'로 '{set_items[0]}', '{set_items[1]}', '{set_items[2]}' 1개, 가격 {set_price}원")
            else:
                item = order[0]
                order_details.append(f"'{item['name']}' {item['quantity']}개, {item['price']}원")
        
        order_string = ", ".join(order_details)
        print(f"{order_string} 주문되어, 총 {total}원 주문되었습니다.")


class IntentChain:
    def __init__(self):
        self.model = gpt4o_mini

    def additional_invoke(self, question):
        intent_chain = ChatPromptTemplate.from_messages([
            ("system", """
            사용자의 질문을 '종료', '결제', '추천' 중 하나로 분류하세요.

            분류 기준:
            - 종료: 주문을 종료하려는 경우
            - 결제: 주문 완료 후 결제를 요청하는 경우, 요청사항이 없다고 하는 경우 (예: '결제할거야', '주문 완료', '없어')
            - 추천: 취소나 결제가 아닌 기타 문의 (예: '추천 메뉴 있어요?', '가장 인기 있는 메뉴는?','아까 주문한거 취소할래')

            출력 형식:
            - 예시: 결제
            """),
            ("ai","추가주문이나 다른 요청이 있으신가요?"),
            ("human", "{input}")
        ]) | self.model | StrOutputParser()
        
        return intent_chain.invoke({"input" :question})

# 추천 모듈 클래스
class RecommendModule:
    def __init__(self, model, shared_memory, shared_window_memory):
        self.model = model
        self.recommend_chain = self.create_recommend_chain()
        self.menu = self.guess_menu()
        self.memory = shared_memory  # 공유 메모리 사용
        self.window_memory = shared_window_memory
    
    def guess_menu(self):
        guess_template = ChatPromptTemplate.from_messages([
        ("system", 
        """
        이전 대화 내역을 참고하여 사용자가 언급한 메뉴를 검색할 수 있도록 필요한 정보를 추출하세요. 
        사용자가 비교 요청 또는 추가 정보를 요구한 메뉴 이름을 "검색_내용" 항목에 포함시키세요.
        사용자가 주문한다고 언급한 모든 메뉴 항목을 반드시 "검색_내용"에 포함시키세요.
        검색이 필요한 항목과 주문 내역 외에는 "검색_내용"에 포함시키지 마세요.

        이전 대화 내역:
        {chat_history}

        출력 형식 예시:
        {{
            "검색_내용": ["메뉴1", "메뉴2"]
        }}

        반드시 위의 출력 형식에 맞춰 JSON 형태로 응답해주세요. 마크다운 표시는 하지 마세요.
        """),
        ("human", "{question}"),
    ])

        guess_chain = {
            "chat_history": RunnableLambda(lambda _: self.memory.load_memory_variables({})["chat_history"]),
            "question": RunnablePassthrough()
        } | guess_template | gpt4o_mini | StrOutputParser()
        
        return guess_chain
    
    def create_recommend_chain(self):
        docs = [
            Document(
                page_content=json.dumps(obj['page_content'], ensure_ascii=False),
            )
            for obj in json.load(open(file_dir, 'r', encoding='utf-8'))
        ]
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=0)
        split_docs = text_splitter.split_documents(docs)

        embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        cache_dir = LocalFileStore(f"./.cache/embeddings/{file_dir.split('/')[-1]}")
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings,
            document_embedding_cache=cache_dir,
            namespace="solar-embedding-1-large",
        )
        vectorstore = FAISS.from_documents(split_docs, cached_embedder)
        faiss = vectorstore.as_retriever(search_kwargs={"k": 4})

        bm25 = BM25Retriever.from_documents(split_docs)
        bm25.k = 2

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25, faiss],
            weights=[0.3, 0.7],
            search_type="mmr",
        )
        
        recommend_template = ChatPromptTemplate.from_messages([
            ("system",
            """
            당신은 맥도날드의 친절한 점원입니다.
            고객님의 주문을 도와드리세요. 나이와 상관없이 '고객님'이라고만 부르고, 모든 설명은 어린아이도 이해할 수 있게 해주세요. 
            대답은 세 문장 이내로, 간결하고 친절하게 응대합니다.
            
            **응대 지침:**
            1. 제공된 정보만으로 답변하며, 주문 확인 외의 불필요한 멘트는 생략합니다.
            2. 질문과 가장 관련성이 높은 정보를 선택해 대답하세요.
            3. 메뉴 추천은 2개 이하로 제한하며, 신메뉴를 우선 추천하세요.
            4. 확실하지 않으면 "정확한 답변을 드리기 어렵습니다만, 추가로 확인 후 도와드리겠습니다."라고 답하세요.
            5. 이전 대화내역을 최우선으로 고려하고, 그 외 현재 대화의 context를 참고합니다.

            **주문 조건:**
            - 메뉴는 세트와 단품으로 구분됩니다.
            - 세트 메뉴는 기본으로 버거, 사이드, 음료가 포함되며, 미디엄 사이즈가 기본입니다.
            - 미디엄 사이즈 세트 메뉴 가격은 context의 'set_price' 가격입니다. 해당 정보가 없으면 "죄송합니다, 세트 구성이 불가능한 항목입니다. 대신 단품으로 주문하시거나 다른 메뉴를 선택해 주세요."라고 안내하세요.
            - 라지 사이즈 세트 메뉴로의 업그레이드는 800원 추가 요금이 부과됩니다.
            - 사이드는 기본으로 후렌치 후라이 미디엄이며, 코울슬로로만 무료 변경 가능합니다.
            - 음료는 기본으로 코카콜라 미디엄이며, 음료 변경 시 차액이 발생할 수 있습니다.

            **주문 절차:**
            - 메뉴가 결정되면 세트 여부와 수량을 모두 확인한 후에만 주문을 완료하세요.
            - 세트 여부와 수량이 모두 확인되지 않은 경우에는 `completion` 값을 `False`로 유지합니다.
            - 모든 정보가 확인되면 `completion`을 `True`로 설정하고, 추가 주문 여부나 주문 내역 확인에 대한 질문은 생략합니다.
            
            **주문 결과 출력:**
            - 모든 LLM 응답을 반드시 '주문 결과 예시'의 JSON 형식으로 출력합니다.
            - 세트 메뉴는 함께 묶어 리스트로 나타내고, 단품은 각각 리스트로 구분하여 출력합니다.
            - 마크다운 표시는 하지 마세요.
            
            **주문 결과 예시:**
            {{
                "completion": false,  // 모든 정보가 확인되지 않은 경우 false 유지
                "reply" : "{{llm_response}}",
                "order" : 
                [
                    [
                        {{"name": "불고기 버거", "quantity": "1", "price": "4200", "set_menu": true, "set_price": "4000"}},
                        {{"name": "후렌치후라이", "set_menu": false}},
                        {{"name": "코카콜라", "set_menu": false}}
                    ],
                    [
                        {{"name": "상하이 치킨 스낵랩", "quantity": "1", "price": "4000", "set_menu": false}}
                    ],
                    [
                        {{"name": "오레오 맥플러리", "quantity": "1", "price": "4300", "set_menu": false}}
                    ]
                ]
            }}
            
            **Context:** {context}
            **Chat History:** {chat_history}
            """),
            ("human", "{question}"),
        ])


        recommend_chain = {
            "context": ensemble_retriever,
            "chat_history": RunnableLambda(lambda _: self.window_memory.load_memory_variables({})["history"]),
            "question": RunnablePassthrough()
        } | recommend_template | self.model
        
        return recommend_chain

    def invoke(self, question):
        result = self.menu.invoke(question)
        query = json.loads(result)
        str_result = f"질문:{question}, 추가 고려내용: {', '.join(query['검색_내용'])}"
        print(str_result)
        return self.recommend_chain.invoke(str_result)

# 주문 모듈 클래스
class OrderModule: 
    def __init__(self, model):
        self.memory = self.get_shared_memory()  # 공유 메모리 생성
        self.window_memory = self.get_shared_window_memory()
        self.intent_chain = IntentChain()
        self.recommend_module = RecommendModule(model, self.memory, self.window_memory)  # 공유 메모리 전달

    def get_shared_memory(self):
        return ConversationBufferMemory(
            return_messages=True,
            memory_key="chat_history"
        )
    def get_shared_window_memory(self):
        return  ConversationBufferWindowMemory(
            k=2, 
            return_messages=True
        )
        
    def save_context(self, user_message, ai_message):
        self.memory.save_context({"input": str(user_message)}, {"output": str(ai_message)})
        self.window_memory.save_context({"input": str(user_message)}, {"output": str(ai_message)})

        
    def extract_json_data(self, text):
        try:
            # 중첩된 배열과 JSON 객체를 모두 추출하는 정규식
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                # JSON 배열만 추출해서 문자열로 반환
                json_str = json_match.group(0)
                
                # 문자열을 Python 리스트/딕셔너리로 변환
                return json.loads(json_str)
            else:
                return None
        except :
            return None
    
    def handle_additional_requests(self):
        while True:
            print("add_req")
            user_message = input("추가 주문이나 다른 요청이 있으신가요?").strip()
            
            intent = self.intent_chain.additional_invoke(user_message)
            print(f"intent:{intent}")
            print(f"type:{type(intent)}")
            
            if intent == "추천":
                self.recommend_module.memory.clear()
                self.execute_additional_order(user_message)  # 추가 주문 처리
            elif intent == "결제":
                ShoppingCart.print_order()
                print("결제를 도와드리겠습니다")
                # 결제 로직 추가 필요
                return False
            elif intent == "종료":
                print("주문을 종료합니다. 다음에 또 뵙겠습니다 고객님.")
                return False
            else:
                print("죄송합니다. 요청을 이해하지 못했습니다.")
    
    def execute_order(self):
        try:
            self.recommend_module.memory.clear()

            while True:
                print("first_req")
                user_message = input("입력: ").strip()

                if not user_message:
                    continue

                response = self.recommend_module.invoke(user_message)
                # JSON 데이터만 추출
                ai_response = response.content if hasattr(response, 'content') else response
                print(f"ai_response : {ai_response}")
                extracted_json = self.extract_json_data(ai_response)
                print(f"extracted_json :{extracted_json}")
                ai_reply = extracted_json['reply']  ##### AI 응답
                self.save_context({"input": user_message}, {"output": ai_reply})
                print(f"AI : {ai_reply}")

                if extracted_json['completion']:
                    rec_order = extracted_json['order']
                    print(f"order_data: {rec_order}")
                    ShoppingCart.add_to_cart(rec_order)
                    print(ShoppingCart.cart)
                    self.handle_additional_requests()
                    break
                    
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            
    def execute_additional_order(self, user_message):
        try:
            while True:
                response = self.recommend_module.invoke(user_message)
                ai_response = response.content if hasattr(response, 'content') else response
                print(f"ai_response : {ai_response}")
                extracted_json = self.extract_json_data(ai_response)
                print(f"extracted_json :{extracted_json}")
                ai_reply = extracted_json['reply']
                self.save_context({"input": user_message}, {"output": ai_reply})
                print(f"AI : {ai_reply}")

                if extracted_json['completion']:
                    rec_order = extracted_json['order']
                    print(f"order_data: {rec_order}")
                    ShoppingCart.add_to_cart(rec_order)
                    print(ShoppingCart.cart)
                    self.handle_additional_requests()
                    break
                
                else:
                    user_message = input("입력:")
                
        except Exception as e:
            print(f"추가 주문 처리 중 오류가 발생했습니다: {e}")

# 빠른 주문 실행
order_module = OrderModule(claude)

# 메인 실행 부분
if __name__ == "__main__":
    try:
        order_module.execute_order()
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
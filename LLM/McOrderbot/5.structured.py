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
from langchain.schema import BaseOutputParser
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

# 메뉴 로드 클래스
class LoadMenu:
    def __init__(self):
        self.menu_list = []
        self.data = data

    def get_list(self):
        for item in self.data:
            menu_name = item['page_content']['name']
            self.menu_list.append(menu_name)
        return self.menu_list

# 장바구니 클래스
class ShoppingCart:
    cart = []
    
    # @classmethod
    # def add_to_cart(cls, order_data, quantity):
    #     mini_cart = []
    #     cart_item = {
    #         "name": order_data["name"],
    #         "quantity": quantity,
    #         "price": order_data["price"],
    #         "set_menu": order_data["set_menu"]
    #     }
    #     if order_data["set_menu"]:
    #         cart_item["set_price"] = order_data["set_price"]  # 나중에 set_price로 변경 예정
    #     mini_cart.append(cart_item)
    #     cls.cart.append(mini_cart)
    #     return cls.cart
    
    @classmethod
    def add_rec_menu_to_cart(cls, data):
        cls.cart += data

    @classmethod
    def calculate_total(cls):
        total = 0
        for order in cls.cart:
            for item in order:
                price = item.get('set_price') if item.get('set_menu') else item.get('price')
                quantity = item.get('num', 1)
                if price:
                    total += int(price) * int(quantity)
        return total
class IntentChain:
    def __init__(self):
        self.model = gpt4o_mini

    def parse_response(self, response_str):
        try:
            response_dict = json.loads(response_str)
        except json.JSONDecodeError:
            response_dict = {"error": "파싱 오류", "원본": response_str}
        return response_dict

    def invoke(self, question):
        chain = ChatPromptTemplate.from_messages([
        ("system", """
        아래 질문을 보고 사용자가 원하는 의도를 '주문', '취소', '결제', '추천'중 하나로 정확하게 분류하세요.

        분류 기준:
        - 주문: 특정 메뉴를 정확히 주문하려는 경우 (예: '빅맥 하나 주세요', '감자튀김 추가')
        - 취소: 이전에 진행된 주문을 취소하려는 경우 (예: '주문 취소해 주세요', '아까 주문한 것 취소하고 싶어요')
        - 결제: 주문 완료 후 결제를 요청하는 경우 (예: '결제할거야', '주문 완료')
        - 추천: 주문, 취소, 결제 이외의 경우

        질문의 분류와 해당 질문에 포함된 주요 키워드를 JSON 형식으로 출력하세요.

        조건:
        - 질문의 분류와 해당 질문에 포함된 주요 키워드를 딕셔너리 형식으로 출력하세요.
        - 분류가 '주문'일 때만 키워드를 출력하세요. 다른 분류에서는 키워드를 출력하지 마세요
        - 키워드는 버거, 사이드, 음료 등 메뉴명만 추출합니다.
        - 주문이 세트메뉴인 경우, 버거명에 세트메뉴 키워드를 포함하세요

        예시 출력: {{"분류": "주문", "키워드": ["불고기 버거 세트", "후렌치후라이", "콜라", "상하이 치킨 스낵랩"]}}
        예시 출력: {{"분류": "추천"}}
        """),
        ("ai","무엇을 도와드릴까요?"),
        ("human", "{input}")
        ])  | self.model | StrOutputParser()

        intent_result = chain.invoke({"input" :question})
        intent = self.parse_response(intent_result)
        print(f"intent:{intent}")
        category = intent['분류']
        keyword = intent.get('키워드', None)
        return category, keyword
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
        self.menu_list = LoadMenu().get_list()
        self.name_chain = self.create_name_chain()
        
    def create_name_chain(self):
        name_template = PromptTemplate.from_template("""
            사용자의 질문과 메뉴 리스트를 비교하여 사용자의 질문과 비슷한 메뉴가 있는지 확인하세요.
            비슷한 이름의 메뉴가 있다면 비슷한 메뉴를 모두 출력하세요
            비슷한 메뉴가 없다면 "없음"이라고 출력하세요.
            출력 시 메뉴 이름만 출력하세요

            <메뉴 리스트>
            {menu_list}
            </메뉴 리스트>

            <질문>
            {question}
            </질문>

            """)
        
        name_chain = RunnableLambda(lambda inputs: {"menu_list": self.menu_list, "question": inputs}) | name_template | self.model | self.parser
        return name_chain

    def invoke(self, question):
        return self.name_chain.invoke(question)

# 빠른 주문 모듈 클래스 추가
class FastOrderModule:
    def __init__(self, model, intent_chain, name_chain):
        self.intent_chain = intent_chain
        self.name_chain = name_chain
        self.menu_data = []
        self.recommend = False
        self.model = model
        self.confirm_menu = self.create_order_prompt()

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
                    "nutrition": item['page_content']['nutrition'],
                    "ingredients":item['page_content']['ingredients'],
                    "allergy_caution": item['page_content']['allergy_caution'],
                    "set_menu": False
                }
        return None
    
    def create_order_prompt(self):
        order_prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        메뉴목록에 있는 모든 메뉴에 대하여 메뉴명과 수량, 세트 여부, 세트 구성을 정확히 확인하세요.
        세트 메뉴일 경우 사이드와 음료 구성이 완전한지 확인하고, 부족할 경우 기본 구성을 안내하세요.
        기본 사이드는 미디엄 사이즈 프렌치 프라이, 기본 음료는 미디엄 사이즈 코카콜라입니다.
        사이드는 무료로 코울슬로로 변경할 수 있으며, 음료는 음료 카테고리에서 변경 가능하나 추가 요금이 발생할 수 있습니다.
        사이즈는 라지로 변경할 수 있으며, 이 경우 사이드와 음료도 모두 라지 사이즈로 변경됩니다.
        
        **주문 결과 출력:**
        - 모든 LLM 응답은 '주문 결과 예시'의 형식으로 출력됩니다.
        - '주문완료', 'LLM 응답', '주문내역'을 JSON 형식으로 출력하세요.

        **주문 결과 예시:**
        {{
            "completion": {{order_complete}},
            "reply" : "{{llm_response}}",
            "order" : 
            [
                [
                    {{"name": "불고기 버거", "num": "1", "price": "4200", "set_menu": True, "set_price": "4000"}},
                    {{"name": "후렌치후라이", "set_menu": True}},
                    {{"name": "코카콜라", "set_menu": True}}
                ],
                [
                    {{"name": "상하이 치킨 스낵랩", "num": "1", "price": "4000", "set_menu": False}},
                    {{"name": "아이스 드립 커피", "num": "1", "price": "2700", "set_menu":  False}}
                ]
            ]
        }}
        
        **메뉴목록**: {context}
        **사용자 주문** : {order}
        """),
        ("ai","주문 확인을 도와드리겠습니다."),
        ("human", "{question}")
        ])
        
        order_chain = {
            "context": RunnableLambda(lambda _: self.menu_data),
            "order": RunnablePassthrough(),
            "question": RunnablePassthrough()
        } | order_prompt_template | self.model
        return order_chain

    def invoke(self, user_message, category, intent_keyword):
        # if category == '주문':
        #     if intent_keyword:
        #         for keyword in intent_keyword: 
        #             menu_data_json = self.save_menu(keyword)
        #             print("메뉴가 존재합니다")
                    
        #             if menu_data_json is None:
        #                 similar_menu = self.name_chain.invoke(keyword)["answer"]
                        
        #                 if similar_menu != "없음":
        #                     menu_data_json = self.save_menu(similar_menu)
        #                     print("비슷한 이름의 메뉴가 존재합니다")

        #             if menu_data_json is None:
        #                 self.recommend = True
        #             else:
        #                 if "세트" in keyword:
        #                     menu_data_json['set_menu'] = True
        #                 self.menu_data.append(menu_data_json)
        #                 print("1")
        #     else:
        #         self.recommend = True
        #         print("2")
        # else: 
        #     self.recommend = True
        #     print("3")
        
        # if self.menu_data:
        #     print("4")
        #     while True:
        #         confirm = self.confirm_menu.invoke({"order": user_message})
        #         ShoppingCart.add_rec_menu_to_cart(confirm)
        #         print("5")
            

        # if self.recommend == True or category != '주문':
        #     print("6")
        #     return {
        #         "return": False,  # 추천 시스템으로 넘어가도록 표시
        #         "메시지": "상담시스템으로 연결해드리겠습니다."
        #     }
        # else:
        #     print("7")
        #     return {
        #         "return": True,  # 추천 시스템으로 넘어가도록 표시
        #         "메시지": "주문이 완료되었습니다."
        #     }  # 빠른 주문 처리 완료

        return {
            "return": False,  # 추천 시스템으로 넘어가도록 표시
            "메시지": "상담시스템으로 연결해드리겠습니다."
        }


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
        사용자가 비교를 요청하거나 추가적인 정보를 요구하는 메뉴 이름을 찾아 "검색_내용" 항목에 포함시키세요.
        사용자가 주문한다고 언급한 모든 메뉴 항목을 반드시 "검색_내용"에 포함시키세요.
        검색이 필요한 항목과 주문 내역 외에는 "검색_내용"에 포함시키지 마세요.

        이전 대화 내역:
        {chat_history}

        출력 형식:
        {{
            "검색_내용": [검색이 필요한 메뉴들, 주문한 메뉴들]
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
            1. 제공된 정보만으로 답변합니다.
            2. 질문과 가장 관련성이 높은 정보를 선택해 대답하세요.
            3. 메뉴 추천은 2개 이하로 제한하며, 신메뉴를 우선 추천하세요.
            4. 확실하지 않으면 "정확한 답변을 드리기 어렵습니다만, 추가로 확인 후 도와드리겠습니다."라고 답하세요.

            **주문 조건:**
            - 메뉴는 세트와 단품으로 구분됩니다.
            - 세트메뉴는 버거, 사이드, 음료가 포함되며, 미디엄 사이즈가 기본입니다.
            - 미디엄 사이즈 세트메뉴 가격은 context의 'set_price' 가격이며, 없을 경우 "죄송합니다, 세트 구성이 불가능한 항목입니다. 대신 단품으로 주문하시거나 다른 메뉴를 선택해 주세요."라고 안내하세요.
            - 라지 사이즈 세트메뉴로의 업그레이드는 800원 추가 요금이 부과됩니다.
            - 사이드는 후렌치 후라이 미디엄이 기본이며, 코울슬로로만 무료 변경 가능합니다.
            - 음료는 코카콜라 미디엄이 기본이며, 음료 변경 시 차액이 발생할 수 있습니다.

            **주문 절차:**
            - 주문이 완료되면 "추가로 원하시는 메뉴가 있으신가요? 결제를 원하시면 '결제할게'라고 말씀해 주세요."라고 안내하세요.
            - 결제 요청이 있으면 주문이 완료되었음을 표시합니다.
            
            **주문 결과 출력:**
            - 모든 LLM 응답을 반드시 '주문 결과 예시'의 JSON 형식으로 출력합니다.
            - 마크다운 표시는 하지 마세요.
            

            **주문 결과 예시:**
            {{
                "completion": {{order_complete}},
                "reply" : "{{llm_response}}",
                "order" : 
                [
                    [
                        {{"name": "불고기 버거", "quantity": "1", "price": "4200", "set_menu": True, "set_price": "4000"}},
                        {{"name": "후렌치후라이", "set_menu": True}},
                        {{"name": "코카콜라", "set_menu": True}},
                    ],
                    [
                        {{"name": "상하이 치킨 스낵랩", "quantity": "1", "price": "4000", "set_menu": False}}
                        {{"name : "오레오 맥플러리", "quantity": "1", "price": "4300", "set_menu":  False}}
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
        str_result = f"질문:{question}, 검색 내용: {', '.join(query['검색_내용'])}"
        print(str_result)
        return self.recommend_chain.invoke(str_result)

# 주문 모듈 클래스
class OrderModule: 
    def __init__(self, model):
        self.memory = self.get_shared_memory()  # 공유 메모리 생성
        self.window_memory = self.get_shared_window_memory()
        self.intent_chain = IntentChain()
        self.name_chain = NameChain(model)
        self.recommend_module = RecommendModule(model, self.memory, self.window_memory)  # 공유 메모리 전달
        self.fast_order_module = FastOrderModule(model, self.intent_chain, self.name_chain)
        self.messages = []

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

    def save_message(self, message, role):
        self.messages.append({"message": message, "role": role})
        
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
        
    def execute_order(self):
        try:
            self.recommend_module.memory.clear()

            while True:
                ####flutter에서 시작할때 "주문을 도와드리겠습니다. 무엇을 드시겠어요?"라고 출력되게 가능할까요?
                # 사용자로부터 입력 받기
                user_message = input("입력: ").strip()

                # 입력이 비어있는 경우 반복문을 계속 진행
                if not user_message:
                    continue

                self.save_message(user_message, "user")

                # 의도 분석
                intent_result = self.intent_chain.invoke(user_message)
                category, intent_keyword = intent_result
                print(f"category:{category}")
                print(f"intent_keyword: {intent_keyword}")

                # 빠른 주문 시도
                fast_order_result = self.fast_order_module.invoke(user_message,category, intent_keyword)
                fast_order_result = fast_order_result['return']
                print(f'fast_order_result : {fast_order_result}')

                # 빠른 주문이 실패했거나 추천이 필요한 경우
                if not fast_order_result:
                    print("추천주문모듈을 진행합니다.")
                    # JSON 주문 결과가 나올 때까지 추천주문 모듈 내에서만 대화 진행
                    while True:
                        response = self.recommend_module.invoke(user_message)
                        # JSON 데이터만 추출
                        ai_response = response.content if hasattr(response, 'content') else response
                        print(f"ai_response : {ai_response}")
                        extracted_json = self.extract_json_data(ai_response)
                        print(f"extracted_json :{extracted_json}")
                        ai_reply = extracted_json['reply']  #이 부분이 LLM응답부분
                        self.save_message(user_message, "user")
                        self.save_message(ai_reply, "assistant")
                        self.save_context({"input": user_message}, {"output": ai_reply})
                        print(f"AI : {ai_reply}")

                        if extracted_json['completion']:
                            rec_order = extracted_json['order']
                            print(f"order_data: {rec_order}")
                            ShoppingCart.add_rec_menu_to_cart(rec_order)
                            print(ShoppingCart.cart)
                            break
                        else:
                            user_message = input("입력:").strip()
                    if extracted_json['completion']:
                        total = ShoppingCart.calculate_total()
                        print({"전송": True, "메시지": f"주문이 완료되었습니다.{total}원 결제 도와드리겠습니다."})
                        break
                else:
                    # print({"전송": True, "메시지": f"주문이 완료되었습니다.{total}원 결제를 도와드리겠습니다."})
                    break
        except KeyboardInterrupt:
            print("\n프로그램이 사용자에 의해 종료되었습니다.")
        finally:
            print("주문 프로세스를 종료합니다. 감사합니다.")
                
# 빠른 주문 실행
order_module = OrderModule(claude)

# 메인 실행 부분
if __name__ == "__main__":
    try:
        order_module.execute_order()
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")
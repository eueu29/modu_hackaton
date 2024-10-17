gpt4o = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0.3)
claude = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

question = input("질문을 입력하세요: ")

### 빠른주문 모듈
recommend = False
set_menu = False
menu_data = []
shopping_cart = []
unrecognized_keywords = []

class ShoppingCart:
    def __init__(self):
        self.cart = []

    def add_to_cart(self, order_data, num):
        cart = {}
    cart["name"] = order_data["name"]
    cart["num"] = num
    cart["price"] = order_data["price"]
    cart["set_menu"] = order_data["set_menu"]
    if order_data["set_menu"]:
        cart["set_price"] = order_data["price"]  #나중에 set_price로 변경예정
    shopping_cart.append(cart)

intent = IntentChain(gpt4o).invoke(question)


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
        - 주문이 세트메뉴인 경우, 버거명에 세트메뉴 키워드를 포함하세요

        예시 출력: {{"분류": "주문", "키워드": "빅맥"}}
        예시 출력: {{"분류": "추천"}}
        """) | self.model | StrOutputParser()

        intent_result = chain.invoke(question)
        parse_intent = parse_response(intent_result)
        return parse_intent
    
    def intent(self, parsed_intent):
        category = parsed_intent.get('분류')
        intent_keyword = parsed_intent.get('키워드')
        return category, intent_keyword
    
    
# 이름추측 LLM
class CustomOutputParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        # 'answer' 키로 '정답'을 반환
        return {"answer": text.strip()}

class MenuList:
    def __init__(self):
        self.menu_list = []
    
    def get_list(self, file_dir):
        with open(file_dir, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for item in data:
            menu_name = item['page_content']['name']
            self.menu_list.append(menu_name)
        return self.menu_list

class NameChain:
    def __init__(self, model):
        self.model = model
        self.parser = CustomOutputParser()

    def invoke(self, question):
        menu_list = self.menu_list.get_list(file_dir)
        chain = PromptTemplate.from_template("""
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

            """) | self.model | self.parser
        
        name_result = chain.invoke(question)
        return name_result

class FastOrderModule:
    def __init__(self, intent_chain, name_chain):
        self.intent_chain = intent_chain
        self.name_chain = name_chain

    def process_keyword(self, keyword):
        keyword = ''.join(keyword)
        keyword = keyword.replace("세트", "").strip()
        keyword = keyword.replace("버거", "").strip()
        return keyword

    def save_menu(keyword):
        menu_data = None
        for item in data:
            if item['page_content']['name'] == keyword:
                menu_data = {
                    "name": item['page_content']['name'],
                    "category": item['page_content']['category'],
                    "description": item['page_content']['description'],
                    "price": item['page_content']['price'],
                    "nutrition": item['page_content']['nutrition'],
                    "origin": item['page_content']['origin']
                }
                break
        if menu_data:
            menu_data_json = json.dumps(menu_data, ensure_ascii=False, indent=4)
            return menu_data_json
        else:
            return None

    def find_menu(keyword, menu_list):
        for menu in menu_list:
            if keyword == menu.replace('버거', '').strip():
                return save_menu(menu)
        return None
    
    def fast_order(self):
        try:
            keyword_list = intent_keyword.split(',')
            for keyword in keyword_list:
                n_keyword = process_keyword(keyword)
                llm_keyword = process_keyword(name_chain.invoke(keyword)["answer"])
                menu_data_json = find_menu(n_keyword, menu_list) or find_menu(llm_keyword, menu_list)
                if menu_data_json is None:
                    unrecognized_keywords.append(keyword)
                    category = '추천'
                else:
                    menu_data_json = json.loads(menu_data_json) if isinstance(menu_data_json, str) else menu_data_json
                    if "세트" in keyword:
                        menu_data_json['set_menu'] = True
                    else:
                        pass
                    menu_data.append(menu_data_json)
            if menu_data:
                for order_data in menu_data:
                    num = input(f"{order_data['name']} 메뉴가 맞으신가요? 수량은 몇 개 드릴까요?")                
                    add_to_cart(order_data, num)
            if unrecognized_keywords:
                print(f"선택하신 메뉴 중 확인할 수 없는 항목이 있습니다: {unrecognized_keywords}. 추천 챗봇을 연결해 드리겠습니다.")
            print(f"shopping_cart : {shopping_cart}")
        except Exception as e:  # 기타 에러 처리
            print(f"에러 발생: {e}")
            category = '추천'


if category == '주문':
    FastOrderModule.fast_order()
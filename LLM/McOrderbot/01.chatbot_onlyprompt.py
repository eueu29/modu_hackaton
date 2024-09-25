import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda 
from langchain.callbacks.base import BaseCallbackHandler

# # .env 파일 로드
# load_dotenv()

# # LangSmith 설정
# handler = StdOutCallbackHandler()
# tracer = LangChainTracer(project_name="McDonald")
# callback_manager = CallbackManager([handler, tracer])

st.set_page_config(page_title="모두의점원 McDonald version", page_icon="🧊")
st.title("모두의점원 McDonald orderbot")
st.markdown(
    """
    반갑습니다! 맥도날드에서 행복한 경험을 드릴 수 있도록 도와드리겠습니다.
    """
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["first_encounter"] = True  # 첫 만남 플래그 설정

# Reset 버튼 추가
if st.button("Reset"):
    st.session_state["messages"] = []
    st.session_state["first_encounter"] = True  # 첫 만남 플래그 재설정
    st.experimental_rerun()

class ChatCallbackHandler(BaseCallbackHandler):
    
    message = ""

    def on_llm_start(self, *args, **kwargs):
        # 빈 위젯을 제공함
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        self.message = ""  # 메시지 초기화
    
    def on_llm_new_token(self, token:str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

# chatGPT는 streaming 지원함(응답을 실시간으로 받을 수 있음)
llm = ChatOpenAI(
    temperature=0.1, 
    streaming=True, 
    callbacks=[ChatCallbackHandler()],
)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    너는 맥도날드 햄버거가게에서 일하는 자동화된 주문시스템이다.\
    질문이 정확하지 않더라도 최대한 추측하여 근접한 선택지를 골라 답변한다.\
    모든 답변은 메뉴목록과 chat history만을 참고하여 답변한다.\
    답을 모르는 경우에는 지어내지 말고 '잘 모르겠습니다. 다시 말씀해주세요'하고 답변한다.\
    주문을 받는 순서는 다음과 같다.\
    제일 처음, 고객을 환영하는 인사를 한다.\
    그 다음, 고객의 주문을 받는다.\
    고객이 주문에 어려움을 겪을경우, 원하는 조건에 대해 묻는다.\
    고객이 원하는 조건이 있을 경우, 조건에 가장 근접한 메뉴를 추천해준다.
    \
    추천한 메뉴가 있다면, 질문이 추천한 메뉴와 연관이 있는 질문인지 먼저 파악한다.\
    부정적인 질문일 경우, 해당 내용은 포함되지 않는 메뉴를 추천한다.\
    하나의 주문이 완성되면 주문 내용을 다시 한번 확인한다.\
    버거메뉴를 선택 시 세트메뉴여부를 꼭 확인한다.\
    세트메뉴는 버거, 사이드, 음료 모두 선택해야 한다.\
    주문이 끝나면, 전체주문을 다시 한 번 확인하고 결제방법을 확인한다.\
    
    메뉴 목록:\
        버거: \
            이름|설명|식재료|중량|칼로리\
            맥크리스피 스리라차 마요|빠삭한 통닭다리살 케이준 패티에 스리라차 마요 소스를 더해 매콤 고소한 맛|난류,우유,대두,밀,토마토,닭고기,쇠고기|289g|663kcal\
            베토디 스리라차 마요|베이컨 토마토 디럭스에 스리라차 마요 소스를 더해 색다른 매콤함|난류,우유,대두,밀,돼지고기,토마토,쇠고기|	251g|621kcal\
            맥스파이시 상하이 버거|쌀가루가 더해져 더 바삭해진 닭가슴살 패티에 아삭아삭한 양상추와 신선한 토마토까지 더 바삭하고 맛있어진 NEW 맥스파이시 상하이 버거로 입맛도 기분도 화끈하게|난류,대두,밀,토마토,닭고기|246g|501kcal\
            토마토 치즈 비프 버거|신선한 토마토와 고소한 치즈버거의 조화|난류,우유,대두,밀,토마토,쇠고기|200g|403kcal\
            더블 쿼터파운더 치즈|좋아하는건 더 많이 즐기시라고, 두 배 이상 커진 파운드 비프 패티가 두 장 육즙이 풍부한 고기 맛을 그대로 살린 순 쇠고기 패티 두 장과 치즈 두 장이 입안 가득 완벽하게 조화되는 놀라운 맛|우유,대두,밀,토마토,쇠고기|275g|770kcal\
            쿼터파운더 치즈|쿼터파운더라는 이름에서 알 수 있듯이 두 배 이상 커진 파운드 비프와 부드러운 치즈 두 장의 환상궁합 두툼한 순 쇠고기 패티와 신선한 치즈의 풍부한 맛으로 세계적으로 사랑받고 있는 맥도날드의 대표적인 프리미엄 버거|우유,대두,밀,토마토,쇠고기|198g|536kcal\

    """),
    ("human", "{question}"),
])


send_message("무엇을 드시겠어요?", "ai", save=False)
paint_history()
message = st.chat_input("주문을 입력해주세요.")
if message:
    send_message(message, "human")
    chain =  prompt | llm
    response = chain.invoke({"question": message})
    send_message(response.content, "ai")








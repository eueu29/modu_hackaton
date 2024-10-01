import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda 
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StdOutCallbackHandler

# .env 파일 로드
load_dotenv()

# LangSmith 설정
handler = StdOutCallbackHandler()
tracer = LangChainTracer(project_name="McDonald")
callback_manager = CallbackManager([handler, tracer])

st.set_page_config(page_title="모두의점원 McDonald version", page_icon="🧊")
st.title("모두의점원 McDonald orderbot")
st.markdown(
    """
    반갑습니다! 맥도날드에서 행복한 경험을 드릴 수 있도록 도와드리겠습니다.
    """
)

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
    # callback_manager=callback_manager
)


loader = CSVLoader(file_path='files/MD_menu.csv', encoding='UTF-8')
cache_dir = LocalFileStore(f"/home/yoojin/ML/aiffel/HackaThon/nomad_fullstackGPT/.cache/embeddings/McDonald")
splitter = CharacterTextSplitter(separator=",", chunk_size=1000, chunk_overlap=0)

documents = loader.load()
texts =splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
vectorstore = FAISS.from_documents(texts, cached_embeddings)
retriever = vectorstore.as_retriever()
    
    
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

    
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

def paint_history():
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["first_encounter"] = True  # 첫 만남 플래그 설정

# Reset 버튼 추가
if st.button("Reset"):
    st.session_state["messages"] = []
    st.session_state["first_encounter"] = True  # 첫 만남 플래그 재설정
    st.cache_resource.clear()
    st.experimental_rerun()

@st.cache_resource
def get_memory():
    return ConversationBufferMemory(
        llm=llm,
        max_token_limit=200,
        return_messages=True,
        memory_key="chat_history"
    )

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are an automated ordering system working at a McDonald's burger restaurant. 
    Even if the customer's question is not clear, try to make the best guess and provide the closest available option. 
    All responses should be based solely on context and chat history. 
    If you don't know the answer, respond with "I didn’t quite understand." Do not make up answers.

    The process for taking an order is as follows:

    First, greet the customer warmly.
    Then, take the customer's order.
    If the customer has difficulty ordering, recommend new menu items.
    If the customer has specific preferences, recommend the menu item that best matches those preferences.
    Once an order is completed, confirm the order details again.
    When selecting a burger menu, always confirm whether the customer wants a combo meal.
    A combo meal includes a burger, a side, and a drink.
    After the order is finished, review the entire order once more and confirm the payment method.
    
    Context: {context}
    
    Chat History: {chat_history}
    """),
    ("human", "{question}"),
])

message = st.chat_input("Ask a question")
if message:
    st.session_state["messages"].append({"message": message, "role": "human"})
    paint_history()  # 사용자 메시지를 즉시 표시
    
    memory = get_memory()
    chain = {
        "context": retriever | RunnableLambda(format_docs),
        "chat_history": RunnableLambda(lambda _: memory.chat_memory),
        "question": RunnablePassthrough()
    } | prompt | llm
    
    with st.chat_message("ai"):
        response = chain.invoke(message, config={"callbacks": [tracer]})
        ai_response = response.content
        st.markdown(ai_response)  # st.write를 st.markdown으로 변경하여 메시지 박스에 출력
        # save_message(ai_response, "ai")  # AI 응답을 세션 상태에 저장
    
    memory.save_context({"input": message}, {"output": ai_response})
    st.experimental_rerun()
else:
    if st.session_state.get("first_encounter", False):
        st.session_state["messages"].append({"message": "주문을 도와드리겠습니다. 말씀해주세요.", "role": "ai"})
        st.session_state["first_encounter"] = False  # 첫 만남 플래그 해제
    paint_history()


import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda 
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from datetime import datetime
from langchain_community.document_loaders import JSONLoader
import json
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# 환경설정
load_dotenv()
logging.langsmith("TEST")

# Models
claude = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")
gpt4o = ChatOpenAI(model_name="gpt-4o-2024-08-06") 
gpt4o_mini = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18")
gpt_turbo = ChatOpenAI(model_name="gpt-3.5-turbo-0125")


st.set_page_config(page_title="Ensemble chatbot", page_icon="🤖")

st.title("모두의점원 McDonald orderbot")
st.markdown(
    """
    반갑습니다! 맥도날드에서 행복한 경험을 드릴 수 있도록 도와드리겠습니다.  
    LLM 모델에 기본 Template + RAG 사용중입니다.  
    현재 사용중인 LLM 모델은 claude 3.5 Sonnet 입니다.
    """
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
        with st.chat_message(message["role"]):
            st.markdown(message["message"])
            
# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["first_encounter"] = True  # 첫 만남 플래그 설정

# Reset 버튼 추가
if st.button("Reset"):
    st.session_state["messages"] = []
    st.session_state["first_encounter"] = True  # 첫 만남 플래그 재설정
    st.cache_resource.clear()
    # st.experimental_rerun() 대신 st.rerun() 사용
    st.rerun()

@st.cache_resource
def embed_file(file_dir):
    with open(file_dir, 'r', encoding='utf-8') as file:
        content = file.read()
        # JSON 객체들을 리스트로 변환
        json_objects = json.loads(content)

    # 파싱된 JSON 객체들을 문서로 변환
    docs = [
        Document(
            page_content=json.dumps(obj['page_content'], ensure_ascii=False),
        )
        for obj in json_objects
]

    
    file_name = file_dir.split("/")[-1]
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_name}")
    
    # CharacterTextSplitter를 사용하여 문서 분할
    text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=100, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    embeddings = UpstageEmbeddings(
        model="solar-embedding-1-large"
    )
    
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=cache_dir,
        namespace="solar-embedding-1-large",
    )

    vectorstore = FAISS.from_documents(
        split_docs,
        cached_embedder,
    )

    faiss = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    bm25 = BM25Retriever.from_documents(split_docs)
    bm25.k = 4

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25, faiss],
        weights=[0.3, 0.7],
        search_type="mmr",
    )
    
    return ensemble_retriever

file_path ="/home/yoojin/ML/aiffel/HackaThon/modu_hackaton/LLM/files/menu_1017.json"
retriever = embed_file(file_path)

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    당신은 맥도날드의 친절한 점원입니다.
    고객님의 주문을 도와드리세요. 나이와 상관없이 '고객님'이라고만 부르고, 모든 설명은 어린아이도 이해할 수 있게 해주세요. 
    대답은 최대한 간결하고 간단하게 설명하세요.
    친절하고 상냥하게 존댓말로 응대합니다.
    
    대답할 때는 다음 사항을 기억하세요:
    1. 제공된 정보만으로 답변합니다.
    2. 질문과 가장 관련성이 높은 정보를 찾아서 답변하세요.
    3. 메뉴 추천은 2개 이하로 제한하고, 신메뉴를 먼저 추천하세요.
    4. 항상 대화 기록을 고려해 대답하세요.
    5. 확실하지 않은 경우 "죄송합니다. 해당 질문에 대한 답을 찾을 수 없습니다."라고 답하세요.

    **주문 조건:**
    - 메뉴는 세트와 단품으로 구분됩니다.
    - 세트메뉴는 버거, 사이드, 음료가 포함되며, 미디엄 사이즈가 기본입니다.
    - 사이즈 업그레이드는 800원 추가 요금이 부과됩니다.
    - 사이드는 후렌치 후라이 미디엄이 기본이며, 코울슬로로 무료 변경 가능합니다.
    - 음료는 코카콜라 미디엄이 기본이며, 음료 변경 시 차액이 발생할 수 있습니다.
    - 세트메뉴 가격은 'set_burger_price'에 5600원을 더한 금액입니다.
    - 단품 후렌치 후라이는 스몰(2300원), 미디엄(3000원), 라지(4000원) 중 선택할 수 있습니다.
    - 단품 음료는 미디엄(2600원) 또는 라지(3100원)입니다.
    
    **주문 절차:**
    - 먼저 고객의 주문을 받고 메뉴가 완성되면 주문이 완료되었는지 한번만 물어보세요.
    - 메뉴의 주문이 완성되지 않은 경우 주문완료가 되었는지 묻지 마세요.
    - 주문이 모두 완료되면 주문 결과를 출력하세요.
            
    context: {context}
    """),
    ("human", "{question}"),
])

message = st.chat_input("질문을 입력해주세요")

if message:
    send_message(message, "human", save=True)
    
    chain = {
        "context": retriever,
        "question": RunnablePassthrough()
    } | prompt | claude
        
    with st.chat_message("ai"):
        response = chain.invoke(message)
        ai_response = response.content
        st.markdown(ai_response)
        
    save_message(ai_response, "ai")
    st.rerun()

else:
    if st.session_state.get("first_encounter", False):
        initial_message = "주문을 도와드리겠습니다. 말씀해주세요."
        st.session_state["messages"].append({"message": initial_message, "role": "ai"})
        st.session_state["first_encounter"] = False  # 첫 만남 플래그 해제
    paint_history()
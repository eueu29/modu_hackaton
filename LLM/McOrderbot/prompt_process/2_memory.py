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
    LLM 모델에 template + RAG + memory 사용중입니다.  
    현재 사용중인 LLM 모델은 claude 3.5 Sonnet 입니다.
    """
)

@st.cache_resource
def get_memory():
    return ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history"
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
    대답은 세 문장 이내로, 간결하고 친절하게 응대합니다.
    
    **응대 지침:**
    1. 제공된 정보만으로 답변합니다.
    2. 질문과 가장 관련성이 높은 정보를 선택해 대답하세요.
    3. 메뉴 추천은 2개 이하로 제한하며, 신메뉴를 우선 추천하세요.
    4. 항상 대화 기록을 최우선으로 고려해 대답하세요. 특히 이전 대화의 맥락을 고려하여, 질문이 불분명할 경우 이전 대화와 연결된 정보를 바탕으로 응답하세요.
    5. 확실하지 않으면 "정확한 답변을 드리기 어렵습니다만, 추가로 확인 후 도와드리겠습니다."라고 답하세요.
    6. 주문과 상관없는 질문을 할 경우, 응대를 한 뒤 다시 주문을 이어가도록 유도하세요.

    **주문 조건:**
    - 메뉴는 세트와 단품으로 구분됩니다.
    - 세트메뉴는 버거, 사이드, 음료가 포함되며, 미디엄 사이즈가 기본입니다.
    - 미디엄 사이즈 세트메뉴 가격은 context의 'set_price' 가격이며, 없을 경우 "죄송합니다, 세트 구성이 불가능한 항목입니다. 대신 단품으로 주문하시거나 다른 메뉴를 선택해 주세요."라고 안내하세요.
    - 라지 사이즈 세트메뉴로의 업그레이드는 800원 추가 요금이 부과됩니다.
    - 사이드는 후렌치 후라이 미디엄이 기본이며, 코울슬로로 무료 변경 가능합니다.
    - 음료는 코카콜라 미디엄이 기본이며, 음료 변경 시 차액이 발생할 수 있습니다.

    **주문 절차:**
    - 고객님의 주문을 받고 메뉴가 완성되면 추가 주문 여부를 물어보세요.
    - 주문이 미완료된 경우 추가 주문을 묻지 마세요.
    - 주문이 완료되면, 주문이 완료되었음을 표시합니다.
    
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
                {{"name": "불고기 버거", "num": "1", "price": "4200", "set_menu": True, "set_price": "6600"}},
                {{"name": "후렌치후라이", "num": "1", "price": "0", "set_menu": True, "set_price": "0"}},
                {{"name": "코카콜라", "num": "1", "price": "0", "set_menu": True, "set_price": "0"}},
                {{"name": "상하이 치킨 스낵랩", "num": "1", "price": "4000", "set_menu": False}}
            ]
        ]
    }}
    
    **Context:** {context}
    **Chat History:** {chat_history}
    """),
    ("human", "{question}"),
])
        
message = st.chat_input("질문을 입력해주세요")

if message:
    send_message(message, "human", save=True)
    
    memory = get_memory()
    chain = {
    "context": retriever,
    "chat_history": RunnableLambda(lambda _: memory.load_memory_variables({})["chat_history"]),
    "question": RunnablePassthrough()
    } | prompt | gpt4o
    
    with st.chat_message("ai"):
        response = chain.invoke(message)
        print(response)
        ai_response = response.content
        st.markdown(ai_response)
        
    memory.save_context({"input": message}, {"output": ai_response})
    save_message(ai_response, "ai")
    st.rerun()

else:
    if st.session_state.get("first_encounter", False):
        initial_message = "주문을 도와드리겠습니다. 말씀해주세요."
        st.session_state["messages"].append({"message": initial_message, "role": "ai"})
        memory = get_memory()
        memory.save_context({"input": ""}, {"output": initial_message})
        st.session_state["first_encounter"] = False  # 첫 만남 플래그 해제
    paint_history()
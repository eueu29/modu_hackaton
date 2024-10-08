import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from kiwipiepy import Kiwi
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from typing import List
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda 
from langchain.memory import ConversationBufferMemory

# 환경설정
load_dotenv()
logging.langsmith("Ensemble")

st.set_page_config(page_title="Ensemble chatbot", page_icon="🤖")

st.title("모두의점원 McDonald orderbot")
st.markdown(
    """
    반갑습니다! 맥도날드에서 행복한 경험을 드릴 수 있도록 도와드리겠습니다.
    """
)

# 한글 형태소 분석기 삽입 시 에러발생, 디버깅예정
# kiwi = Kiwi()

# def kiwi_tokenize(text):
#     return [token.form for token in kiwi.tokenize(text)]


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
    st.experimental_rerun()
    
@st.cache_resource
def get_memory():
    return ConversationBufferMemory(
        llm=llm,
        max_token_limit=200,
        return_messages=True,
        memory_key="chat_history"
    )


@st.cache_resource  # st.cache_data 대신 st.cache_resource 사용
def embed_file(file_dir):
    loader = TextLoader(file_dir)
    docs = loader.load()
    
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
        namespace="solar-embedding-1-large",  # Solar 임베딩 모델 이름으로 namespace 변경
    )

    faiss_vectorstore = FAISS.from_documents(
        split_docs,
        cached_embedder,
    )

    faiss = faiss_vectorstore.as_retriever(search_kwargs={"k": 4})
    
    kiwi_bm25 = BM25Retriever.from_documents(split_docs)
    # kiwi_bm25 = BM25Retriever.from_documents(split_docs, preprocess_func=kiwi_tokenize)
    kiwi_bm25.k = 4

    ensemble_retriever = EnsembleRetriever(
        retrievers=[kiwi_bm25, faiss],  # 사용할 검색 모델의 리스트
        weights=[0.3, 0.7],  # 각 검색 모델의 결과에 적용할 가중치
        search_type="mmr",  # 검색 결과의 다양성을 증진시키는 MMR 방식을 사용
    )
    
    return ensemble_retriever

retriever = embed_file('/home/yoojin/ML/aiffel/HackaThon/modu_hackaton/LLM/files/menu_1002_noallergy.txt')

llm = ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

# qa = RetrievalQA.from_chain_type(
#     llm = llm,
#     chain_type = "stuff",
#     retriever = retriever,
#     return_source_documents = True
# )

prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    너는 맥도날드의 자동화된 주문시스템이다. 
    고객의 질문이 명확하지 않아도 최대한 추측해서 가장 가까운 옵션을 제공해야 한다.
    모든 응답은 컨텍스트와 채팅 기록에 기반해야 한다.
    답변을 모르는 경우 "이해하지 못했어요. 다시 말씀해주세요"라고 응답해야 한다.
    
    주문 순서는 다음과 같다:
    1. 고객의 주문을 받는다. 고객이 주문에 어려움을 겪는 경우, 신제품 메뉴들을 추천한다.
    2. 고객이 특별한 취향이 있는 경우, 그에 가장 가까운 메뉴를 추천한다.
    3. 주문이 완료되면, 주문 상세를 다시 한 번 확인한다.
    4. 버거 메뉴를 선택할 때는 항상 고객이 세트 메뉴를 원하는지 확인한다.
    5. 세트 메뉴는 버거, 사이드, 음료가 포함된다.
    6. 주문이 완료되면, 전체 주문을 다시 한 번 검토하고 결제 방법을 확인한다.
    
    Context: {context}
    Chat History: {chat_history}
    """),
    ("human", "{question}"),
])


message = st.chat_input("질문을 입력해주세요")

if message:
    send_message(message, "human", save=True)
    
    memory = get_memory()
    chain = {
    "context": retriever,
    "chat_history": RunnableLambda(lambda _: memory.chat_memory),
    "question": RunnablePassthrough()
} | prompt | llm
    
    with st.chat_message("ai"):
        response = chain.invoke(message)
        ai_response = response.content
        st.markdown(ai_response)
        
    memory.save_context({"input": message}, {"output": ai_response})
    st.experimental_rerun()
else:
    if st.session_state.get("first_encounter", False):
        st.session_state["messages"].append({"message": "주문을 도와드리겠습니다. 말씀해주세요.", "role": "ai"})
        st.session_state["first_encounter"] = False  # 첫 만남 플래그 해제
    paint_history()
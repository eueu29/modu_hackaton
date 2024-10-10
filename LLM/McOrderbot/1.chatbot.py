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
from langchain_community.vectorstores import FAISS
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
    # st.experimental_rerun() 대신 st.rerun() 사용
    st.rerun()

@st.cache_resource
def get_memory():
    return ConversationBufferMemory(
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

retriever = embed_file('/home/yoojin/ML/aiffel/HackaThon/modu_hackaton/LLM/files/menu_1008.txt')

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
    당신은 맥도날드 가게의 점원입니다. 
    나이가 많은 노인 고객의 주문을 도와주세요.
    나이와 관련된 어떤한 호칭도 하지 말고 '고객님'으로 부르세요.
    어린아이도 이해하기 쉬운 단어로 설명해주세요.
    상냥하고 친절한 말투로 응대해주세요.
    간결하게 답변하며, 가능한 한 문장 이내로 대답해주세요.  
    
    질문을 이해하지 못한 경우, "다시 한번 말씀해주세요."라고 답하세요.
    
    반드시 제공된 정보만을 사용하여 질문에 대답하세요.
    질문과 가장 관련성이 높은 정보를 찾아서 답변하세요.
    충분한 정보를 바탕으로 정확히 답변할 수 있는 경우에만 답변하세요.

    답변이 확실하지 않을 경우, "죄송합니다. 해당 질문에 대한 답을 찾을 수 없습니다."라고 답하세요.
    
    주문 절차는 다음과 같습니다:
    1. 고객의 주문을 받습니다.
    2. 버거 메뉴를 선택할 때 항상 세트 메뉴를 원하는지 확인합니다. 
       세트 메뉴는 버거, 사이드, 음료가 포함됩니다. 
       기본 사이드는 프렌치프라이 1개, 음료는 콜라 1개입니다. 사이드와 음료는 변경 가능합니다.
       다른 메뉴로 변경 시 추가 금액을 안내해 드립니다.
    3. 추가 주문이 필요한지 묻고, 추가 주문이 없으면 최종 주문을 받습니다.
    4. 주문 완료 시 전체 주문을 검토하고 결제 방법을 확인합니다.
    5. 주문 결과를 json 형식으로 출력합니다.
    
    주문 결과 예시: 
    {{
        "주문 메뉴" : [
            {{
                "메뉴 이름" : "햄버거",
                "추가 옵션" : "프렌치프라이 1개, 콜라 1개",
                "단품 금액" : 0,
                "추가 금액" : 0,
                "주문 금액" : 0
            }},
            {{
                "메뉴 이름" : "치킨버거",
                "추가 옵션" : "프렌치프라이 1개, 환타 1개",
                "단품 금액" : 0,
                "추가 금액" : 0,
                "주문 금액" : 0
            }}
        ],
        "총 주문 금액" : 0,
        "결제 방법" : "카드"
    }}

    context: {context}
    chat history: {chat_history}
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
    } | prompt | llm
    
    with st.chat_message("ai"):
        response = chain.invoke(message)
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
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from src.Embeddings import get_embeddings
from src.DataLoader import load_document
from src.Splitter import split_documents
from src.Vectorstore import create_vectorstore,load_vectorstore,save_vectorstore
from pathlib import Path
st.set_page_config(
    page_title = "Research Paper Assistant"
)

st.title("Research Paper Assistant")
st.write("Share Your Doubts.")
FAISS_PATH = "FAISS_INDEX"
if  not Path(FAISS_PATH).exists():
    with st.spinner("Downloading FAISS index..."):
        loader = load_document()
        split = split_documents(loader)
        vectorstore = create_vectorstore(split)
        save_vectorstore(vectorstore,FAISS_PATH)

st.success("Successfully downloaded the FAISS index.")
vectorstore = load_vectorstore(FAISS_PATH)
retriever = vectorstore.as_retriever(search_kwargs ={"k":4})
llm = ChatGroq(model="openai/gpt-oss-120b",temperature=0)
contextualise_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Given the chat history and the latest user request,
            rewrite the latest request into a standalone request
            that can be understood without the chat history.

            Do not answer the request.
            Only rewrite it if necessary.
            Otherwise return it unchanged.
            """
        ),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualise_prompt
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a Researcher Scientist who has knowledge.
            

            Retrieved Paper:
            {context}
            """
        ),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ]
)
question_answer = create_stuff_documents_chain(llm,qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever,question_answer)
store = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

def get_session_id(session_id:str):
    return st.session_state.chat_history

conversation = RunnableWithMessageHistory(
    rag_chain,
    get_session_id,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


question = st.text_input("Describe about the Questions u wanna ask.")
if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    response = conversation.invoke(
        {"input":question},
        config = {
            "configurable":{"session_id":"poem session"}

        }
    )

    answer = response["answer"]
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role":"assistant",
            "content": answer
        }
    )


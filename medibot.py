import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

# Custom CSS for styling the UI
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
            font-family: Arial, sans-serif;
        }
        .stChatMessage {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            font-size: 16px;
        }
        .user { 
            background-color: #cce5ff;
            text-align: left;
        }
        .assistant {
            background-color: #d4edda;
            text-align: left;
        }
        .title {
            color: #007BFF;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
        }
        .sidebar-title {
            font-size: 22px;
            color: #17a2b8;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

def main():
    st.markdown("<h1 class='title'>🩺 Medical Chatbot</h1>", unsafe_allow_html=True)
    
    # Sidebar for chatbot details
    with st.sidebar:
        st.markdown("<h2 class='sidebar-title'>💡 About MediBot</h2>", unsafe_allow_html=True)
        st.info("""
        **MediBot** is your AI-powered medical assistant. Ask any health-related questions, and MediBot will provide reliable information based on available medical data.
        """)
        st.markdown("**🔹 How to Use:**")
        st.write("- Type your query in the chat box below.")
        st.write("- Click Enter to get a response.")
        st.write("- Responses are based on the provided medical database.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        role_class = "user" if message['role'] == 'user' else "assistant"
        st.markdown(f"""
        <div class='stChatMessage {role_class}'>
            <b>{'🧑‍⚕️ You:' if message['role'] == 'user' else '🤖 MediBot:'}</b>
            <br>
            {message['content']}
        </div>
        """, unsafe_allow_html=True)

    prompt = st.chat_input("Type your medical query here...")
    
    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.markdown(f"""
        <div class='stChatMessage user'>
            <b>🧑‍⚕️ You:</b><br>{prompt}
        </div>
        """, unsafe_allow_html=True)

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Don't provide anything out of the given context.
        
        Context: {context}
        Question: {question}
        
        Start the answer directly. No small talk, please.
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")
        
        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]

            st.session_state.messages.append({'role': 'assistant', 'content': result})
            st.markdown(f"""
            <div class='stChatMessage assistant'>
                <b>🤖 MediBot:</b><br>{result}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

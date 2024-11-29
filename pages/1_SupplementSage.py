# Imports
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Union
from tempfile import NamedTemporaryFile

import streamlit as st
from streamlit_option_menu import option_menu
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, UnstructuredHTMLLoader, YoutubeLoader, WebBaseLoader,Docx2txtLoader
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from supabase import create_client, Client
from st_login_form import login_form
from streamlit_supabase_auth import login_form as supabase_login, logout_button
# import os 
# os.environ['USER_AGENT'] = 'myagent'
# tf.keras.config.disable_interactive_logging()

# Setup: Supabase
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# User Authentication
session = supabase_login(
    url=SUPABASE_URL,
    apiKey=SUPABASE_KEY,
    providers=["github", "google"],
)
with st.sidebar:
    home=st.button("Home")
    if home:
        st.switch_page("Home.py")
    st.divider()
 # If the user is not logged in, stop the app
if not session:
    st.stop()
with st.sidebar:
#     if session:
    st.write(f"Welcome {session['user']['email']}")
    logout_button()

# Objects
# class KerasOCRLoader(BaseLoader):
#     """Custom Loader to extract text from images using Keras-OCR."""

#     def __init__(self, file_path: str):
#         """
#         Initialize the KerasOCRLoader with the file path.
#         :param file_path: Path to the image file.
#         """
#         self.file_path = file_path
#         self.pipeline = keras_ocr.pipeline.Pipeline()

#     def _extract_text_from_image(self) -> str:
#         """Uses Keras-OCR to extract text from the image."""
#         # Read and process the image
#         image = keras_ocr.tools.read(self.file_path)

#         # Perform OCR
#         predictions = self.pipeline.recognize([image])[0]

#         # Combine detected text into a single string
#         extracted_text = "\n".join([text for text, _ in predictions])
#         return extracted_text

#     def load(self) -> List[Document]:
#         """
#         Load and extract text from the image file using Keras-OCR.
#         :return: A list of Documents containing the extracted text.
#         """
#         try:
#             extracted_text = self._extract_text_from_image()
#             document = Document(
#                 page_content=extracted_text,
#                 metadata={"source": self.file_path}
#             )
#             return [document]
#         except Exception as e:
#             raise ValueError(f"Failed to process image file: {e}")

#Functions
# Initialize session state for additional inputs
if "web_inputs" not in st.session_state:
    st.session_state.web_inputs = []
if "youtube_inputs" not in st.session_state:
    st.session_state.youtube_inputs = []

# Function to dynamically add web loader inputs
def add_web_input():
    if len(st.session_state.web_inputs) < 2:
        st.session_state.web_inputs.append("")

# Function to dynamically add YouTube loader inputs
def add_youtube_input():
    if len(st.session_state.youtube_inputs) < 2:
        st.session_state.youtube_inputs.append("")

# Initialize session state for saved sources
if "saved_sources" not in st.session_state:
    st.session_state.saved_sources = []
# Initialize unique document IDs if not present
if "doc_ids" not in st.session_state:
    st.session_state.doc_ids = []

if "db" not in st.session_state:
    st.session_state.db= Chroma(
    collection_name="example_collection",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory="db"
)
# Function to remove a source
def remove_source(doc_id: str):
    # Remove the document by its unique ID from the vector store
    st.session_state.db.delete(ids=[doc_id])
    # Update session state by removing the deleted source
    st.session_state.saved_sources = [
        source for source in st.session_state.saved_sources if source["id"] != doc_id
    ]
    # Update document IDs
    st.session_state.doc_ids = [
        doc_id for doc_id in st.session_state.doc_ids if doc_id != doc_id
    ]

def load_sources(doc: Dict[str, Union[List[Any], List[str]]]):
    """
    Load documents grouped by type and process them.
    """
    loaders = {
        "txt": TextLoader,
        "pdf": PyPDFLoader,
        "web": WebBaseLoader,
        "youtube": YoutubeLoader,
        "docx":Docx2txtLoader,
        # "jpg":KerasOCRLoader
    }
    doc_id_counter = len(st.session_state.doc_ids)  # Start ID counter for new docs
    for file_type, sources in doc.items():
        loader_class = loaders.get(file_type)
        if not loader_class:
            print(f"No loader available for type: {file_type}")
            continue

        for source in sources:
            if isinstance(source, str):  # For web and YouTube links
                loader = loader_class(source)
                load=loader.load()
                source_name = load[0].metadata["title"] 
            else:  # For file uploads
                with NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
                    temp_file.write(source.getvalue())
                    loader = loader_class(temp_file.name)
                load=loader.load()
                source_name = source.name 
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=64)
            documents=text_splitter.split_documents(load)
            print(documents)
            # Generate unique IDs for documents
            new_doc_ids = [str(doc_id_counter + i) for i in range(len(documents))]
            doc_id_counter += len(documents)
            documents[0].metadata={"source": source_name, "type": file_type}
            print(documents)
            # Add documents with IDs and metadata
            st.session_state.db.add_documents(
                documents,
                ids=new_doc_ids,
            )

            # Save source details to session state
            st.session_state.saved_sources.append({"id": new_doc_ids[0], "name": source_name, "type": file_type})
            st.session_state.doc_ids.extend(new_doc_ids)

def group_sources(files: List, ytb_urls: str, web_urls: str) -> Dict[str, List]:
    """
    Group input sources into their respective types: text, PDF, YouTube, and web.
    """
    sources = {ext: [] for ext in ["txt", "pdf", "docx", "doc", "youtube", "web","jpg"]}
     # Add YouTube URLs
    for url in ytb_urls:
        if url.strip():
            sources["youtube"].append(url)

    # Add Web URLs
    for url in web_urls:
        if url.strip():
            sources["web"].append(url)
    for file in files:
        file_type = file.name.split('.')[-1].lower()
        if file_type in sources:
            sources[file_type].append(file)
    return sources

def add_college_questions_to_vector_store(selected_colleges, full_data):
    """
    Add college supplemental questions to the vector store.
    """
    documents = [
        Document(page_content=question, metadata={"college": college})
        for college in selected_colleges
        if college in full_data
        for question in full_data[college]
    ]
    if documents:
        st.session_state.db.add_documents(documents)
        print("Added college questions to vector store.")

def load_college_data(json_file):
    """
    Load JSON data and extract college names.
    """
    return list(json.loads(Path(json_file).read_text(encoding='utf-8')).keys())
def get_response(query, chat_history):
        
    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is."""
    llm = HuggingFaceEndpoint(
        repo_id="microsoft/Phi-3.5-mini-instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )   

    retriever = st.session_state.db.as_retriever()
  
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    print(history_aware_retriever)
    
    ### Answer question ### 
    qa_system_prompt = """You are a college essay assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    {context}"""
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain.invoke({"input":query,"chat_history":chat_history})
@st.cache_data
def get_full_data():
    full_data = json.loads(Path("college_data.json").read_text(encoding='utf-8'))
    return full_data
# Sidebar: User Inputs
with st.sidebar:
    st.title("🤖💬 Supplement Sage")
    os.environ['LANGCHAIN_API_KEY'] = st.text_input('Enter Langchain API key:', type='password')
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.text_input('Enter HuggingFace API key:', type='password')

    if not (os.environ['LANGCHAIN_API_KEY'] and os.environ["HUGGINGFACEHUB_API_TOKEN"]):
        st.warning('Please enter your credentials!', icon='⚠️')
    else:
        st.success('Credentials verified. Proceed!', icon='👉')

    st.title("Sources")
    files = st.file_uploader("Upload files", accept_multiple_files=True, type=["txt", "pdf", "docx", "csv"])
    # Prepare inputs for grouping
    # Base web and YouTube inputs
    web_url = st.text_input("Web Loader URL (default)")
    ytb_url = st.text_input("YouTube Loader URL (default)")

    # Buttons to add more inputs
    if st.button("Add another Web Loader URL"):
        add_web_input()

    if st.button("Add another YouTube Loader URL"):
        add_youtube_input()

    # Display additional inputs for Web Loader
    for i, value in enumerate(st.session_state.web_inputs):
        st.session_state.web_inputs[i] = st.text_input(f"Additional Web Loader URL {i+1}", value=value)

    # Display additional inputs for YouTube Loader
    for i, value in enumerate(st.session_state.youtube_inputs):
        st.session_state.youtube_inputs[i] = st.text_input(f"Additional YouTube Loader URL {i+1}", value=value)
    all_web_urls = [web_url] + st.session_state.web_inputs
    all_ytb_urls = [ytb_url] + st.session_state.youtube_inputs


    college_names = load_college_data("college_data.json")
    selected_colleges = st.multiselect("Search or select a college:", options=college_names)
    if st.button("See Questions"):
        for college in selected_colleges:
            st.subheader(college)
            for question in get_full_data().get(college, []):
                st.write(question)

    st.divider()

    # Save Sources
    if st.button("Save Sources"):
        grouped_sources = group_sources(files, all_ytb_urls, all_web_urls)
        if selected_colleges:
            add_college_questions_to_vector_store(selected_colleges, get_full_data())
        if grouped_sources:
            load_sources(grouped_sources)

        st.success("Sources saved and questions added to the vector store!")
    # Display saved sources in an expander
    with st.expander("Saved Sources", expanded=True):
        for source in st.session_state.saved_sources:
            col1, col2 = st.columns([4, 1])  # Create columns for source name and cancel button
            with col1:
                st.text(f"{source['name']} ({source['type']})")  # Display source name and type
            with col2:
                if st.button(f"Cancel", key=source["id"]):
                    remove_source(source["id"])  # Remove source when cancel button is clicked

# Main Application
st.title("Supplement Sage")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message("Human" if isinstance(message, HumanMessage) else "AI"):
        st.markdown(message.content)

if prompt := st.chat_input("Ask a question"):
    with st.chat_message("Human"):
        st.markdown(prompt)

    with st.chat_message("AI"):
        response = get_response(prompt, st.session_state.messages)["answer"]
        st.markdown(response)
        st.session_state.messages.extend([HumanMessage(prompt), AIMessage(response)])

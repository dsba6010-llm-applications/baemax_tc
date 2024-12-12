import json
import os
import sys
import time
from pathlib import Path

import docx
import PyPDF2
import streamlit as st
import toml
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from openai import OpenAI

config_path = os.path.join(os.path.dirname(__file__), ".streamlit", "config.toml")

# Load the configuration settings
config = toml.load(config_path)

run_on_save = config["server"]["runOnSave"]
theme_base = config["theme"]["base"]
primary_color = config["theme"]["primaryColor"]

# Import functions directly from core module
from core import (
    create_retriever,
    generate_document_summary,
    initialize_rag,
    initialize_vectorstore_with_metadata,
    process_uploaded_tc,
    read_file_content,
    retrieve_context_per_question,
    show_context,
    text_wrap,
)

temperature = 0.4
max_tokens = 250
top_p = 1.0
presence_penalty = 0.0
frequency_penalty = 0.0

load_dotenv()

# Get the directory of the current script
current_dir = Path(__file__).parent
image_path = current_dir / "assets" / "images" / "logo.png"
data_dir = current_dir / "data"
metadata_path = data_dir / "metadata.json"

# Create data directory if it doesn't exist
data_dir.mkdir(exist_ok=True)

# If metadata file doesn't exist, create it with empty list
if not metadata_path.exists():
    with open(metadata_path, "w") as f:
        json.dump([], f)

def reset_chat_state():
    """
    Reset the entire chat state, clearing all session variables
    and requiring a new API key.
    """
    # Clear all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Ensure API key and messages are reset
    st.session_state.api_key = ""
    st.session_state.messages = []
    st.session_state.retriever = None

    # Force rerun to show initial API key input
    st.rerun()

def load_metadata(metadata_path=metadata_path):
    """
    Loads metadata from the specified JSON file.
    """
    try:
        # Create data directory if it doesn't exist
        metadata_path.parent.mkdir(exist_ok=True)

        # If file doesn't exist, create it with empty list
        if not metadata_path.exists():
            with open(metadata_path, "w") as f:
                json.dump([], f)

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"Metadata handling error: {str(e)}")
        return []

def get_base64_image(image_path):
    import base64

    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    # Set page configuration
    st.set_page_config(page_title="Baemax T&C Analyzer")

    # Check if API key is set
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        # Show API key input screen
        col1, col2 = st.columns([4, 1])
        with col2:
            st.markdown(
                f"""
                <div style="text-align: right;">
                    <img src="data:image/jpg;base64,{get_base64_image(str(image_path))}" 
                         style="width: 200px; height: 200px; object-fit: contain;">
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.title("API Key Required")
        st.markdown("Please enter your OpenAI API Key to continue")
        
        api_key = st.text_input(
            "OpenAI API Key", 
            type="password", 
            key="initial_api_key_input"
        )
        
        if st.button("Submit API Key"):
            if api_key:
                try:
                    # Validate API key by creating a client
                    client = OpenAI(api_key=api_key)
                    client.models.list()  # Quick validation check
                    
                    # Store API key in session state
                    st.session_state.api_key = api_key
                    st.session_state.messages = []
                    
                    # Initialize other necessary session states
                    st.session_state.retriever = None
                    
                    # Refresh the app
                    st.rerun()
                except Exception as e:
                    st.error("Invalid API Key. Please check and try again.")
            else:
                st.warning("Please enter an API key.")
        
        # Stop further execution
        st.stop()

    # Rest of the original script remains the same...
    col1, col2 = st.columns([4, 1])
    with col2:
        st.markdown(
            f"""
            <div style="text-align: right;">
                <img src="data:image/jpg;base64,{get_base64_image(str(image_path))}" 
                     style="width: 200px; height: 200px; object-fit: contain;">
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <style>
            .title {
                font-size: 2.5rem;
                font-weight: 700;
                margin-bottom: 1rem;
            }
            .subheader {
                font-size: 1.2rem;
                color: #424242;
                line-height: 1.6;
                margin-bottom: 2rem;
            }
        </style>
        
        <div class="title">Baemax T&C</div>
        <div class="subheader">
            An App to help demystify terms and conditions agreements.<br>
            Please upload a document or ask about the terms we have in our database.
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Consolidated API key handling
    st.sidebar.title("Powered by OpenAI")

    # Model selection
    model_options = ["gpt-4", "gpt-4-turbo-preview"]
    selected_model = st.sidebar.selectbox("Select Model", model_options)
    st.session_state["openai_model"] = selected_model

    try:
        client = OpenAI(api_key=st.session_state.api_key)

        # Add a visual separator
        st.sidebar.markdown("---")

        # Add file upload section
        st.sidebar.header("What do you want to analyze?")
        uploaded_file = st.sidebar.file_uploader(
            "Upload T&C document",
            type=["pdf", "docx", "txt"],
            help="Supported formats: PDF, DOCX, TXT",
        )

        # Process uploaded file
        if uploaded_file:
            try:
                # Read content
                content = read_file_content(uploaded_file)

                # Process content and get summary
                chunks, summary = process_uploaded_tc(content, client, st.session_state)

                # Create embeddings and update vector store
                embeddings = OpenAIEmbeddings()
                custom_vectorstore = FAISS.from_texts(
                    chunks,
                    embeddings,
                    metadatas=[{"source": uploaded_file.name, "title": uploaded_file.name}]
                    * len(chunks),
                )

                # Merge with existing retriever if it exists
                if "retriever" in st.session_state and st.session_state.retriever:
                    existing_vectorstore = st.session_state.retriever.vectorstore
                    existing_vectorstore.merge_from(custom_vectorstore)
                else:
                    st.session_state.retriever = create_retriever(custom_vectorstore)

                st.sidebar.success(f"Successfully processed {uploaded_file.name}")

                # Display summary in main area
                st.markdown("### Document Summary")
                st.markdown(summary)

            except Exception as e:
                st.sidebar.error(f"Error processing file: {str(e)}")

    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        st.stop()

    # Initialize RAG system after API key verification
    if "retriever" not in st.session_state:
        init_message = st.empty()
        try:
            os.environ["OPENAI_API_KEY"] = st.session_state.api_key  # Set API key for RAG system
            st.session_state.retriever = initialize_rag()
            init_message.success("RAG system initialized successfully!")
            time.sleep(3)
            init_message.empty()
        except Exception as e:
            init_message.error(
                f"Failed to initialize RAG system. Error: {str(e)}\nContinuing without RAG functionality."
            )
            print(f"RAG initialization error: {str(e)}", file=sys.stderr)
            st.session_state.retriever = None

        # Add a visual separator
        st.sidebar.markdown("---")

    # Dropdown to display available T&Cs
    metadata = load_metadata()
    if metadata:
        titles = ["Browse Companies"] + [meta["title"] for meta in metadata]
        selected_tc = st.sidebar.selectbox(
            "Available Terms and Conditions", titles, index=0, key="browse_only"
        )
    else:
        st.sidebar.warning("No terms and conditions available.")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Reset button
    if st.sidebar.button("Reset Chat"):
        reset_chat_state()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            formatted_content = (
                message["content"].replace("\n*", "\n\n*").replace("\n-", "\n\n-")
            )
            st.markdown(formatted_content)
        else:
            st.markdown(message["content"])

# System prompt
system_prompt = """You are a helpful assistant with expertise in answering questions and providing insights. 
When relevant context is provided, prioritize using it to answer the query. 
If the question is unrelated to the context or no context is provided, respond based on your general knowledge.
If you do not know the answer, tell the user that their question is outside of your scope and do not lie.      

You can:
- Answer questions using the provided context.
- Summarize information from context or general knowledge.
- Assist with general queries unrelated to the context.

Maintain a conversational and approachable tone. Always aim to provide clear, concise, and accurate answers."""

# Accept user input
if prompt := st.chat_input("What would you like to know?", disabled=st.session_state.api_key == ""):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Initialize variables
    context = ""
    is_metadata_query = False

    # Check if retriever exists
    if "retriever" in st.session_state and st.session_state.retriever is not None:
        try:
            # Check if it's a metadata query
            if any(
                keyword in prompt.lower()
                for keyword in [
                    "terms and conditions",
                    "available",
                    "what terms",
                    "companies",
                ]
            ):
                is_metadata_query = True
                context = retrieve_context_per_question(
                    prompt, st.session_state.retriever
                )

                # Display metadata titles in the chat
                with st.chat_message("assistant"):
                    if context:
                        st.write("### Available Terms and Conditions:")
                        for title in context:
                            st.write(f"- {title}")
                    else:
                        st.write("No terms and conditions available.")

            # Retrieve context for general queries
            if not is_metadata_query:
                context = retrieve_context_per_question(
                    prompt, st.session_state.retriever
                )

        except Exception as e:
            st.warning(f"RAG retrieval error: {str(e)}")

    # Handle general queries or continue if context exists
    if not is_metadata_query:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                # Include context in the prompt if available
                system_prompt_with_context = system_prompt
                if context:
                    if isinstance(context, list):
                        context_str = "\n\n".join(context)
                    else:
                        context_str = context
                    system_prompt_with_context = (
                        f"{system_prompt}\n\nRelevant context:\n{context_str}"
                    )

                # Generate assistant response
                messages = [{"role": "system", "content": system_prompt_with_context}]
                messages.extend(
                    [
                        {"role": str(m["role"]), "content": str(m["content"])}
                        for m in st.session_state.messages
                    ]
                )

                stream = client.chat.completions.create(
                    model=st.session_state["openai_model"],
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                )

                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")

                # Format final response
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                full_response = "I apologize, but I encountered an error while processing your request."

            # Save assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

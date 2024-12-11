import asyncio
import io
import random
import textwrap
from pathlib import Path
from typing import List, Tuple

import docx
import numpy as np
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from openai import RateLimitError
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi


def read_file_content(uploaded_file) -> str:
    """
    Read content from various file formats (PDF, DOCX, TXT)

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        str: Extracted text content from the file
    """
    content = ""
    file_extension = Path(uploaded_file.name).suffix.lower()

    if file_extension == ".pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        for page in pdf_reader.pages:
            content += page.extract_text() + "\n"

    elif file_extension == ".docx":
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        content = "\n".join([paragraph.text for paragraph in doc.paragraphs])

    elif file_extension == ".txt":
        content = uploaded_file.getvalue().decode()

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return content


def generate_document_summary(content: str, client) -> str:
    """
    Generate a summary of the uploaded T&C document using OpenAI.

    Args:
        content (str): The full text content of the document
        client: OpenAI client instance

    Returns:
        str: A structured summary of the document
    """
    system_prompt = """You are a legal document analyzer specializing in Terms and Conditions analysis. 
    Provide a clear, structured summary of the document covering these key aspects:
    1. Document Overview (2-3 sentences)
    2. Key Terms and Definitions
    3. Main User Rights and Obligations
    4. Important Limitations or Restrictions
    5. Notable Clauses or Provisions
    
    Keep the summary concise but informative. Focus on the most important points that users should be aware of and do not lie."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Please analyze and summarize this Terms and Conditions document:\n\n{content}",
                },
            ],
            temperature=0.5,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def process_uploaded_tc(content: str, client, session_state) -> Tuple[List[str], str]:
    """
    Process uploaded T&C content and generate summary, with session state handling

    Args:
        content (str): The full text content of the document
        client: OpenAI client instance
        session_state: Streamlit session state object


    Returns:
        Tuple[List[str], str]: A tuple containing (chunks of text, document summary)
    """
    # Check if content is already in session state
    content_hash = hash(content)  # Create a hash of the content to use as identifier

    if "processed_documents" not in session_state:
        session_state.processed_documents = {}

    if content_hash in session_state.processed_documents:
        # Return cached results
        return session_state.processed_documents[content_hash]

    # Generate summary first
    summary = generate_document_summary(content, client)

    # Then process chunks for RAG
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunks = text_splitter.split_text(content)

    # Store results in session state
    session_state.processed_documents[content_hash] = (chunks, summary)

    return chunks, summary


def replace_t_with_space(list_of_documents):
    """
    Replaces all tab characters ('\t') with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """

    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace(
            "\t", " "
        )  # Replace tabs with spaces
    return list_of_documents


def text_wrap(text, width=120):
    """
    Wraps the input text to the specified width.

    Args:
        text (str): The input text to wrap.
        width (int): The width at which to wrap the text.

    Returns:
        str: The wrapped text.
    """
    return textwrap.fill(text, width=width)


def encode_pdf(path, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk.
        chunk_overlap: The amount of overlap between consecutive chunks.

    Returns:
        A FAISS vector store containing the encoded book content.
    """

    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore


def encode_from_string(content, chunk_size=1000, chunk_overlap=200):
    """
    Encodes a string into a vector store using OpenAI embeddings.

    Args:
        content (str): The text content to be encoded.
        chunk_size (int): The size of each chunk of text.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        FAISS: A vector store containing the encoded content.

    Raises:
        ValueError: If the input content is not valid.
        RuntimeError: If there is an error during the encoding process.
    """

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    try:
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents([content])

        # Assign metadata to each chunk
        for chunk in chunks:
            chunk.metadata["relevance_score"] = 1.0

        # Generate embeddings and create the vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        raise RuntimeError(f"An error occurred during the encoding process: {str(e)}")

    return vectorstore


def retrieve_all_metadata(vectorstore):
    """
    Retrieve all unique metadata titles from the vectorstore.

    Args:
        vectorstore: The FAISS vectorstore containing documents and metadata.

    Returns:
        list: A list of unique document titles.
    """
    try:
        if hasattr(vectorstore, "docstore") and vectorstore.docstore:
            documents = vectorstore.docstore._dict.values()  # Access stored documents
            titles = {doc.metadata.get("title", "Unknown") for doc in documents}
            return sorted(titles)
        else:
            raise ValueError("Vectorstore does not have a valid 'docstore' or metadata.")
    except Exception as e:
        raise ValueError(f"Metadata retrieval error: {e}")


def retrieve_context_per_question(question, retriever):
    """
    Retrieves metadata or context based on the user's query.

    Args:
        question (str): User's question.
        retriever: A retriever object.

    Returns:
        list: A list of metadata titles if the query is about terms, or context otherwise.
    """
    if any(keyword in question.lower() for keyword in ["what terms and conditions do you have access to", "what companies terms and conditions do you have access to"]):
        try:
            return retrieve_all_metadata(retriever.vectorstore)
        except Exception as e:
            raise ValueError(f"Error retrieving metadata: {e}")

    # Retrieve relevant context for general questions
    try:
        results = retriever.get_relevant_documents(question)
        return [doc.page_content for doc in results] if results else ["No relevant context found."]
    except Exception as e:
        raise ValueError(f"Error retrieving context: {e}")


class QuestionAnswerFromContext(BaseModel):
    """
    Model to generate an answer to a query based on a given context.

    Attributes:
        answer_based_on_content (str): The generated answer based on the context.
    """

    answer_based_on_content: str = Field(
        description="Generates an answer to a query based on a given context."
    )


def create_question_answer_from_context_chain(llm):
    # Initialize the ChatOpenAI model with specific parameters
    question_answer_from_context_llm = llm

    # Define the prompt template for chain-of-thought reasoning
    question_answer_prompt_template = """ 
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """

    # Create a PromptTemplate object with the specified template and input variables
    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    # Create a chain by combining the prompt template and the language model
    question_answer_from_context_cot_chain = (
        question_answer_from_context_prompt
        | question_answer_from_context_llm.with_structured_output(
            QuestionAnswerFromContext
        )
    )
    return question_answer_from_context_cot_chain


def answer_question_from_context(question, context, question_answer_from_context_chain):
    """
    Answer a question using the given context by invoking a chain of reasoning.

    Args:
        question: The question to be answered.
        context: The context to be used for answering the question.

    Returns:
        A dictionary containing the answer, context, and question.
    """
    input_data = {"question": question, "context": context}
    print("Answering the question from the retrieved context...")

    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}


def show_context(context):
    """
    Display the contents of the provided context list.

    Args:
        context (list): A list of context items to be displayed.

    Prints each context item in the list with a heading indicating its position.
    """
    for i, c in enumerate(context):
        print(f"Context {i + 1}:")
        print(c)
        print("\n")


def read_pdf_to_string(path):
    """
    Read a PDF document from the specified path and return its content as a string.

    Args:
        path (str): The file path to the PDF document.

    Returns:
        str: The concatenated text content of all pages in the PDF document.
    """
    # Open the PDF file in binary mode
    with open(path, "rb") as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        content = ""

        # Iterate through each page
        for page_num in range(len(pdf_reader.pages)):
            # Get the page object
            page = pdf_reader.pages[page_num]
            # Extract text from the page and append to content
            content += page.extract_text() + "\n"

    return content


def bm25_retrieval(
    bm25: BM25Okapi, cleaned_texts: List[str], query: str, k: int = 5
) -> List[str]:
    """
    Perform BM25 retrieval and return the top k cleaned text chunks.

    Args:
    bm25 (BM25Okapi): Pre-computed BM25 index.
    cleaned_texts (List[str]): List of cleaned text chunks corresponding to the BM25 index.
    query (str): The query string.
    k (int): The number of text chunks to retrieve.

    Returns:
    List[str]: The top k cleaned text chunks based on BM25 scores.
    """
    # Tokenize the query
    query_tokens = query.split()

    # Get BM25 scores for the query
    bm25_scores = bm25.get_scores(query_tokens)

    # Get the indices of the top k scores
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]

    # Retrieve the top k cleaned text chunks
    top_k_texts = [cleaned_texts[i] for i in top_k_indices]

    return top_k_texts


async def exponential_backoff(attempt):
    """
    Implements exponential backoff with a jitter.

    Args:
        attempt: The current retry attempt number.

    Waits for a period of time before retrying the operation.
    The wait time is calculated as (2^attempt) + a random fraction of a second.
    """
    # Calculate the wait time with exponential backoff and jitter
    wait_time = (2**attempt) + random.uniform(0, 1)  # nosec B311
    print(f"Rate limit hit. Retrying in {wait_time:.2f} seconds...")

    # Asynchronously sleep for the calculated wait time
    await asyncio.sleep(wait_time)


async def retry_with_exponential_backoff(coroutine, max_retries=5):
    """
    Retries a coroutine using exponential backoff upon encountering a RateLimitError.

    Args:
        coroutine: The coroutine to be executed.
        max_retries: The maximum number of retry attempts.

    Returns:
        The result of the coroutine if successful.

    Raises:
        The last encountered exception if all retry attempts fail.
    """
    for attempt in range(max_retries):
        try:
            # Attempt to execute the coroutine
            return await coroutine
        except RateLimitError as e:
            # If the last attempt also fails, raise the exception
            if attempt == max_retries - 1:
                raise e

            # Wait for an exponential backoff period before retrying
            await exponential_backoff(attempt)

    # If max retries are reached without success, raise an exception

    raise Exception("Max retries reached")

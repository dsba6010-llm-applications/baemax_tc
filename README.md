
# Click Here! 👉 [![Streamlit](https://docs.streamlit.io/logo.svg)](https://baymaxtc.streamlit.app/)

# 🚀 Baymax T&C: Your Friendly Terms of Service Summarizer

This repository contains the code for Baymax T&C, an LLM application designed to simplify complex Terms of Service (ToS) agreements.  We use Retrieval Augmented Generation (RAG) and large language models (LLMs) to provide clear and concise explanations of legal jargon, making ToS documents easier to understand.


## ℹ️ About Baymax T&C

Baymax T&C uses a powerful LLM and a large repository of ToS agreements. When you ask a question about a specific ToS, the application searches its document store for relevant information (using RAG). This information is then processed by the LLM, which translates the legal language into plain English and generates a concise explanation.

**Who can use Baymax T&C?**  Anyone! Consumers, employees, students, and anyone needing to understand online terms and conditions.

## ⚙️ Key Features

* **User-Friendly Interface:** Simple and intuitive.
* **Comprehensive Coverage:**  Covers a wide range of ToS agreements (currently includes: HireVue, Llama2, Mettler Toledo, Open AI, Pentair, Qdoba Mexican Eats, Bank of America, Ulta Beauty, Verizon, Truist - constantly expanding!).
* **Accurate and Reliable:**  Powered by an LLM trained on a vast dataset of legal documents.
* **Customizable:** Users can adjust the level of detail in the explanations.


## 🎯 Addressing Common ToS Problems

### Baymax T&C tackles the challenges of:

* **Legal jargon:** Complex legal language is translated into plain English.
* **Length and complexity:**  Long and complicated ToS documents are summarized.
* **Lack of clarity:**  Important terms and concepts are clearly explained.

# 🚀 RAG Evaluation and Implementation 

This repository contains a comprehensive system for building and evaluating Retrieval Augmented Generation (RAG) pipelines.  It leverages Langchain, OpenAI, FAISS, and deepeval to provide a robust and efficient solution.

## 📁 Project Overview

This project offers a complete workflow for creating a question-answering system that leverages external documents. The key components are:

* **📄 Document Processing (helper_functions.py):**  Handles the ingestion and processing of text files. This includes:
    * Text Encoding: Processes both PDFs and text strings.
    * Chunking: Splits text into overlapping chunks for efficient embedding.
    * Embedding: Generates OpenAI embeddings for each chunk.
    * Vector Store: Stores embeddings in a FAISS vector database for fast similarity search.  This allows for efficient retrieval of relevant information.

* **⚙️ RAG Pipeline (simple_rag.ipynb):**  Implements the core RAG pipeline using OpenAI:
    * Document Loading: Loads documents from specified folders and creates a FAISS index.
    * Retrieval: Uses FAISS for efficient retrieval of relevant document chunks.  Includes a fallback BM25 method.
    * Answer Generation: Uses an OpenAI LLM (specified in the notebook) to generate answers based on the retrieved context.

* **🤖 RAG Evaluation (evaluate_rag.py):**  Provides a thorough evaluation of the RAG system using the deepeval library. Metrics include:
    * Correctness (GEval): How factually accurate are the answers?
    * Faithfulness: How well do answers align with the information in the source document?
    * Answer Relevancy: How well do the answers address the specific question and its context within the source document?

## 💪 Technical Details
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)

This project uses:

* **Python 3.11:** Our runtime environment.
* **OpenAI API:**  Currently used for LLM processing.
* **FAISS:** For efficient vector search within the document store.
* **Deepeval:** For evaluating model performance (using Correctness, Faithfulness, and Contextual Relevancy metrics).
* Dependencies listed in requirements.txt

## 🛠️ Code Quality Tools
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

We utilize:

* **Black:** For code formatting.
* **isort:** For import sorting.
* **Bandit:** For security analysis.
* **Pre-commit:** To automate code quality checks before each commit.

## 🦾 What's Under The Hood
![Screenshot 2024-12-06 192958](https://github.com/user-attachments/assets/5715a14c-aa15-4380-9a02-7decbb7cabf7)

## 🐍 Setup

Create a conda environment:
```
conda create -n your_env python=3.11
```
Before running the code, ensure you have the necessary libraries installed. You can install them using pip:

```bash
pip install -r requirements.txt
```
Add secrets.toml file:

![Screenshot 2024-12-06 221129](https://github.com/user-attachments/assets/ba4c491d-2dd0-4672-9a78-59212f56e35e)

![image](https://github.com/user-attachments/assets/c0c2d0f0-a913-43ed-8e68-30079dfd96b3)

You will also need an OpenAI API key. Create a .env file in the root directory and add your key:

```bash
OPENAI_API_KEY=your_api_key_here
```
In the terminal: 
```bash
cd frontend/
```
```
streamlit run app.py
```

# 🤪 Quickstart Guide
[<img src="https://github.com/user-attachments/assets/25b1217d-4778-4790-909f-f2d95ba55822" width="200"/>](https://share.streamlit.io/)

![image](https://github.com/user-attachments/assets/dc17c7b8-7492-426d-8f88-a600737bf6dd)

![image](https://github.com/user-attachments/assets/438232b2-ce8e-4f6c-9390-183ea72351b3)


# 🤖 Usage
Ask Away! 

🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

📝 License

Copyrights of the Baymax T&C Team

Happy RAG-ing! 🎉

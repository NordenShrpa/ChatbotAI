# ChatbotAI
ChatbotAI using langchain streamlit and Gemini API - Nepali and English 

# Nepali PDF Chatbot (Powered by Gemini)

A Streamlit-based chatbot that enables users to interact with a Nepali PDF document by asking natural language questions. The chatbot extracts text from a PDF, splits the content into manageable chunks, creates semantic embeddings with SentenceTransformers, indexes the document using FAISS, and generates answers using the Gemini generative AI API—all while seamlessly handling translation between English and Nepali.

## Features

- **PDF Processing:**  
  Extract text from uploaded PDFs using PyPDF2.

- **Text Chunking:**  
  Split the extracted text into smaller, context-preserving segments.

- **Semantic Search:**  
  Generate embeddings with SentenceTransformers and index them using FAISS for efficient retrieval.

- **Multilingual Translation:**  
  Translate queries and results between English and Nepali leveraging the `googletrans` library.

- **Generative Answering:**  
  Build context-rich prompts and generate answers via the Gemini AI API.

- **Streamlit Interface:**  
  A clean, interactive web UI for uploading PDFs, asking questions, and displaying dynamic answers.




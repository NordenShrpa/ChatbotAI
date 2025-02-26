import os
import streamlit as st
from backend import (
    extract_text_from_pdf,
    split_text,
    generate_embeddings,
    create_faiss_index,
    translate_to_nepali,
    translate_to_english,
    retrieve_relevant_chunks,
    generate_answer_gemini
)


# Streamlit App Interface
st.set_page_config(page_title="Nepali PDF Chatbot", layout="wide")

st.title("Nepali PDF Chatbot (Powered by Gemini)")

# PDF Upload Section 
pdf_file = st.file_uploader("Microsoft Word - Sambidhan 2072 - Single page dumy.pdf", type=["pdf"])

if pdf_file:
    temp_pdf_path = "temp_uploaded.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    st.info("Extracting text from PDF...")
    nepali_text = extract_text_from_pdf(temp_pdf_path)
    
    if not nepali_text:
        st.error("Could not extract text from the PDF. Please try another file.")
    else:
        st.success("Text extracted successfully!")
        
        chunks = split_text(nepali_text)
        
        st.info("Generating embeddings...")
        embeddings = generate_embeddings(chunks)
        
        faiss_index = create_faiss_index(embeddings)
        st.success("Document processed and indexed!")

else:
    st.info("Please upload a PDF to begin.")

# Chat Section
if pdf_file and nepali_text:
    st.markdown("---")
    response_lang = st.radio("Select response language", options=["English", "Nepali"])
    user_question = st.text_input("Enter your question (in English)")
    
    if st.button("Ask") and user_question:
        query_for_search = translate_to_nepali(user_question)  # Translate to Nepali for retrieval
        
        retrieved_chunks = retrieve_relevant_chunks(query_for_search, faiss_index, chunks, k=3)
        
        translated_context_chunks = [translate_to_english(chunk) for chunk in retrieved_chunks]
        
        st.info("Generating answer using Gemini API...")
        answer = generate_answer_gemini(user_question, translated_context_chunks)
        
        final_answer = translate_to_nepali(answer) if response_lang == "Nepali" else answer
        
        st.markdown("**Answer:**")
        st.write(final_answer)

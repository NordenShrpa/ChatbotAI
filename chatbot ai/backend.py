import os
import google.generativeai as genai
from dotenv import load_dotenv
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
from googletrans import Translator

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("AIzaSyAu8l2IZOdljcjl53ex_wCEqx09tH6F3qU")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize translator
translator = Translator()



# PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None



# Text Splitting for Processing
def split_text(text, chunk_size=512):
    sentences = text.split("\n")
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# Generate Embeddings

def generate_embeddings(chunks):
    return embedding_model.encode(chunks)


# reate FAISS Index
def create_faiss_index(embeddings):
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


# Retrieve Relevant Chunks
def retrieve_relevant_chunks(query, index, chunks, k=3):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, k)
    
    retrieved_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return retrieved_chunks


# Translation Functions
def translate_to_nepali(text):
    try:
        return translator.translate(text, src="en", dest="ne").text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails


def translate_to_english(text):
    try:
        return translator.translate(text, src="ne", dest="en").text
    except Exception as e:
        print(f"Translation error: {e}")
        return text



#  Generate Answer
def generate_answer_gemini(user_question, context_chunks):
    context_text = "\n".join(context_chunks)
    
    prompt = f"Context:\n{context_text}\n\nQuestion: {user_question}\nAnswer:"

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        return response.text.strip() if response and response.text else "Sorry, I couldn't generate an answer."

    except Exception as e:
        return f"Error: {str(e)}"

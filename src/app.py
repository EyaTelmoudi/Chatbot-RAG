import psycopg
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# -------------------------
# Configuration
# -------------------------

# Groq API key (keep it in environment for safety)
client = Groq(api_key="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")

# PostgreSQL connection
conn = psycopg.connect("dbname=rag_chatbot user=postgres password=aya")

# SentenceTransformer embedding model (your original one)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Functions
# -------------------------

def embed_text(text: str):
    """Generate embeddings using SentenceTransformer."""
    return embedding_model.encode([text])[0].tolist()

def retrieve_documents(query_embedding, top_k=3):
    """Retrieve top-k most similar documents from PostgreSQL using pgvector."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT text, embedding <-> %s::vector AS distance
            FROM documents
            ORDER BY embedding <-> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        return [row[0] for row in cur.fetchall()]

def generate_answer(user_query: str):
    """RAG: combine retrieval + generator (Groq)"""
    # Step 1: embedding
    query_emb = embed_text(user_query)
    
    # Step 2: retrieve context
    context_docs = retrieve_documents(query_emb)
    context_text = "\n---\n".join(context_docs)
    
    # Step 3: prepare prompt
    prompt = f"""
You are a helpful assistant. Answer the user's question ONLY using the context below.

Context:
{context_text}

Question:
{user_query}
"""
    # Step 4: generate answer via Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    
    return response.choices[0].message.content

# -------------------------
# Command-line chatbot
# -------------------------

if __name__ == "__main__":
    print("RAG Chatbot is running! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        answer = generate_answer(user_input)
        print("Bot:", answer)

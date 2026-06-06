import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

genai.configure(api_key="gemini_api_key")

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("transformer_paper")

question = input("Ask a question: ")

question_embedding = embedding_model.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

context = "\n\n".join(results["documents"][0])
print(context[:1000])

prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

response = gemini_model.generate_content(prompt)

print("\nAnswer:\n")
print(response.text)
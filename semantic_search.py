from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini_model = genai.GenerativeModel("gemini-2.5-flash")
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read PDF
reader = PdfReader("data/transformer_paper.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text

# Create chunks
chunk_size = 500

chunks = [
    full_text[i:i + chunk_size]
    for i in range(0, len(full_text), chunk_size)
]

print("Chunks:", len(chunks))

# Create embeddings
chunk_embeddings = model.encode(chunks)

# Ask question
question = input("Ask a question: ")

# Create question embedding
question_embedding = model.encode([question])

# Compare with all chunks
scores = cosine_similarity(
    question_embedding,
    chunk_embeddings
)

import numpy as np

top_k = 3

top_indices = np.argsort(scores[0])[-top_k:]

context = "\n\n".join(
    [chunks[i] for i in top_indices]
)

print("Top Chunks:", top_indices)
print("\nRetrieved Context:\n")
print(context)
prompt = f"""
Answer ONLY using the provided context.

If the answer is not present in the context,
say: 'The document does not contain enough information.'

Context:
{context}

Question:
{question}
"""

response = gemini_model.generate_content(prompt)

print("\nAnswer:\n")
print(response.text)

print("\nTop Similarity Scores:")
for i in top_indices:
    print(f"Chunk {i}: {scores[0][i]:.4f}")
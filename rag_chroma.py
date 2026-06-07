import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini FIRST
genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Load models once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")


def ask_question(collection_name, question):

    collection = client.get_collection(collection_name)

    # Create question embedding
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    # Retrieve top chunks
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=10
    )
    print("\nRetrieved Documents:\n")

    for doc in results["documents"][0]:
        print(doc[:300])
        print("-" * 50)
    context = "\n\n".join(
        results["documents"][0]
    )

    prompt = f"""
Answer ONLY using the provided context.

If the answer is not present in the context,
say:
"I cannot find that information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = gemini_model.generate_content(
        prompt
    )

    return response.text


if __name__ == "__main__":

    collection_name = input(
        "Enter collection name: "
    )

    question = input(
        "Ask a question: "
    )

    answer = ask_question(
        collection_name,
        question
    )

    print("\nAnswer:\n")
    print(answer)
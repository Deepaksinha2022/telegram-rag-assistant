import chromadb
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("transformer_paper")

question = input("Ask a question: ")

# Create embedding
question_embedding = model.encode(question).tolist()

# Search
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

print("\nTop Results:\n")

for doc in results["documents"][0]:
    print(doc)
    print("\n" + "=" * 50 + "\n")
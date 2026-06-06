import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read PDF
reader = PdfReader("data/transformer_paper.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text

# Chunking
chunk_size = 500

chunks = [
    full_text[i:i + chunk_size]
    for i in range(0, len(full_text), chunk_size)
]

print("Chunks:", len(chunks))

# Create embeddings
embeddings = model.encode(chunks).tolist()

# ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="transformer_paper"
)

# Store chunks
for i, chunk in enumerate(chunks):
    collection.add(
        ids=[str(i)],
        embeddings=[embeddings[i]],
        documents=[chunk]
    )

print("Stored successfully!")
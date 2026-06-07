import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re

# Load model once
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")


def ingest_pdf(pdf_path):
    print(f"\nProcessing: {pdf_path}")

    # Collection name from PDF filename
    collection_name = os.path.splitext(
    os.path.basename(pdf_path)
)[0]

    collection_name = re.sub(
    r"[^a-zA-Z0-9_-]",
    "_",
    collection_name
)

    collection = client.get_or_create_collection(
        name=collection_name
    )

    # Clear existing data if re-indexing
    try:
        existing = collection.get()

        if existing["ids"]:
            collection.delete(
                ids=existing["ids"]
            )
            print("Old collection data cleared.")
    except Exception:
        pass

    # Read PDF
    reader = PdfReader(pdf_path)

    full_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            full_text += text

    # Chunking
    chunk_size = 1000

    chunks = [
        full_text[i:i + chunk_size]
        for i in range(
            0,
            len(full_text),
            chunk_size
        )
    ]

    print(f"Chunks created: {len(chunks)}")

    # Generate embeddings
    embeddings = model.encode(chunks).tolist()

    # Store in ChromaDB
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[str(i)],
            embeddings=[embeddings[i]],
            documents=[chunk]
        )

    print(f"Stored successfully in collection: {collection_name}")


if __name__ == "__main__":

    pdf_path = input(
        "Enter PDF path: "
    ).strip()

    if not os.path.exists(pdf_path):
        print("PDF file not found.")
    else:
        ingest_pdf(pdf_path)
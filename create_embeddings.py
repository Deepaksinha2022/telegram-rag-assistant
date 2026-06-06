from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read PDF
reader = PdfReader("data/transformer_paper.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text

# Create chunks
chunk_size = 1000

chunks = [
    full_text[i:i + chunk_size]
    for i in range(0, len(full_text), chunk_size)
]

print("Chunks:", len(chunks))

# Create embedding for first chunk
embedding = model.encode(chunks[0])

print("Embedding length:", len(embedding))
print("First 10 values:", embedding[:10])
from pypdf import PdfReader

reader = PdfReader("data/transformer_paper.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text

chunk_size = 1000

chunks = [
    full_text[i:i + chunk_size]
    for i in range(0, len(full_text), chunk_size)
]

print("Total Chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0])
from pypdf import PdfReader

reader = PdfReader("data/transformer_paper.pdf")

print("Pages:", len(reader.pages))

first_page = reader.pages[0]

text = first_page.extract_text()

print(text[:2000])
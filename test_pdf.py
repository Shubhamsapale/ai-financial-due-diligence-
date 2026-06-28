from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/transcripts/pdf1.pdf")

docs = loader.load()

print("Total Pages:", len(docs))

print("\nFirst Page Text:\n")
print(docs[0].page_content[:2000])
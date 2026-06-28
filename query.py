from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)






db = FAISS.load_local(

    "vector_db",

    embedding,

    allow_dangerous_deserialization=True

)

docs = db.similarity_search(

    "which year higher profit?",

    k=5

)

for d in docs:

    print(d.page_content)


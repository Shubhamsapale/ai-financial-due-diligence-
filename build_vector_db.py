import os
import pandas as pd

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#########################################################
# CONFIG
#########################################################

PDF_FOLDER = "Financial_rag/data/transcripts"

CSV_FILES = {
    "quarterly": "Financial_rag/data/quarterly.csv",
    "profit_loss": "Financial_rag/data/profit_loss.csv",
    "balance_sheet": "Financial_rag/data/balance_sheet.csv",
    "cashflow": "Financial_rag/data/cashflow.csv",
    "ratios": "Financial_rag/data/ratios.csv",
    "shareholding": "Financial_rag/data/shareholding.csv"
}

VECTOR_DB_PATH = "vector_db"

#########################################################
# LOAD PDFS
#########################################################

print("=" * 60)
print("Loading Earnings Call PDFs...")
print("=" * 60)

pdf_documents = []

pdf_files = sorted(os.listdir(PDF_FOLDER))

for pdf in pdf_files:

    if not pdf.endswith(".pdf"):
        continue

    path = os.path.join(PDF_FOLDER, pdf)

    print(f"Loading {pdf}")

    loader = PyPDFLoader(path)

    pages = loader.load()

    for page in pages:

        page.metadata["document_type"] = "transcript"

        page.metadata["file"] = pdf

    pdf_documents.extend(pages)

print(f"\nTotal PDF Pages : {len(pdf_documents)}")

#########################################################
# CHUNK PDFS
#########################################################

print("\nChunking PDFs...")

splitter = RecursiveCharacterTextSplitter(

    chunk_size=800,

    chunk_overlap=150

)

pdf_chunks = splitter.split_documents(pdf_documents)

print("PDF Chunks :", len(pdf_chunks))

#########################################################
# LOAD CSVs
#########################################################

financial_documents = []

print("\nLoading Financial CSVs...")

def dataframe_to_documents(df, statement_name):

    docs = []

    years = list(df.columns)[1:]

    for _, row in df.iterrows():

        metric = str(row.iloc[0]).strip()

        text = f"{statement_name}\n\n"

        text += f"Financial Metric : {metric}\n\n"

        for year in years:

            value = row[year]

            text += f"In {year}, {metric} was {value}.\n"

        docs.append(

            Document(

                page_content=text,

                metadata={

                    "document_type": "financial",

                    "statement": statement_name,

                    "metric": metric

                }

            )

        )

    return docs

for statement_name, path in CSV_FILES.items():

    print(f"Loading {statement_name}")

    df = pd.read_csv(path)

    docs = dataframe_to_documents(df, statement_name)

    financial_documents.extend(docs)

print("\nFinancial Documents :", len(financial_documents))

#########################################################
# MERGE ALL DOCUMENTS
#########################################################

documents = pdf_chunks + financial_documents

print("\nTotal Documents :", len(documents))

#########################################################
# LOAD EMBEDDING MODEL
#########################################################

print("\nLoading Embedding Model...")

embedding = HuggingFaceEmbeddings(

    model_name="BAAI/bge-large-en-v1.5"

)

#########################################################
# BUILD VECTOR DB
#########################################################

print("\nCreating FAISS Vector Database...")

db = FAISS.from_documents(

    documents,

    embedding

)

#########################################################
# SAVE
#########################################################

print("Saving Vector Database...")

db.save_local(VECTOR_DB_PATH)

print("\n" + "=" * 60)
print("Vector Database Created Successfully")
print("=" * 60)

print("PDF Chunks :", len(pdf_chunks))
print("Financial Rows :", len(financial_documents))
print("Total Stored :", len(documents))
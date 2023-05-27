import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv

from consts import INDEX_NAME

load_dotenv()

pinecone.init(
  api_key=os.getenv('PINECONE_API_KEY'),
  environment=os.getenv("PINECONE_ENVIRONMENT_REGION"),
)

print(os.getenv("PINECONE_API_KEY"))


def ingest_docs() -> None:
  # LOAD THE DOCS
  loader = ReadTheDocsLoader(
    path="langchain-docs/python.langchain.com/en/latest"
  )
  raw_documents = loader.load()
  print(f"loaded {len(raw_documents)}documents")

  # SPLIT THE DOCS
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
  )

  # SPLIT THE DOCS
  chunks = text_splitter.split_documents(documents=raw_documents)
  print(f"Splitted into {len(chunks)} chunks")

  # EMBED THE DOCS
  for doc in chunks:
    old_path = doc.metadata["source"]
    new_url = old_path.replace("langchain-docs", "https:/")
    doc.metadata.update({"source": new_url})

  print(f"Going to insert {len(chunks)} to Pinecone")
  embeddings = OpenAIEmbeddings()
  Pinecone.from_documents(
    chunks[3969:], embeddings, index_name=INDEX_NAME
  )
  print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
  ingest_docs()

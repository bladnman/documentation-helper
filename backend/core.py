import os
from typing import List, Dict, Any

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv

load_dotenv()

from consts import INDEX_NAME

pinecone.init(
  api_key=os.getenv('PINECONE_API_KEY'),
  environment=os.getenv("PINECONE_ENVIRONMENT_REGION"),
)


def run_llm(query: str, chat_history: List[Dict[str, Any]]) -> Any:
  embeddings = OpenAIEmbeddings()
  docsearch = Pinecone.from_existing_index(
    index_name=INDEX_NAME, embedding=embeddings
  )
  chat = ChatOpenAI(verbose=True, temperature=0)
  qa = ConversationalRetrievalChain.from_llm(
    llm=chat,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    return_source_documents=True
  )
  return qa(
    {
      "query": query,
      "question": query,
      "chat_history": chat_history}
  )


if __name__ == "__main__":
  resp = run_llm(query="What is a RetrievalQA chain?")
  print(resp)

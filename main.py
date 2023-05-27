import os
import time
from typing import Set

from backend.core import run_llm
import streamlit as st
from streamlit_chat import message

from dotenv import load_dotenv

load_dotenv()


def create_sources_string(source_urls: Set[str]) -> str:
  if not source_urls:
    return ""
  sources_lists = list(source_urls)
  sources_lists.sort()
  sources_string = "sources:\n"
  for i, source in enumerate(sources_lists):
    sources_string += f"{i + 1}. {source}\n"
  return sources_string


# initialize our page title
st.header("Langchain Udemy Course - Documentation Helper Bot")

# initialize our prompt
question = st.text_input("Prompt", placeholder="Questions?")

# initialize our state values
if "question_history" not in st.session_state:
  st.session_state["question_history"] = []
if "answer_history" not in st.session_state:
  st.session_state["answer_history"] = []
if "qa_history" not in st.session_state:
  st.session_state["qa_history"] = []

# ANSWER THE QUESTION
if question:
  with st.spinner("Let me ponder that..."):
    generated_response = run_llm(query=question)
    sources = set(
      [doc.metadata["source"] for doc in
       generated_response["source_documents"]]
    )

    answer = f"{generated_response['result']}\n\n{create_sources_string(sources)}"

    st.session_state["question_history"].append(question)
    st.session_state["answer_history"].append(answer)
    st.session_state["qa_history"].append(
      {
        "question": question,
        "answer": answer,
        "sources": sources,
        "key": time.time()
      }
    )

    question = ""

# PRINT THE LIST
if st.session_state["qa_history"]:
  for qa_item in st.session_state["qa_history"]:
    this_question = qa_item["question"]
    this_answer = qa_item["answer"]
    this_key = qa_item["key"]
    message(this_question, is_user=True, key=f"Q:{this_key}")
    message(this_answer, is_user=False, key=f"A:{this_key}")

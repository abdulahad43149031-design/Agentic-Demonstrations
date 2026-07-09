import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.load_local(
    "2-RagWithLangChain/faiss_index",
    embeddings = embedding_model,
    allow_dangerous_deserialization = True
)

llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key = os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_template(
    """
        You are a helpful customer support assistant.

        Answer ONLY using the context below.

        If the answer is not present in the context,
        say:

        "I couldn't find that information in the knowledge base."

        Context:
        {context}

        Question:
        {question}
    """
)

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    docs = vector_db.similarity_search(
        question,
        k = 2
    )

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # fill in prompt template

    messages = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(messages)

    print("\nAssistant:\n")
    print(response.content)





"""
File docs: query_vector_db.py

    1. Loads the FAISS database from disk
    2. Ask the user for a question
    3. Convert that question into an embedding
    4. Search the vector database for relevant chunks
    5. Print the retrived chunks

    ''' NO LLMS YET'''
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------- Load embeddings and chunks ---------------
folder_path = "2-RagWithLangchain/faiss_index"

vector_db = FAISS.load_local(
    folder_path = folder_path,
    embeddings = embedding_model,
    allow_dangerous_deserialization = True
)

print(len(embedding_model.embed_query("Document 1")))

print("Vector databases loaded successfully!")

# ----------- Ask the user a question -------------

question = input("\nAsk a question: ")

"""
    vectors_db.similarity_search()

    Internally it:

    1. Converts the question into an embedding.
    2. Compares it against every stored vector.
    3. Finds the closest vectors.
    4. Returns the corresponding Document objects.
"""

results = vector_db.similarity_search(
    query = question,
    k = 3
)

for i, document in enumerate(results):

    print(f"\nResult {i+1}")
    print("-" * 40)

    print(document.page_content)

"""
    FAISS never "understands" the text. It only searches vectors. LangChain then maps those vectors back to the original Document objects and returns their text.
"""
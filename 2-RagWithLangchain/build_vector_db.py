import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Reads a text file and converts it into LangChain Document objects
from langchain_community.document_loaders import TextLoader # we have PDFloader, CSVloader, WebBaseLoader (for html of sites), directoryLoader, JSONloader......

# Splits large documents into smaller ones
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Loads a pretrained embedding model
from langchain_huggingface import HuggingFaceEmbeddings

# FAISS vector database
from langchain_community.vectorstores.faiss import FAISS

# --------- Load the knowledge base -----------------

"""
    Document data structure has 2 major components:
        page_content: str
        metadata: dict
"""

loader = TextLoader("2-RagWithLangchain/knowledge_base.txt")
documents = loader.load()

print( f"\n{"=" * 25} Metadata {"=" * 25}\n")
print(documents[0].metadata)

print( f"\n{"=" * 25} Page {"=" * 25}")
print(documents[0].page_content)

# -------- Split into chunks -----------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 250,
    chunk_overlap = 50,
)

chunks = splitter.split_documents(documents) # an iterable

print("\n")
print("=" * 60)
print("Chunks")
print("=" * 60)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}")
    print("-" * 60)
    print(chunk.metadata)
    print("-" * 60)
    print(chunk.page_content)

# --------- Load embedding model -----------------

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

print("\nEmbedding model loaded")

# ---------- Build the FAISS vector database ---------

"""
    DocString for FAISS.from_documents(chunks, embedding model)

    1. Takes every chunk
    2. Extracts its text
    3. Sends the text to the embedding model
    4. Gets a vector 
    5. Stores the vector and the original chunk inside FAISS

    After this step, our knowledge base becomes searchable using semantic similarity
"""

vector_db = FAISS.from_documents(
    documents=chunks,
    embedding=embedding_model
)


print("FAISS vector database created successfully")

# -------- Save the database ------------------

"""
    If we don't save it, we'd have to rebuild it everytime we run our chatbot

    This creates a folder called: faiss_index/

    which contains everything needed to load the vector database later
"""

vector_db.save_local("2-RagWithLangchain/faiss_index")

print("Vector database saved as 'faiss_index/'")

"""
    index.faiss: Stores the vectors and the FAISS search index. This is what makes similarity search fast.

    index.pkl: Stores the original Document objects (their text and metadata), so when FAISS finds a matching vector, LangChain can return the corresponding chunk of text.
"""
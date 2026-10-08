from langchain_community.vectorstores import FAISS
from src.Embeddings import get_embeddings

def create_vectorstore(chunk):
    print("Chunk entering Faiss",len(chunk))
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(
        chunk,
        embeddings,
    )
    return vectorstore
def save_vectorstore(vectorstore,path:str="FAISS_INDEX"):
    vectorstore.save_local(path)

def load_vectorstore(path:str="FAISS_INDEX"):
    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore



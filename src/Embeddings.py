from langchain_huggingface import HuggingFaceEmbeddings
def get_embeddings():
    embeddings = HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
    return embeddings

get_embeddings()

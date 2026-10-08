from langchain_text_splitters import RecursiveCharacterTextSplitter
def split_documents(documents):
    spliter = RecursiveCharacterTextSplitter()
    chunk = spliter.split_documents(documents)
    return chunk


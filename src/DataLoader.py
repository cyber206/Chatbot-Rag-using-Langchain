from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
def load_document(data_path: str = "data"):
    documents = []
    path = Path(data_path).rglob("*.pdf")

    for file in path:
        print("Loading", file)

        loader = PyPDFLoader(str(file))
        doc = loader.load()

        documents.extend(doc)

    return documents
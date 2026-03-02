import os
from langchain_community.document_loaders import PyPDFLoader

DOCS_DIR = "./PDF"

import os
from langchain_community.document_loaders import PyPDFLoader

DOCS_DIR = "./PDF"

def approccio_base():
    documents = []
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(f"{DOCS_DIR}/{filename}")
            documents.extend(loader.load())
    
    # Usiamo repr() per mostrare i caratteri speciali invisibili!
    print("--- TESTO GREZZO CON CARATTERI NASCOSTI ---")
    print(repr(documents[0].page_content[:300]))
    
    print("\n--- METADATI SPORCHI ---")
    print(documents[0].metadata)

if __name__ == "__main__":
    approccio_base()


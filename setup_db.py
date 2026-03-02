import os
from dotenv import load_dotenv
from pymilvus import (
    connections, utility, FieldSchema, CollectionSchema, DataType, Collection
)

load_dotenv()

COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "documenti_aziendali")
VECTOR_FIELD_NAME = os.getenv("MILVUS_VECTOR_FIELD_NAME", "embedding")  
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # Dimensione per gemini-embedding-001
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_ALIAS = os.getenv("MILVUS_ALIAS", "default")

def setup_database():
    print(f"[INFO] Connessione a Milvus su {MILVUS_HOST}:{MILVUS_PORT} (alias={MILVUS_ALIAS})...")
    connections.connect(alias=MILVUS_ALIAS, host=MILVUS_HOST, port=MILVUS_PORT)
    
    if utility.has_collection(COLLECTION_NAME):
        print(f"[INFO] La collection '{COLLECTION_NAME}' esiste già: la elimino per ripartire da zero.")
        utility.drop_collection(COLLECTION_NAME)

    print(f"[INFO] Creazione dello schema per '{COLLECTION_NAME}'...")
    
    # 1. Definiamo i campi della nostra tabella vettoriale
    id_field = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64)
    embedding_field = FieldSchema(name=VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    
    # Metadati aziendali che ci torneranno utili per filtri avanzati
    autore_field = FieldSchema(name="Autore", dtype=DataType.VARCHAR, max_length=256)
    data_validita_field = FieldSchema(name="DataValidita", dtype=DataType.VARCHAR, max_length=32)
    dipartimento_field = FieldSchema(name="Dipartimento", dtype=DataType.VARCHAR, max_length=128)
    sources_field = FieldSchema(name="sources", dtype=DataType.VARCHAR, max_length=512)
    testo_field = FieldSchema(name="testo", dtype=DataType.VARCHAR, max_length=65535)

    # 2. Assembliamo lo schema
    schema = CollectionSchema(
        fields=[id_field, embedding_field, autore_field, data_validita_field, dipartimento_field, sources_field, testo_field],
        description="Collection base per RAG aziendale"
    )

    # 3. Creiamo la collection fisica
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    # 4. Creiamo l'indice vettoriale (Fondamentale per la ricerca veloce)
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    collection.create_index(field_name=VECTOR_FIELD_NAME, index_params=index_params)
    
    # 5. Carichiamo la collection in memoria, cosi e' subito pronta per insert/search.
    collection.load()
    
    print("[INFO] Database inizializzato correttamente. Schema, Indici e Load completati.")
    connections.disconnect(alias=MILVUS_ALIAS)

if __name__ == "__main__":
    setup_database()

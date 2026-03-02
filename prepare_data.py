import json
import os
import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = Path("./PDF")
OUTPUT_DIR = Path("./prepared")
OUTPUT_JSONL = OUTPUT_DIR / "chunks.jsonl"

@dataclass
class PreparedChunk:
    id: str
    text: str
    source: str
    dipartimento: str
    autore: str
    data_validita: str
    metadata: dict[str, Any]

def extract_dipartimento_from_filename(filename: str) -> str:
    """Estrae il dipartimento dal prefisso del file (es. HR_Procedura.pdf -> HR)"""
    parts = Path(filename).stem.split("_")
    return parts[0].strip() if parts else "Generale"

def load_and_clean_documents() -> list[Any]:
    print(f"[INFO] Lettura PDF dalla cartella: {DOCS_DIR.resolve()}")
    documents = []

    for filepath in DOCS_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(filepath))
        docs = loader.load()

        for doc in docs:
            # 1. Pulizia testo (rimuoviamo eventuali \n e spazi anomali)
            doc.page_content = doc.page_content.replace("\n", " ")
            doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()

            # 2. Arricchimento Metadati
            doc.metadata["source"] = filepath.name
            doc.metadata["Dipartimento"] = extract_dipartimento_from_filename(filepath.name)
            doc.metadata["Autore"] = "Azienda XYZ"
            doc.metadata["DataValidita"] = "31-12-2026"

        documents.extend(docs)

    return documents

def chunk_documents(documents: list[Any]) -> list[Any]:
    print("[INFO] Avvio chunking token-aware (tiktoken)...")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=300,
        chunk_overlap=20,
    )
    return splitter.split_documents(documents)

def print_overlap_demo(chunks: list[Any]):
    """Stampa la fine del primo chunk e l'inizio del secondo per mostrare l'overlap."""
    if len(chunks) >= 2:
        print("\n" + "="*60)
        print("🔍 DIMOSTRAZIONE VISIVA DELL'OVERLAP (SOVRAPPOSIZIONE)")
        print("="*60)
        print("\n--- FINE DEL CHUNK 1 ---")
        print(f"...{chunks[0].page_content[-200:]}")
        
        print("\n--- INIZIO DEL CHUNK 2 ---")
        print(f"{chunks[1].page_content[:200]}...")
        print("="*60 + "\n")

def to_prepared_chunks(chunks: list[Any]) -> list[PreparedChunk]:
    prepared = []
    for idx, chunk in enumerate(chunks):
        source = chunk.metadata.get("source")
        dipartimento = chunk.metadata.get("Dipartimento")
        autore = chunk.metadata.get("Autore")
        data_val = chunk.metadata.get("DataValidita")
        
        text = chunk.page_content
        # Generiamo un ID univoco basato su origine, indice e contenuto
        chunk_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}|{idx}|{text[:50]}"))

        prepared.append(
            PreparedChunk(
                id=chunk_id, text=text, source=source, 
                dipartimento=dipartimento, autore=autore, data_validita=data_val, 
                metadata=chunk.metadata
            )
        )
    return prepared

def print_dataset_statistics(prepared_chunks: list[PreparedChunk]) -> None:
    """Stampa metriche essenziali per validare il corpus prima dell'embedding."""
    total = len(prepared_chunks)
    if total == 0:
        print("[INFO] Nessun chunk disponibile per le statistiche.")
        return

    lengths = [len(c.text) for c in prepared_chunks]
    avg_len = sum(lengths) / total

    print("\n" + "-"*40)
    print("📊 ANALISI STATISTICA DEL CORPUS")
    print("-"*40)
    print(f"Totale Mattoncini (Chunks): {total}")
    print(f"Lunghezza Media:          {avg_len:.0f} caratteri")
    print(f"Chunk più grande:         {max(lengths)} caratteri")
    print(f"Chunk più piccolo:        {min(lengths)} caratteri")
    print("-" * 40 + "\n")

def main():
    raw_docs = load_and_clean_documents()
    chunks = chunk_documents(raw_docs)
    
    # Mostriamo la magia dell'overlap a terminale!
    print_overlap_demo(chunks)
    
    prepared = to_prepared_chunks(chunks)
    
    # Stampiamo le statistiche finali
    print_dataset_statistics(prepared)
    
    # Prepariamo la cartella di output e salviamo in JSONL
    OUTPUT_JSONL.parent.mkdir(exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for item in prepared:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            
    print(f"[INFO] Dataset pulito e salvato in {OUTPUT_JSONL}.")
    print("[INFO] Fine Modulo 1. Pronti per generare i Vettori!")

if __name__ == "__main__":
    main()
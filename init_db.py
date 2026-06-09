import os
import pandas as pd
import chromadb
from google import genai

SHEETS_CSV_URL = "https://docs.google.com/spreadsheets/d/1uRc9wZLzRKc703i75T6N0nGrralFn2yl6VYlu3Ovz10/export?format=csv"
MODEL_EMBEDDINGS = "gemini-embedding-001"

def inicializar_banco():
    if os.path.exists("./chroma_db"):
        print("Banco vetorial já existe. Pulando inicialização.")
        return

    print("Criando banco vetorial...")

    try:
        df = pd.read_csv(SHEETS_CSV_URL)
        print(f"Dataset carregado do Google Sheets: {len(df)} registros.")
    except Exception:
        csv_local = "dataset_ia.csv"
        if not os.path.exists(csv_local):
            print(f"Erro: não foi possível carregar o dataset.")
            return
        df = pd.read_csv(csv_local)
        print(f"Dataset carregado do arquivo local: {len(df)} registros.")

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.create_collection(name="tcc_rag_collection")

    for index, row in df.iterrows():
        texto = row["Information"]
        contexto = row["Context"]
        id_string = str(row["ID"])

        result = client.models.embed_content(
            model=MODEL_EMBEDDINGS,
            contents=texto
        )
        vetor = result.embeddings[0].values

        collection.add(
            embeddings=[vetor],
            documents=[texto],
            metadatas=[{"context": contexto}],
            ids=[id_string]
        )
        print(f"  [{index+1}/{len(df)}] {id_string} indexado.")

    print("Banco vetorial criado com sucesso!")

if __name__ == "__main__":
    inicializar_banco()

import os
import pandas as pd
import chromadb
from google import genai

def inicializar_banco():
    if os.path.exists("./chroma_db"):
        print("Banco vetorial já existe. Pulando inicialização.")
        return

    print("Criando banco vetorial nativo...")
    
    # IMPORTANTE: certifique-se de que o seu 'dataset_ia.csv' esteja na raiz do projeto
    csv_url = "dataset_ia.csv" 
    if not os.path.exists(csv_url):
        print(f"Erro: O arquivo {csv_url} não foi encontrado na raiz do projeto.")
        return

    df = pd.read_csv(csv_url)

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.create_collection(name="tcc_rag_collection")

    for index, row in df.iterrows():
        texto = row["Information"]
        contexto = row["Context"]
        id_string = str(row["ID"])
        
        result = client.models.embed_content(
            model='text-embedding-004',
            contents=texto
        )
        vetor = result.embeddings[0].values
        
        collection.add(
            embeddings=[vetor],
            documents=[texto],
            metadatas=[{"context": contexto}],
            ids=[id_string]
        )
    print("Banco vetorial criado com sucesso!")

if __name__ == "__main__":
    inicializar_banco()
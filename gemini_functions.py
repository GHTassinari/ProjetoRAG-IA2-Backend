import os
import chromadb
from google import genai
from google.genai import types

model_embeddings = 'gemini-embedding-001'
modelo_llm = 'gemini-3-flash-preview'

_chroma_client = None
_collection = None

def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
        _collection = _chroma_client.get_collection(name="tcc_rag_collection")
    return _collection

def gerarBuscarConsulta(consulta):
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    result = client.models.embed_content(model=model_embeddings, contents=consulta)
    vetor = result.embeddings[0].values

    resultado_busca = _get_collection().query(
        query_embeddings=[vetor],
        n_results=1
    )
    
    if resultado_busca['documents'] and len(resultado_busca['documents'][0]) > 0:
        return resultado_busca['documents'][0][0]
    return "Nenhum contexto relevante encontrado na base de dados."

def melhorarResposta(inputText):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=inputText)],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""
            Você é um assistente baseado em RAG (Retrieval-Augmented Generation).
            Utilize exclusivamente o conteúdo recuperado da base de conhecimento para responder à consulta do usuário.
            Considere a consulta e o contexto recuperado, e gere uma resposta clara, objetiva e coerente, reescrevendo as informações de forma natural sem copiar literalmente o texto original.
            Não invente informações que não estejam presentes no contexto fornecido e não apresente múltiplas opções de resposta.
            """),
        ],
    )

    response = client.models.generate_content(
        model=modelo_llm,
        contents=contents,
        config=generate_content_config,
    )

    return response.text
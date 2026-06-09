import sys
try:
    import pysqlite3
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Módulo SQLite atualizado com sucesso via pysqlite3-binary.")
except ImportError:
    print("pysqlite3 não encontrado. Mantendo o SQLite nativo do sistema.")

from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()

from init_db import inicializar_banco
inicializar_banco()

from gemini_functions import gerarBuscarConsulta, melhorarResposta

app = Flask(__name__)
app.json.ensure_ascii = False

CORS(app)

CHAVE_INTERNA_APP = os.getenv('APP_API_KEY', 'chave_padrao_teste')


@app.route("/")
def home():
    return jsonify({
        "status": "Servidor RAG Online",
        "ambiente": "Azure App Service" if os.getenv('WEBSITES_PORT') else "Local"
    }), 200


@app.route("/api", methods=["POST"])
def results():
    auth_key = request.headers.get("Authorization")
    if auth_key != CHAVE_INTERNA_APP:
        return jsonify({"error": "Unauthorized - API Key inválida ou ausente no cabeçalho"}), 401
        
    data = request.get_json(force=True)
    
    consulta = data.get("query")
    if not consulta:
        return jsonify({"error": "O campo 'query' é obrigatório no corpo da requisição."}), 400
        
    try:
        resultado_contexto = gerarBuscarConsulta(consulta)
        
        prompt_final = f"Consulta: {consulta} Resposta: {resultado_contexto}"
        resposta_refinada = melhorarResposta(prompt_final)
        
        return jsonify({"mensagem": resposta_refinada}), 200
        
    except Exception as e:
        return jsonify({
            "error": "Erro interno ao processar a requisição RAG.",
            "detalhes": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
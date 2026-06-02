from flask import Flask, jsonify, request
import google.generativeai as generativeai
from flask_cors import CORS
from dotenv import load_dotenv
import os
from gemini_functions import gerarBuscarConsulta, melhorarResposta

load_dotenv()
app = Flask(__name__)
CORS(app)

chave_gemini = os.getenv('GEMINI_API_KEY')
generativeai.configure(api_key=chave_gemini)

CHAVE_INTERNA_APP = os.getenv('APP_API_KEY')

@app.route("/")
def home():
    return jsonify({"status": "Servidor RAG Online na Azure"}), 200

@app.route("/api", methods=["POST"])
def results():
    auth_key = request.headers.get("Authorization")
    if auth_key != CHAVE_INTERNA_APP:
        return jsonify({"error": "Unauthorized - API Key inválida ou ausente"}), 401
        
    data = request.get_json(force=True)
    
    consulta = data.get("query")
    if not consulta:
        return jsonify({"error": "O campo 'query' é obrigatório no corpo da requisição."}), 400
        
    resultado = gerarBuscarConsulta(consulta)
    prompt = f"Consulta: {consulta} Resposta: {resultado}"
    response = melhorarResposta(prompt)
    
    return jsonify({"mensagem": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
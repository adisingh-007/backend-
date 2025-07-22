from flask import Flask, request, jsonify
from flask_cors import CORS
import main

app = Flask(__name__)
CORS(app, origins=["https://comply-ai-gy6l.vercel.app"])

db_cache = {}

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    try:
        file_path = main.upload_pdf(file)
        db = main.create_vector_store(file_path)
        db_cache[file.filename] = db
        return jsonify({"status": "ok", "filename": file.filename})
    except Exception as e:
        return jsonify({"error": f"Error uploading or processing file: {str(e)}"}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    filename = data.get("filename")
    question = data.get("question")
    if not filename or not question:
        return jsonify({"error": "Missing filename or question"}), 400
    db = db_cache.get(filename)
    if db is None:
        return jsonify({"error": "Document not found or not indexed. Please upload again."}), 400
    try:
        docs = main.retrieve_docs(db, question)
        result = main.question_pdf(question, docs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Error generating answer: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(port=5000)

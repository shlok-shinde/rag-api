from fastapi import FastAPI
import chromadb
import ollama
import os

app = FastAPI()
chroma = chromadb.PersistentClient(path="./db")
collection = chroma.get_or_create_collection("docs")

@app.post("/query")
def query(q: str):
    results = collection.query(query_texts=[q], n_results=1)
    context = results["documents"][0][0] if results and results.get("documents") and len(results["documents"]) > 0 and len(results["documents"][0]) > 0 else ""
    
    # Check if mock mode is enabled
    use_mock = os.getenv("USE_MOCK_LLM", "0") == "1"
        
    if use_mock:
        # Return retrieved context directly (deterministic!)
        return {"answer": context}
    else:
        # Use real LLM (production mode)
        answer = ollama.generate(
            model="tinyllama",
            prompt=f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer clearly and concisely:"
        )
        return {"answer": answer["response"]}

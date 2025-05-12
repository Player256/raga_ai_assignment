# agents/retriever_agent/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Retriever Agent")

# In-memory vector store for demo
documents = []
embeddings = []
model = SentenceTransformer("all-MiniLM-L6-v2")
index = None


class IndexRequest(BaseModel):
    docs: List[str]


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/index")
def index_docs(request: IndexRequest):
    global documents, embeddings, index
    docs = request.docs
    docs_embeddings = model.encode(docs)
    documents.extend(docs)
    if embeddings:
        embeddings.extend(docs_embeddings)
    else:
        embeddings = list(docs_embeddings)
    # Build/rebuild FAISS index
    emb_array = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(emb_array.shape[1])
    index.add(emb_array)
    return {"status": "indexed", "num_docs": len(documents)}


@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    global documents, embeddings, index
    if not index or not documents:
        return {"error": "No documents indexed."}
    query_emb = model.encode([request.query]).astype("float32")
    D, I = index.search(query_emb, request.top_k)
    results = [
        {"doc": documents[i], "score": float(D[0][idx])} for idx, i in enumerate(I[0])
    ]
    return {"results": results}

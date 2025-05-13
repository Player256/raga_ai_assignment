# agents/retriever_agent/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retriever Agent")

# Get persistence path from environment
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./faiss_index_store")  # Default path

# Global variables for the model and vector store
# Initialize lazily or handle potential errors during startup
model: Optional[Embeddings] = None
vectorstore: Optional[FAISS] = None


def get_embedding_model():
    """Initialize and return the Sentence Transformer model."""
    global model
    if model is None:
        try:
            # Using a common, efficient model
            model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Sentence Transformer model loaded.")
        except Exception as e:
            logger.error(f"Error loading Sentence Transformer model: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}")
    return model


def get_vectorstore():
    """Load or create the FAISS vector store."""
    global vectorstore
    if vectorstore is None:
        embedding_model = get_embedding_model()  # Ensure model is loaded
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                # Load existing index
                vectorstore = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    embedding_model,
                    allow_dangerous_deserialization=True,
                )
                logger.info(f"FAISS index loaded from {FAISS_INDEX_PATH}")
            except Exception as e:
                logger.error(f"Error loading FAISS index from {FAISS_INDEX_PATH}: {e}")
                # Decide if you want to start fresh or fail
                vectorstore = FAISS.from_texts(
                    ["Initial dummy document"], embedding_model
                )  # Start fresh
                vectorstore.save_local(FAISS_INDEX_PATH)
                logger.warning("Creating a new FAISS index due to loading error.")
        else:
            # Create new index with a dummy document
            # LangChain FAISS needs at least one document to initialize properly
            vectorstore = FAISS.from_texts(["Initial dummy document."], embedding_model)
            vectorstore.save_local(FAISS_INDEX_PATH)
            logger.info(f"New FAISS index created and saved to {FAISS_INDEX_PATH}")

    return vectorstore


class IndexRequest(BaseModel):
    docs: List[str]


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/index")
def index_docs(request: IndexRequest):
    """Indexes documents into the persistent FAISS vector store."""
    try:
        vecstore = get_vectorstore()
        embedding_model = get_embedding_model()

        if not request.docs:
            logger.warning("No documents provided for indexing.")
            return {
                "status": "no documents provided",
                "num_docs": vecstore.index.ntotal,
            }

        logger.info(f"Indexing {len(request.docs)} documents.")
        # Add documents to the vector store
        vecstore.add_texts(request.docs)

        # Save the updated index to disk
        vecstore.save_local(FAISS_INDEX_PATH)
        logger.info(
            f"Index updated and saved to {FAISS_INDEX_PATH}. Total documents: {vecstore.index.ntotal}"
        )

        return {"status": "indexed", "num_docs": vecstore.index.ntotal}

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    """Retrieves top_k relevant documents for a given query from the vector store."""
    try:
        vecstore = get_vectorstore()

        # LangChain FAISS needs at least one document indexed beyond the dummy
        if (
            vecstore.index.ntotal <= 1
            and "Initial dummy document." in vecstore.docstore._dict.values()
        ):
            logger.warning("Vector store contains only the initial dummy document.")
            return {"results": [], "error": "No meaningful documents indexed yet."}

        logger.info(
            f"Retrieving documents for query: '{request.query}' (top_k={request.top_k})"
        )
        # Perform similarity search
        results = vecstore.similarity_search_with_score(request.query, k=request.top_k)

        # Format results
        formatted_results = [
            {"doc": doc.page_content, "score": float(score)} for doc, score in results
        ]
        logger.info(f"Retrieved {len(formatted_results)} results.")

        return {"results": formatted_results}

    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")


# Optional: Add startup event to pre-load the model and index
# @app.on_event("startup")
# async def startup_event():
#     get_embedding_model()
#     get_vectorstore() # This will load or create the index on startup

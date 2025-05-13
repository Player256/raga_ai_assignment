from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional


from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import logging

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retriever Agent")

FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH", "/app/faiss_index_store"
)  # Path inside container

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


embedding_model_instance: Optional[Embeddings] = None
vectorstore_instance: Optional[FAISS] = None


def get_embedding_model() -> Embeddings:
    """Initialize and return the SentenceTransformerEmbeddings model."""
    global embedding_model_instance
    if embedding_model_instance is None:
        try:
            logger.info(
                f"Loading SentenceTransformerEmbeddings with model: {EMBEDDING_MODEL_NAME}"
            )

            embedding_model_instance = SentenceTransformerEmbeddings(
                model_name=EMBEDDING_MODEL_NAME
            )
            logger.info(
                f"SentenceTransformerEmbeddings model '{EMBEDDING_MODEL_NAME}' loaded successfully."
            )
        except Exception as e:
            logger.error(
                f"Error loading SentenceTransformerEmbeddings model '{EMBEDDING_MODEL_NAME}': {e}",
                exc_info=True,
            )
            raise RuntimeError(f"Could not load embedding model: {e}")
    return embedding_model_instance


def get_vectorstore() -> FAISS:
    """Load or create the FAISS vector store."""
    global vectorstore_instance
    if vectorstore_instance is None:
        emb_model = get_embedding_model()
        if os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH):
            try:
                logger.info(
                    f"Attempting to load FAISS index from {FAISS_INDEX_PATH}..."
                )
                vectorstore_instance = FAISS.load_local(
                    FAISS_INDEX_PATH,
                    emb_model,
                    allow_dangerous_deserialization=True,
                )
                logger.info(
                    f"FAISS index loaded from {FAISS_INDEX_PATH}. Documents: {vectorstore_instance.index.ntotal if vectorstore_instance.index else 'N/A'}"
                )
            except Exception as e:
                logger.error(
                    f"Error loading FAISS index from {FAISS_INDEX_PATH}: {e}",
                    exc_info=True,
                )
                logger.warning("Creating a new FAISS index due to loading error.")
                try:
                    vectorstore_instance = FAISS.from_texts(
                        texts=["Initial dummy document for FAISS."],
                        embedding=emb_model,
                    )
                    vectorstore_instance.save_local(FAISS_INDEX_PATH)
                    logger.info(
                        f"New FAISS index created with dummy doc and saved to {FAISS_INDEX_PATH}"
                    )
                except Exception as create_e:
                    logger.error(
                        f"Failed to create new FAISS index: {create_e}", exc_info=True
                    )
                    raise RuntimeError(f"Could not create new FAISS index: {create_e}")
        else:
            logger.info(
                f"FAISS index path {FAISS_INDEX_PATH} not found or invalid. Creating new index."
            )
            try:
                vectorstore_instance = FAISS.from_texts(
                    texts=["Initial dummy document for FAISS."], embedding=emb_model
                )
                vectorstore_instance.save_local(FAISS_INDEX_PATH)
                logger.info(f"New FAISS index created and saved to {FAISS_INDEX_PATH}")
            except Exception as create_e:
                logger.error(
                    f"Failed to create new FAISS index: {create_e}", exc_info=True
                )
                raise RuntimeError(f"Could not create new FAISS index: {create_e}")
    return vectorstore_instance


class IndexRequest(BaseModel):
    docs: List[str]


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 3


@app.post("/index")
def index_docs(request: IndexRequest):
    try:
        vecstore = get_vectorstore()
        if not request.docs:
            logger.warning("No documents provided for indexing.")
            return {
                "status": "no documents provided",
                "num_docs_in_store": vecstore.index.ntotal if vecstore.index else 0,
            }
        logger.info(f"Indexing {len(request.docs)} new documents.")
        vecstore.add_texts(texts=request.docs)
        vecstore.save_local(FAISS_INDEX_PATH)
        logger.info(
            f"Index updated and saved. Total documents in store: {vecstore.index.ntotal}"
        )
        return {"status": "indexed", "num_docs_in_store": vecstore.index.ntotal}
    except Exception as e:
        logger.error(f"Error during indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    try:
        vecstore = get_vectorstore()
        if not vecstore.index or vecstore.index.ntotal == 0:
            logger.warning(
                "Vector store is empty or index not initialized. Cannot retrieve."
            )
            return {
                "results": [],
                "message": "Vector store is empty. Index documents first.",
            }

        if vecstore.index.ntotal == 1:

            try:
                first_doc_id = list(vecstore.docstore._dict.keys())[0]
                first_doc_content = vecstore.docstore._dict[first_doc_id].page_content
                if "Initial dummy document for FAISS" in first_doc_content:
                    logger.warning(
                        "Vector store contains only the initial dummy document."
                    )

            except Exception:
                logger.warning(
                    "Could not inspect docstore for dummy document, proceeding with retrieval."
                )

        logger.info(
            f"Retrieving documents for query: '{request.query}' (top_k={request.top_k}). Total docs: {vecstore.index.ntotal}"
        )
        results_with_scores = vecstore.similarity_search_with_score(
            query=request.query, k=request.top_k
        )
        formatted_results = [
            {"doc": doc.page_content, "score": float(score)}
            for doc, score in results_with_scores
        ]
        logger.info(f"Retrieved {len(formatted_results)} results.")
        return {"results": formatted_results}
    except Exception as e:
        logger.error(f"Error during retrieval: {e}", exc_info=True)

        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

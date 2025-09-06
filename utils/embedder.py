import os
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_NAME = os.getenv("EMBEDDING_NAME", "sentence-transformers/all-MiniLM-L6-v2")

class STEmbeddingFunction:
    """Callable wrapper compatible with Chroma's embedding_function interface."""
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or EMBEDDING_NAME
        self._model = SentenceTransformer(self.model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.tolist()

def encode_one(text: str) -> List[float]:
    model = SentenceTransformer(EMBEDDING_NAME)
    v = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.tolist()

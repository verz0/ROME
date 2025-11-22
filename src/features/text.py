import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np

class TextFeatureExtractor:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the BERT-based text encoder.
        Using 'all-MiniLM-L6-v2' for speed/performance balance.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading Text Model {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def extract(self, text_segments: List[str]) -> np.ndarray:
        """
        Encodes a list of text strings into embeddings.
        Returns: numpy array of shape (N, 384)
        """
        embeddings = self.model.encode(text_segments, convert_to_numpy=True)
        return embeddings

    def get_embedding_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

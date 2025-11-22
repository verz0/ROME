import torch
import numpy as np
from src.features.text import TextFeatureExtractor

class RoleEncoder:
    def __init__(self, text_extractor: TextFeatureExtractor):
        self.text_extractor = text_extractor
        self.role_cache = {}

    def encode_role(self, role_description: str) -> np.ndarray:
        """
        Encodes a role description (e.g., "Project Manager interested in deadlines")
        into a vector using the same text encoder as the transcript.
        """
        if role_description in self.role_cache:
            return self.role_cache[role_description]
            
        # Encode as a single string
        embedding = self.text_extractor.extract([role_description])[0]
        self.role_cache[role_description] = embedding
        return embedding

    def get_embedding_dim(self) -> int:
        return self.text_extractor.get_embedding_dim()

import librosa
import numpy as np
from typing import Optional

class AudioFeatureExtractor:
    def __init__(self, sr: int = 16000, n_mfcc: int = 13):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def extract_segment_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Extracts MFCC features for a given audio segment.
        Returns the mean MFCC vector (shape: n_mfcc,).
        """
        if len(audio_segment) == 0:
            return np.zeros(self.n_mfcc)
            
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio_segment, sr=self.sr, n_mfcc=self.n_mfcc)
        
        # Take mean across time to get a single vector per segment
        mfcc_mean = np.mean(mfccs, axis=1)
        
        return mfcc_mean

    def get_embedding_dim(self) -> int:
        return self.n_mfcc

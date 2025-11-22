import pandas as pd
import numpy as np
from typing import List, Dict

class DataProcessor:
    @staticmethod
    def align_transcript_to_video(transcript_df: pd.DataFrame, video_fps: float) -> pd.DataFrame:
        """
        Adds frame indices to the transcript dataframe.
        """
        if video_fps <= 0:
            raise ValueError("FPS must be positive")
            
        df = transcript_df.copy()
        df['start_frame'] = (df['start_time'] * video_fps).astype(int)
        df['end_frame'] = (df['end_time'] * video_fps).astype(int)
        return df

    @staticmethod
    def normalize_audio(audio_array: np.ndarray) -> np.ndarray:
        """
        Normalizes audio amplitude to [-1, 1].
        """
        max_val = np.max(np.abs(audio_array))
        if max_val > 0:
            return audio_array / max_val
        return audio_array

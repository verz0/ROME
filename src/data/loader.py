import os
import cv2
import librosa
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

class MeetingLoader:
    def __init__(self, video_path: str, transcript_path: str):
        self.video_path = video_path
        self.transcript_path = transcript_path
        self.meeting_id = os.path.basename(video_path).split('.')[0]
        
    def load_video_metadata(self) -> Dict:
        """Returns basic video metadata (fps, duration, resolution)."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration
        }

    def load_audio(self, sr: int = 16000):
        """Loads audio from the video file using librosa."""
        # Note: librosa can load audio directly from video files if ffmpeg is installed
        try:
            y, sr = librosa.load(self.video_path, sr=sr)
            return y, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None

    def load_transcript(self) -> pd.DataFrame:
        """
        Parses the transcript file. 
        Assumes a simple CSV/JSON format for now, or VTT parsing logic can be added.
        Expected columns: ['start_time', 'end_time', 'speaker', 'text']
        """
        if self.transcript_path.endswith('.csv'):
            df = pd.read_csv(self.transcript_path)
        elif self.transcript_path.endswith('.json'):
            df = pd.read_json(self.transcript_path)
        elif self.transcript_path.endswith('.xml'):
            return self._parse_ami_xml()
        else:
            # Placeholder for VTT/SRT parsing
            raise NotImplementedError("Only CSV/JSON/XML transcripts supported for prototype.")
            
        required_cols = {'start_time', 'end_time', 'speaker', 'text'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Transcript must contain columns: {required_cols}")
            
        return df

    def _parse_ami_xml(self) -> pd.DataFrame:
        """
        Parses AMI Corpus XML transcript format.
        """
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(self.transcript_path)
        root = tree.getroot()
        
        # AMI XML structure: <segment><word>...</word></segment>
        # We want to group words into segments or just use the segments if they have text.
        # The dummy transcript I created has words inside segments.
        
        rows = []
        # Namespace handling might be needed if nite:id is used, but usually findall works with tags
        # Let's try to iterate segments
        
        # Find all 'segment' tags (ignoring namespace for simplicity if possible, or handling it)
        # The dummy transcript uses <segment> and <word>
        
        for segment in root.findall('.//segment'):
            # Get timing from segment or first/last word
            start = float(segment.get('starttime', 0))
            end = float(segment.get('endtime', 0))
            
            # Collect text from words
            words = segment.findall('word')
            if words:
                text = " ".join([w.text for w in words if w.text])
            else:
                text = segment.text if segment.text else ""
                
            if text:
                rows.append({
                    "start_time": start,
                    "end_time": end,
                    "speaker": "Speaker 1", # AMI XMLs are usually per-speaker, but for this sample we assume single file
                    "text": text.strip()
                })
                
        return pd.DataFrame(rows)

class SegmentGenerator:
    def __init__(self, window_size_sec: int = 30, step_size_sec: int = 15):
        self.window_size = window_size_sec
        self.step_size = step_size_sec

    def segment_meeting(self, duration: float, transcript_df: pd.DataFrame) -> List[Dict]:
        """
        Creates fixed-length segments and maps transcript lines to them.
        """
        segments = []
        current_time = 0.0
        
        while current_time < duration:
            end_time = min(current_time + self.window_size, duration)
            
            # Find transcript lines that overlap with this window
            # Overlap logic: (StartA <= EndB) and (EndA >= StartB)
            mask = (transcript_df['start_time'] <= end_time) & (transcript_df['end_time'] >= current_time)
            segment_transcript = transcript_df[mask]
            
            # Concatenate text
            text_content = " ".join(segment_transcript['text'].astype(str).tolist())
            
            segments.append({
                "start_time": current_time,
                "end_time": end_time,
                "text": text_content,
                "transcript_rows": segment_transcript.to_dict('records')
            })
            
            current_time += self.step_size
            
        return segments

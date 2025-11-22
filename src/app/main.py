import streamlit as st
import os
import pandas as pd
import tempfile
from src.data.loader import MeetingLoader, SegmentGenerator
from src.features.text import TextFeatureExtractor
from src.models.role_encoder import RoleEncoder
from src.app.utils import generate_highlight_video

# Page Config
st.set_page_config(page_title="RoME: Role-aware Meeting Summarizer", layout="wide")

st.title("RoME: Role-aware Multimodal Meeting Summarizer")
st.markdown("Upload a meeting video and transcript, select your role, and get a personalized highlight reel.")

# Sidebar: Inputs
st.sidebar.header("1. Upload Data")
video_file = st.sidebar.file_uploader("Upload Meeting Video", type=['mp4', 'mov', 'avi'])
transcript_file = st.sidebar.file_uploader("Upload Transcript (CSV/JSON/XML)", type=['csv', 'json', 'xml'])

st.sidebar.header("2. Select Role")
role_options = [
    "Project Manager (Deadlines, Budget, Blockers)",
    "Backend Developer (API, Database, Architecture)",
    "Frontend Designer (UX, UI, CSS, Flow)",
    "QA Engineer (Bugs, Testing, Release)"
]
selected_role_desc = st.sidebar.selectbox("Choose your perspective:", role_options)

# Main Area
if video_file and transcript_file:
    # Save uploaded files temporarily
    tfile_video = tempfile.NamedTemporaryFile(delete=False, suffix=f".{video_file.name.split('.')[-1]}") 
    tfile_video.write(video_file.read())
    video_path = tfile_video.name
    
    tfile_trans = tempfile.NamedTemporaryFile(delete=False, suffix=f".{transcript_file.name.split('.')[-1]}")
    tfile_trans.write(transcript_file.read())
    transcript_path = tfile_trans.name
    
    # Load Data using MeetingLoader
    loader = MeetingLoader(video_path, transcript_path)
    transcript_df = loader.load_transcript()
        
    st.info(f"Loaded Video: {video_file.name} | Transcript: {len(transcript_df)} lines")
    
    if st.button("Generate Highlights"):
        with st.spinner("Processing Meeting... (This may take a minute)"):
            # 1. Load & Segment
            # loader already initialized above
            # metadata = loader.load_video_metadata()
            
            segmenter = SegmentGenerator(window_size_sec=30, step_size_sec=30)
            # For prototype, we use a dummy duration or calculate it
            # duration = metadata['duration']
            duration = transcript_df['end_time'].max()
            segments = segmenter.segment_meeting(duration, transcript_df)
            
            st.write(f"Divided meeting into {len(segments)} segments.")
            
            # 2. Feature Extraction (Text Only for Prototype Speed)
            text_extractor = TextFeatureExtractor()
            role_encoder = RoleEncoder(text_extractor)
            
            # 3. Scoring (Dummy Logic for Prototype until Model is Trained)
            # We will use Cosine Similarity between Role Embedding and Segment Text Embedding
            role_emb = role_encoder.encode_role(selected_role_desc)
            
            scored_segments = []
            segment_texts = [s['text'] for s in segments]
            seg_embs = text_extractor.extract(segment_texts)
            
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity([role_emb], seg_embs)[0]
            
            for i, seg in enumerate(segments):
                seg['score'] = scores[i]
                scored_segments.append(seg)
                
            # 4. Filter Top Segments
            # Sort by score and take top 3 or top 20%
            scored_segments.sort(key=lambda x: x['score'], reverse=True)
            top_segments = scored_segments[:3] # Top 3 segments
            
            # Sort back by time for the video
            top_segments.sort(key=lambda x: x['start_time'])
            
            # 5. Generate Video
            output_video_path = "highlight_reel.mp4"
            success = generate_highlight_video(video_path, top_segments, output_video_path)
            
            if success:
                st.success("Highlight Reel Generated!")
                st.video(output_video_path)
                
                st.subheader("Summary of Highlights")
                for seg in top_segments:
                    st.markdown(f"**{seg['start_time']:.1f}s - {seg['end_time']:.1f}s** (Score: {seg['score']:.2f})")
                    st.write(seg['text'])
                    st.divider()
            else:
                st.error("Failed to generate video.")
                
else:
    st.info("Please upload both a video and a transcript to begin.")

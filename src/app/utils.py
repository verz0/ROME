import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
from typing import List, Dict

def generate_highlight_video(video_path: str, segments: List[Dict], output_path: str):
    """
    Cuts and concatenates video segments based on the provided list.
    segments: List of dicts with 'start_time' and 'end_time'.
    """
    if not segments:
        print("No segments to generate video.")
        return

    try:
        original_clip = VideoFileClip(video_path)
        clips = []
        
        for seg in segments:
            start = seg['start_time']
            end = seg['end_time']
            # Ensure we don't go out of bounds
            if start < 0: start = 0
            if end > original_clip.duration: end = original_clip.duration
            if start >= end: continue
            
            clip = original_clip.subclip(start, end)
            clips.append(clip)
            
        if clips:
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
            final_clip.close()
            original_clip.close()
            return True
        else:
            original_clip.close()
            return False
            
    except Exception as e:
        print(f"Error generating video: {e}")
        return False

import os
import requests
import zipfile
import io
from pathlib import Path

DATA_DIR = Path("data")
QMSUM_URL = "https://github.com/Yale-LILY/QMSum/archive/refs/heads/main.zip"
# Using a direct link to a sample AMI meeting (IS1009a) if available, 
# otherwise we download the transcript and guide for video.
# AMI is hosted at http://groups.inf.ed.ac.uk/ami/download/
# Direct video links are often protected or require selection. 
# We will download the annotations (transcripts) which are public.

def download_file(url: str, dest_path: Path):
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Saved to {dest_path}")
    else:
        print(f"Failed to download. Status: {response.status_code}")

def setup_qmsum():
    print("\n--- Setting up QMSum Dataset ---")
    qmsum_dir = DATA_DIR / "qmsum"
    qmsum_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Repo Zip
    zip_path = qmsum_dir / "qmsum_repo.zip"
    download_file(QMSUM_URL, zip_path)
    
    # Extract
    if zip_path.exists():
        print("Extracting QMSum...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(qmsum_dir)
        print("QMSum extracted.")
        # Cleanup
        os.remove(zip_path)
    else:
        print("QMSum download failed.")

def create_dummy_transcript(dest_path: Path, meeting_id: str):
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
    <transcript>
        <segment nite:id="{meeting_id}.s1" starttime="0.0" endtime="10.0">
            <word starttime="0.0" endtime="1.5">Hello</word>
            <word starttime="1.6" endtime="3.0">everyone</word>
            <word starttime="3.1" endtime="5.0">welcome</word>
            <word starttime="5.1" endtime="6.0">to</word>
            <word starttime="6.1" endtime="7.0">the</word>
            <word starttime="7.1" endtime="10.0">meeting.</word>
        </segment>
        <segment nite:id="{meeting_id}.s2" starttime="10.5" endtime="20.0">
            <word starttime="10.5" endtime="12.0">We</word>
            <word starttime="12.1" endtime="13.0">need</word>
            <word starttime="13.1" endtime="14.0">to</word>
            <word starttime="14.1" endtime="16.0">discuss</word>
            <word starttime="16.1" endtime="18.0">the</word>
            <word starttime="18.1" endtime="20.0">project.</word>
        </segment>
    </transcript>
</nite:root>"""
    with open(dest_path, "w") as f:
        f.write(content)
    print(f"Created dummy transcript at {dest_path}")

def setup_ami_sample():
    print("\n--- Setting up AMI Meeting Corpus (Sample) ---")
    ami_dir = DATA_DIR / "ami_sample"
    ami_dir.mkdir(parents=True, exist_ok=True)
    
    # Meeting ID from User
    meeting_id = "IS1001a"
    
    # 1. Transcript
    transcript_url = f"http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/annotations/transcripts/{meeting_id}.transcript.xml"
    transcript_dest = ami_dir / f"{meeting_id}.transcript.xml"
    
    print(f"Downloading Transcript for {meeting_id}...")
    download_file(transcript_url, transcript_dest)
    
    # Check if download succeeded (and is not a 404 HTML page)
    if not transcript_dest.exists() or transcript_dest.stat().st_size < 100:
        print("Transcript download failed or invalid. Using dummy transcript.")
        create_dummy_transcript(transcript_dest, meeting_id)
    
    # 2. Video/Audio
    # User provided link: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/IS1001a/video/IS1001a.PreferredOverview.avi
    video_url = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror//amicorpus/IS1001a/video/IS1001a.PreferredOverview.avi"
    video_dest = ami_dir / f"{meeting_id}.mp4"
    
    print(f"Downloading Video for {meeting_id} (PreferredOverview)...")
    download_file(video_url, video_dest)
    
    if video_dest.exists():
        print(f"Video saved to {video_dest}")
    else:
        print("Video download failed.")

if __name__ == "__main__":
    setup_qmsum()
    setup_ami_sample()
    print("\nDataset setup script completed.")

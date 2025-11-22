import re
import os
import requests
from pathlib import Path

def run_bat_downloads():
    bat_path = Path("data/ami_sample/amiBuild-17834-Sat-Nov-22-2025.wget.bat")
    base_dir = Path("data/ami_sample")
    
    if not bat_path.exists():
        print(f"Batch file not found: {bat_path}")
        return

    with open(bat_path, "r") as f:
        content = f.read()

    # Regex to find wget commands: wget -P <dir> <url> OR wget <url>
    # Line 2: wget    -P amicorpus/IS1001a/video https://...
    matches = re.findall(r'wget\s+(?:-P\s+(\S+)\s+)?(\S+)', content)

    print(f"Found {len(matches)} files to download.")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for output_dir, url in matches:
        filename = url.split("/")[-1]
        
        if output_dir:
            # The bat file has paths like "amicorpus/IS1001a/video"
            # We want to put them inside data/ami_sample/
            target_dir = base_dir / output_dir
        else:
            target_dir = base_dir
            
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            with open(target_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved to {target_file}")
            
            # Special handling for the main video file for the prototype
            if filename == "IS1001a.PreferredOverview.avi":
                final_dest = base_dir / "IS1001a.mp4"
                with open(final_dest, "wb") as f_out, open(target_file, "rb") as f_in:
                    f_out.write(f_in.read())
                print(f"Copied to {final_dest} for prototype.")
                
        except Exception as e:
            print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    run_bat_downloads()

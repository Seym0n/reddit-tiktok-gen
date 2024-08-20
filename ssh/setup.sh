#!/bin/bash
# Clone the repo
git clone https://github.com/jwtly10/reddit-tiktok-gen.git

# Create a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install the requirements
pip install -r requirements.txt

# Install deps
sudo apt install ffmpeg

# Run deps
docker run -p 8765:8765 lowerquality/gentle

# Install fonts
sudo cp assets/Poppins-SemiBold.ttf /usr/share/fonts/
sudo cp assets/Mont-HeavyDEMO.otf /usr/share/fonts/
sudo fc-cache -fv

# Copy background media file(s) to the remote machine
scp ./assets/minecraft_background_video_1.mp4 user@host:/root/reddit-tiktok-gen/assets/
scp ./assets/subwaysurfer_background_video_1.mp4 user@host:/root/reddit-tiktok-gen/assets/
scp ./assets/templerun_background_video_1.mp4 user@host:/root/reddit-tiktok-gen/assets/
scp ./assets/railrush_background_video_1.mp4 user@host:/root/reddit-tiktok-gen/assets/
scp ./assets/zombierun_background_video_1.mp4 user@host:/root/reddit-tiktok-gen/assets/


# Copy output files to local machine
mkdir ./ssh-output
scp -r user@host:/root/reddit-tiktok-gen/automation ./ssh-output

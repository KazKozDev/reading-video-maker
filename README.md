<p align="center">
  <img width="400" height="106" alt="loggo" src="https://github.com/user-attachments/assets/a8939a4a-3a1c-43f3-a78e-1c95cd7b3ed5" />
</p>

<h2 align="center">Reading Video Maker</h2>


Create a text-synchronized "video book" from an audio/video file and a plain-text transcript. The app aligns words to the audio, paginates the text to fit a configurable page, and renders a video with word highlighting.

### Perfect for:

📺 YouTube Creators - Transform your podcasts, interviews, or spoken content into engaging visual videos with automatic word highlighting. No more static audio-only content!

📚 Educational Content Makers - Turn lectures, tutorials, or educational audio into professional-looking video lessons. Students can follow along visually while listening.

🌍 Language Learning - Create immersive language learning videos where learners can see and hear each word simultaneously. Perfect for pronunciation practice and reading comprehension.

🎤 Content Creators - Generate beautiful karaoke-style videos from any speech or narration. Ideal for audiobooks, storytelling, or motivational content.

📖 Accessibility - Make your audio content accessible to hearing-impaired audiences and improve comprehension for all viewers with synchronized visual text.

🎬 Video Production - Automatically generate professional subtitles and transcriptions with precise timing. Save hours of manual work!

The app handles the complex audio-to-text alignment using AI (Whisper), creates beautiful book-like pages, and outputs a polished video where each word lights up exactly when spoken - like a high-end teleprompter or karaoke system.

<img width="902" height="730" alt="Screenshot" src="https://github.com/user-attachments/assets/469dc896-5097-4969-85db-dc6489fd08ec" />


### Features
- Extract audio from video (or accept audio directly)
- Align text to audio using stable-whisper (stable-ts)
- Paginate text with automatic layout scaling
- Render frames with word highlighting in sync with audio
- Minimal Tkinter UI with progress logs and Stop control

### Requirements
- Python 3.9+
- FFmpeg available in PATH (on macOS typically via Homebrew)
- Packages:
  - stable-ts (stable-whisper)
  - opencv-python
  - pillow
  - moviepy
  - numpy
  - imageio-ffmpeg (helps MoviePy find ffmpeg)

### Installation
```bash
# Recommended
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install \
  git+https://github.com/jianfch/stable-ts.git \
  opencv-python pillow moviepy numpy imageio-ffmpeg
```

On macOS, FFmpeg via Homebrew:
```bash
brew install ffmpeg
```

### Run the App
```bash
python3 main.py
```
If you hit Tkinter + MoviePy issues on macOS, try:
```bash
pythonw main.py
```

### Usage (GUI)
1. Choose input Audio/Video and Text (.txt)
2. Optionally choose a TTF/OTF font and set output size/FPS
3. Click "Build" to start
4. Use "Stop" to cancel rendering safely

The output file defaults to `final_video_book.mp4` in the project directory.

### Tips
- Text should match the narration reasonably well for best alignment quality
- For better typography, specify a font file (e.g., `.ttf`)
- On macOS the app forces SDL to headless mode to avoid conflicts with Tkinter

## Troubleshooting
- "ffmpeg not found": install via Homebrew or ensure it’s in PATH
- MoviePy import issues on macOS: install `imageio-ffmpeg` or run with `pythonw`
- OpenCV writer fails: the app falls back to MJPG/AVI automatically


---

If you like this project, please give it a star ⭐

For questions, feedback, or support, reach out to:

[Artem KK](https://www.linkedin.com/in/kazkozdev/) | MIT [LICENSE](LICENSE) 

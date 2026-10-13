# EMO HTML Renderer

A lightweight HTML/CSS/JavaScript face renderer for the EMO Raspberry Pi project. Python sends face state through a small Flask JSON endpoint; CSS/JavaScript handles the visual animation.

## Features

- Animated eyes and idle glances/blinks
- Listening, thinking, talking, happy, sad, angry, curious, sleepy, love, focus and other expression states
- Animated talking mouth
- Forehead thinking spinner
- Overheat flame effect and temperature indicator
- Camera status indicator

## Files

- `face.html`: frontend renderer
- `html_face.py`: Flask bridge/state controller
- `emo.py`: main AI, audio, thermal and camera backend (add from your Pi)

## Setup on Raspberry Pi OS

Use a 64-bit OS and a virtual environment:

```bash
sudo apt update
sudo apt install -y python3-full python3-venv
mkdir -p ~/emo
cd ~/emo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install Flask
```

Place `face.html`, `html_face.py`, and your backend `emo.py` in the same `~/emo` directory.

## Test the renderer alone

```bash
python - <<'PY'
import time
from html_face import HtmlFace

face = HtmlFace()
face.start_server()
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    face.stop_server()
PY
```

Open `http://127.0.0.1:8765/` from Chromium running on the Pi's graphical desktop.

## Backend integration

The HTML controller preserves these methods from the previous face class:

```python
face.set_state("listening")
face.set_state("thinking")
face.set_state("talking")
face.set_talking(True)
face.set_talking(False)
face.set_state("happy", 2.0)
```

Your backend must create `HtmlFace(status)`, call `face.start_server()`, and keep the Flask server alive.

## Important hardware note

A Raspberry Pi with around 1 GB RAM may not be able to run Chromium, local Whisper, Ollama and Chatterbox Base concurrently. Test the renderer and each model independently. The HTML page does not make model inference lighter.

## Safety and repository hygiene

Do not commit virtual environments, AI model weights, recordings, API keys, or device-specific secrets. See `.gitignore`.

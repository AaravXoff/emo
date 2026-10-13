# EMO — Local AI Robot Assistant

EMO is a modular Python assistant for Raspberry Pi OS with an HTML/CSS face, local speech capture, Whisper speech recognition, fast local instructions, a SQLite context/response cache, LM Studio fallback, and speech output.

Repository: https://github.com/AaravXoff/emo

## Current capabilities

- Local microphone input with energy-based voice activity detection and configurable wake phrase (`Hey EMO` by default).
- Whisper language detection and transcription; offline Argos Translate packs or Whisper's built-in translation task can translate non-English speech to English.
- Local, low-latency intent handling for greetings, current time, weather, animal sound requests, and an original EMO tune.
- LM Studio OpenAI-compatible endpoint for fallback questions (configured for Qwen 2.5 0.5B; use the exact model ID reported by your server).
- Replies localized with installed offline translation packs or generated in the detected language by the local LLM.
- Lazy Chatterbox TTS support with `espeak-ng` fallback.
- SQLite stores text transcripts, detected language, translated input, route, response, timing and cache-hit metadata. Raw audio is deleted after recognition.
- HTML face with expressions, speaking mouth, thinking spinner, thermal flames, temperature/status API, and a boot progress screen.
- Thermal monitoring and optional GPIO touch sensor / single servo.

The NPU is not used by this code yet because the exact NPU board and runtime were not specified. The camera indicator remains available in the renderer, but camera gestures are not currently implemented in this version.

## Important Pi 3B+ limitations

A Raspberry Pi 3B+ has very limited RAM and CPU for modern neural models. This is why the default Whisper model is `tiny` and model loading is lazy. Test one model at a time before increasing model size. Chatterbox Base and Chromium plus Whisper plus a local LLM may not fit in the same memory budget. In auto mode, EMO avoids loading Chatterbox on a device with less than about 1.8 GB of RAM and uses `espeak-ng` instead.

**Whisper Turbo does not translate speech into English.** The upstream Whisper documentation says Turbo returns the source language even when `task="translate"` is passed. To use Turbo, install Argos language packs for the languages you need. Otherwise, use `tiny` or `base` for Whisper's translation task. A realistic starting setting for a Pi 3B+ is `WHISPER_MODEL=tiny`.

Python/PyTorch/Whisper wheel availability changes by operating system and Python version on ARM devices. The installer installs the small core package only; install Whisper separately and check `python -c "import whisper; print('Whisper import OK')"` before relying on live recognition.

## 1. Install the core on the Pi

Clone this repository into `~/emo` or update an existing checkout:

```bash
cd ~
git clone https://github.com/AaravXoff/emo.git
cd ~/emo
bash install_pi.sh
```

If you already cloned the repository:

```bash
cd ~/emo
git pull
bash install_pi.sh
```

Edit configuration:

```bash
nano ~/emo/.env
```

The installer copies `.env.example` to `.env`. Never commit `.env`; it is ignored by Git.

## 2. Install optional AI components

### Whisper speech recognition

```bash
cd ~/emo
source .venv/bin/activate
pip install -r requirements-stt.txt
python -c "import whisper; print('Whisper import OK')"
```

First model use may download weights. Keep `WHISPER_MODEL=tiny` initially. You can set `WHISPER_MODEL=base` if it runs acceptably. Avoid `turbo` on Pi 3B+ unless you have measured memory use and configured offline translation packs.

### Offline language translation (optional)

```bash
source .venv/bin/activate
pip install -r requirements-translation.txt
python translation_setup.py --pairs hi-en,en-hi,pa-en,en-pa
```

Only install language packs needed for your languages; they take storage. Use `--pairs` to customize. Argos translations run locally after packages are installed.

### LM Studio / Qwen

The program sends fallback questions to the configured LM Studio server endpoint (`LM_STUDIO_BASE_URL`). Start the local server in LM Studio, load your Qwen 2.5 0.5B Instruct model, and find its exact model ID at:

```bash
curl http://127.0.0.1:1234/v1/models
```

If LM Studio is running on a different computer, set `LM_STUDIO_BASE_URL` in the Pi's `.env` to that computer's LAN address, e.g. `http://192.168.1.40:1234/v1`, and configure LM Studio to accept LAN connections. The example IP is only illustrative. Do not expose the server to the public internet.

### Chatterbox TTS (optional)

Chatterbox is memory- and dependency-heavy and may not install or run well on Pi 3B+.

```bash
source .venv/bin/activate
pip install -r requirements-chatterbox.txt
```

Use `TTS_ENGINE=auto` to try Chatterbox on larger machines and fall back to `espeak-ng` on low-RAM boards. To force a lightweight voice, set `TTS_ENGINE=espeak`. To explicitly attempt neural voice output on your Pi, set `TTS_ENGINE=chatterbox` and expect potential installation/memory problems.

## 3. Weather and time

- Time uses the TimeAPI.io endpoint for the configured IANA zone in `TIMEZONE` (default `Asia/Kolkata`) and falls back to the Pi's system clock when the API is unavailable. Online use of the time API requires internet; the fallback works offline if the Pi clock is correct.
- Weather uses the Open-Meteo public geocoding/current-weather endpoints and needs an internet connection. No API key is required. Set `WEATHER_CITY` in `.env`, or ask for a named city (e.g., “what's the weather in Delhi?”). Weather responses are cached briefly.

## 4. Run

```bash
cd ~/emo
source .venv/bin/activate
python emo.py
```

Open `http://127.0.0.1:8765/` in a browser on the Pi to see the face. The JSON state is at `/api/state`, boot data at `/api/boot`, and health check at `/health`.

Raspberry Pi OS Lite does not include a desktop/browser by default. The HTML page needs a graphical browser session to appear on the local display; otherwise the Python assistant can run headless and the screen page can be opened from a browser on the same device if a graphical session is available.

## 5. Start at login with systemd

Run these after the files are in `~/emo` and `.venv` exists:

```bash
mkdir -p ~/.config/systemd/user
cp ~/emo/systemd/emo.service ~/.config/systemd/user/emo.service
systemctl --user daemon-reload
systemctl --user enable --now emo.service
systemctl --user status emo.service --no-pager
journalctl --user -u emo.service -f
```

A graphical auto-start for Chromium must be configured separately if you want the face shown on a local desktop/kiosk.

## 6. Tests

Install Flask in a development environment, then run the local logic tests:

```bash
python -m pip install Flask
python -m unittest discover -s tests -v
```

Tests exercise direct-intent routing, wake-phrase stripping, cache keys and context logging. Hardware, microphone, Whisper model inference, Open-Meteo, LM Studio and Chatterbox require device-specific integration testing.

## 7. Hardware and safety notes

- Set `TOUCH_GPIO` and `SERVO_GPIO` only after confirming physical pin numbering and signal wiring; `0` disables the optional component.
- Do not power a servo directly from a GPIO pin. Use a suitable external servo supply and a common ground.
- Thermal monitoring requests a pause for heavy inference at the configured critical temperature and resumes after cooling.
- No raw microphone audio is kept. Transcribed text and answers are stored locally in `data/emo_cache.sqlite3`. Set `CACHE_ENABLED=false` to disable the text cache/log, or remove the database when you want to erase its history.
- Weather requires connectivity. LM Studio must be reachable for fallback Q&A. Direct greeting/time/animal/tune actions can work without LM Studio.

## Project layout

- `emo.py`: startup, wake phrase loop, thermal lockout.
- `assistant.py`: direct intents, response cache, LM Studio fallback and orchestration.
- `speech.py`: Whisper transcription/language detection and translation stage.
- `audio_io.py`: microphone recording and synthesized sound effects.
- `translation.py`, `translation_setup.py`: optional offline translation.
- `response_cache.py`: local SQLite cache and text-only turn context.
- `llm_client.py`: LM Studio OpenAI-compatible API client.
- `tts_engine.py`: Chatterbox integration with lightweight fallback.
- `hardware.py`, `state.py`: optional GPIO/thermal status.
- `html_face.py`, `face.html`: Flask bridge and UI.
- `.env.example`: editable configuration template.

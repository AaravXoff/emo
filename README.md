# EMO — Local AI Robot Assistant

EMO is a modular Python assistant for Raspberry Pi OS with an HTML/CSS face, local speech capture, Whisper speech recognition, fast local instructions, a SQLite context/response cache, LM Studio fallback, and speech output.

Repository: https://github.com/AaravXoff/emo

## Current capabilities

- Local microphone input with input-device auto-selection, energy-based voice activity detection and configurable wake phrase (`Hey EMO` by default).
- Whisper.cpp or optional Python Whisper for language detection and transcription; Whisper.cpp can translate non-English speech to English and offline Argos Translate packs are also supported.
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

**Recommended STT for a Pi 3B+: Whisper.cpp.** It builds a native CPU executable, avoids installing the huge Python/PyTorch stack, supports Raspberry Pi, and runs multilingual models. The setup script below downloads the multilingual `tiny` model by default.

The original Python Whisper backend remains optional for machines with enough RAM and storage, but on your Pi its PyTorch package download may be very large. The native Whisper.cpp backend is what we recommend for the current hardware.

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

### Whisper speech recognition — recommended: Whisper.cpp

The native build is the recommended option for Raspberry Pi 3B+. It avoids installing a large Python/PyTorch wheel.

```bash
cd ~/emo
bash setup_whisper_cpp.sh
```

The build uses one compile job to keep peak RAM down. It downloads the multilingual tiny model into \`~/whisper.cpp/models/ggml-tiny.bin\`. Model download and initial compilation need an internet connection.

Open your existing local config file:

```bash
nano ~/emo/.env
```

Add or update these settings (if they're not present already):

```ini
WHISPER_BACKEND=cpp
WHISPER_CPP_BIN=~/whisper.cpp/build/bin/whisper-cli
WHISPER_CPP_MODEL=~/whisper.cpp/models/ggml-tiny.bin
WHISPER_CPP_THREADS=2
```

Save in Nano with Ctrl+O, Enter, then Ctrl+X. The program runs the small multilingual model on the CPU, detects the spoken language, and can run a second pass to translate speech into English.

If you deliberately want the Python Whisper backend on a larger computer, set \`WHISPER_BACKEND=python\` and install \`openai-whisper\` separately. We do not recommend repeating that large PyTorch install on a Pi 3B+ right now.

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

### Microphone troubleshooting

A Raspberry Pi 3B+ does not have a built-in microphone. Connect a USB microphone or a supported audio-input adapter. List ALSA capture hardware with `arecord -l` and Python/PortAudio devices with:

```bash
source ~/emo/.venv/bin/activate
python -c "import sounddevice as sd; print(sd.query_devices()); print('default device:', sd.default.device)"
```

EMO prefers a valid OS-default input and otherwise chooses the first detected input. To force a specific device, edit `AUDIO_INPUT_DEVICE` in `.env` to the device index (number) or a unique substring of the device name shown by `sd.query_devices()`. Use `auto` to return to automatic selection. EMO tries `AUDIO_SAMPLE_RATE` first; if the microphone does not support that rate, it tries the device's reported native rate and resamples to the model's required rate. When no capture device exists, EMO reports a throttled warning rather than printing a traceback every two seconds. The listening loop also observes shutdown signals so it does not wait for the full silence timeout when you stop the program.

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

- `TOUCH_GPIO` and `SERVO_GPIO` use **BCM GPIO numbers** (not physical header-pin positions); `0` disables the optional component. Valid config values are `0`–`27`; verify your board's pin map and wiring before enabling hardware.
- Do not power a servo directly from a GPIO pin. Use a suitable external servo supply and a common ground.
- Thermal monitoring requests a pause for heavy inference at the configured critical temperature and resumes after cooling.
- No raw microphone audio is kept. Transcripts and answers are stored locally in `data/emo_cache.sqlite3`. Set `CACHE_ENABLED=false` to disable that text cache/log. Saved notes and to-dos live separately in `data/emo_features.sqlite3` and remain when the transcript cache is disabled; remove the corresponding database file to erase each store. Relative data/cache paths are resolved from the project directory, not the shell's current directory.
- Weather requires connectivity. LM Studio must be reachable for fallback Q&A. Direct greeting/time/animal/tune actions can work without LM Studio.

## Local feature hub (100+ commands)

EMO includes a standard-library feature hub that runs before the local LLM fallback. The catalog covers calendar/date helpers, safe arithmetic, unit conversion, text transformations, base64/URL/binary/hex/JSON helpers, random utilities, saved notes, to-do items, countdown timers, Raspberry Pi/system diagnostics, and explicit face expressions. The feature inventory lives in `feature_hub.py`; the current catalog contains over 100 named capabilities.

Try saying or typing:

- `help` or `what can you do`
- `calculate 12 * (3 + 4)`, `square root of 81`, `average of 3, 7, 9`
- `convert 5 km to miles`, `100 c to f`
- `count words in: hello there`, `uppercase: hello emo`, `is this a palindrome: never odd or even`
- `base64 encode: hello`, `json pretty: {"ok":true,"count":3}`
- `add note: check the wiring`, `list notes`, `add todo: test the microphone`, `complete todo 1`
- `set timer for 10 seconds to take a break`, `cancel timer`, `start pomodoro`
- `show device status`, `memory status`, `cpu temperature`, `dance`

Notes and tasks persist locally in `data/emo_features.sqlite3`. Timers are in-memory and are cancelled when EMO shuts down. System diagnostics use local OS interfaces and do not perform shell execution; the network-status response confirms only whether a local interface is present, not internet reachability. Arithmetic is evaluated with a restricted syntax tree, not Python `eval`. The feature hub adds no third-party runtime dependencies.

The catalog is tested on GitHub Actions, but timers, GPIO, audio, real temperature sensors, and speech output still need testing on the actual Raspberry Pi. This is a large set of local utility commands, not a claim that every command uses an AI model.

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

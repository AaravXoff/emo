# EMO OS image builder

This directory configures a custom 64-bit Raspberry Pi OS Lite image for the Raspberry Pi 3B+.
A PC-style ISO is not the correct format for Raspberry Pi; the workflow creates a compressed,
flashable SD-card disk image (img.xz) using the official Raspberry Pi pi-gen build system.

## Building and downloading

The GitHub Actions workflow at .github/workflows/build-emo-image.yml builds this image when files
in image/ change, or when manually started from GitHub → Actions → Build EMO OS image → Run workflow.

When the build passes, it:
- runs EMO's Python compilation and unit tests;
- cross-compiles the pinned Whisper.cpp release for ARM64 and verifies the CLI flags used by EMO;
- creates the OS image with the EMO service enabled and the multilingual Whisper tiny model installed;
- validates XZ integrity and publishes a SHA-256 checksum;
- uploads an Actions artifact and a prerelease containing the image and checksum.

The image is headless to keep RAM use lower on the Pi 3B+. It does not bake in Wi-Fi credentials,
a developer .env, or a shared default password. The first boot must be configured with your own
username/password and network settings using Raspberry Pi Imager's OS customisation options.

## Flash and boot

1. Download the emo-os image and its .sha256 file from the latest EMO OS prerelease.
2. Verify the hash. On Linux, run sha256sum -c IMAGE-FILE.img.xz.sha256 from the directory that contains the image.
3. Open Raspberry Pi Imager, choose Use custom, select the downloaded image, choose the microSD card,
   configure your own user, password, Wi-Fi and SSH settings, then write it.
4. Insert the card into the Pi and power it on. Allow the first boot a few minutes.
5. Check service status with:
   
   systemctl status emo.service --no-pager
   journalctl -u emo.service -n 100 --no-pager

6. The image has no desktop/browser. EMO's face endpoint is local-only at http://127.0.0.1:8765/.
   To view it from another computer over SSH, use:
   
   ssh -L 8765:127.0.0.1:8765 YOUR_USER@emo-robot.local

   then open http://127.0.0.1:8765/ on that computer.

## Configuration and hardware

- Edit the local settings using sudo nano /opt/emo/.env. They are not part of the downloadable source image.
- Connect a USB microphone or supported audio input adapter; the Pi 3B+ has no built-in microphone.
- The image includes Whisper.cpp and the tiny multilingual model, plus lightweight espeak-ng speech output.
- For open-ended answers, configure LM_STUDIO_BASE_URL in /opt/emo/.env to a reachable LM Studio server. A local LLM is not preinstalled because the Pi 3B+ has limited RAM.
- No GPIO sensor or servo is enabled by default. Verify your wiring before changing those options.

## Rebuilding locally

The supported automated build is the GitHub Actions workflow. It relies on network access to Raspberry Pi package repositories and PyPI and uses the pinned
pi-gen and Whisper.cpp source revisions. The result is build-tested, but actual microphone, speaker, and SD-card boot behaviour still needs testing on the target Pi.

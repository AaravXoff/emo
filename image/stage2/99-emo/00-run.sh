#!/bin/bash -e
: "${ROOTFS_DIR:?pi-gen did not provide ROOTFS_DIR}"

install -d "${ROOTFS_DIR}/opt/emo"
rsync -a --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.env' \
  --exclude='data/' \
  --exclude='*.sqlite3*' \
  --exclude='*.wav' \
  --exclude='*.swp' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='.github/' \
  --exclude='image/' \
  --exclude='tests/' \
  files/emo/ "${ROOTFS_DIR}/opt/emo/"

install -D -m 0755 files/whisper/whisper-cli \
  "${ROOTFS_DIR}/opt/whisper.cpp/build/bin/whisper-cli"
install -D -m 0644 files/whisper/ggml-tiny.bin \
  "${ROOTFS_DIR}/opt/whisper.cpp/models/ggml-tiny.bin"

# Create a fresh non-secret local config; never copy a developer's .env into an image.
install -m 0640 "${ROOTFS_DIR}/opt/emo/.env.example" "${ROOTFS_DIR}/opt/emo/.env"
sed -i \
  -e 's|^WHISPER_BACKEND=.*|WHISPER_BACKEND=cpp|' \
  -e 's|^WHISPER_CPP_BIN=.*|WHISPER_CPP_BIN=/opt/whisper.cpp/build/bin/whisper-cli|' \
  -e 's|^WHISPER_CPP_MODEL=.*|WHISPER_CPP_MODEL=/opt/whisper.cpp/models/ggml-tiny.bin|' \
  -e 's|^EMO_DATA_DIR=.*|EMO_DATA_DIR=/var/lib/emo/data|' \
  "${ROOTFS_DIR}/opt/emo/.env"

install -D -m 0644 files/emo-system.service \
  "${ROOTFS_DIR}/etc/systemd/system/emo.service"
install -d "${ROOTFS_DIR}/etc/systemd/system/multi-user.target.wants"
ln -sfn ../emo.service "${ROOTFS_DIR}/etc/systemd/system/multi-user.target.wants/emo.service"

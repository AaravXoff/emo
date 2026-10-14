#!/bin/bash -e
if ! id -u emo >/dev/null 2>&1; then
  useradd --system --create-home --home-dir /var/lib/emo --shell /usr/sbin/nologin emo
fi
if getent group audio >/dev/null 2>&1; then
  usermod -a -G audio emo
fi

install -d -o emo -g emo -m 0750 /var/lib/emo /var/lib/emo/data

python3 -m venv /opt/emo/.venv
PIP_NO_CACHE_DIR=1 /opt/emo/.venv/bin/python -m pip install \
  --disable-pip-version-check --no-cache-dir --upgrade pip setuptools wheel
PIP_NO_CACHE_DIR=1 /opt/emo/.venv/bin/python -m pip install \
  --disable-pip-version-check --no-cache-dir -r /opt/emo/requirements.txt

test -x /opt/whisper.cpp/build/bin/whisper-cli
test -s /opt/whisper.cpp/models/ggml-tiny.bin
grep -q '^WHISPER_BACKEND=cpp$' /opt/emo/.env
grep -q '^EMO_DATA_DIR=/var/lib/emo/data$' /opt/emo/.env

chown -R root:emo /opt/emo
chmod 0755 /opt/emo
chmod 0640 /opt/emo/.env
chown root:emo /opt/emo/.env
chown -R root:root /opt/whisper.cpp
chmod 0755 /opt/whisper.cpp/build/bin/whisper-cli
chown -R emo:emo /var/lib/emo

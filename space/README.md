---
title: Nes2Net-X Fake Voice Detection
emoji: 🎙️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 1h
pinned: false
license: apache-2.0
short_description: Nes2Net-X (XLS-R) speech deepfake / spoof detector
models:
  - SpeechAntiSpoofingBenchmarks/Nes2Net
  - facebook/wav2vec2-xls-r-300m
---

# Nes2Net-X demo

Demo for *Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing*
(Liu et al., IEEE T-IFS 2025, arXiv:2504.05657). It uses the authors' released wav2vec 2.0 XLS-R + Nes2Net-X
checkpoint (`nes2net_x_DF1.65.pth`, mirrored at `SpeechAntiSpoofingBenchmarks/Nes2Net`). The fairseq checkpoint
is loaded into HF `Wav2Vec2Model` through a key converter, so fairseq is not needed. Runs on ZeroGPU.

Reproduction code, Kaggle notebook and ablation study:
https://github.com/trinhphamtrunghieu/nes2net-fake-voice-detection
(this Space is deployed automatically from `space/` by GitHub Actions).

Research demo only. The model was trained on ASVspoof 2019 LA, so accuracy drops on modern TTS and on noisy or
non-English audio.

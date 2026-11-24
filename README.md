# Spoken-Command Recognition Pipeline

> **About this repository.** This is a write-up of an industry project I built as a Senior Data Scientist (Machine Learning and Robotics) at Infiswift Technologies. The source code is proprietary and not included. Client names and the exact command vocabulary are omitted.

An always-on model that recognises short spoken commands for an industrial robot and runs on a **Hailo edge-AI accelerator**. It covers a **40+ word command vocabulary**: motion words, axes, joints, speed, tool and similar. No full speech recogniser is needed in the loop, and detection latency is about **300 ms** on live audio.

The hard part is safety. In a factory, a false trigger moves a real robot, so the model has to reject near-misses and background speech. That made this mostly a data problem.

**Role:** Lead and sole developer

**Stack:** Python · PyTorch · openWakeWord · ONNX · quantisation · Hailo Edge AI · TTS voice cloning · NumPy

---

## Pipeline

![Training and runtime pipeline](docs/architecture.svg)

The model has three stages:

| Stage | What it does | Trained? |
|---|---|---|
| **1. Mel spectrogram** | Turns 16 kHz audio into mel-spectrogram frames | Pretrained; I reimplemented it natively from extracted weights so the device doesn't need ONNX tooling |
| **2. Speech embedding model** | Turns each window of mel frames into a compact speech embedding (16 time steps × 96 features) | Pretrained and frozen |
| **3. Command classifier** | Classifies the embedding sequence into one of 40+ command words or "background" | **Custom, trained by me** |

Freezing the first two stages means only a small classifier has to be trained. That makes it practical to train on synthetic data and to run on an edge device.

## How it works

### Command classifier
- **Architecture.** A compact network of dense layers (Linear + LayerNorm blocks), with a configurable number of blocks. It ends in one softmax over the command words plus a background class and is trained with cross-entropy.
- **NPU version.** For the Hailo accelerator, the same network was rewritten with convolutional layers so it compiles for the NPU. I fixed the export tensor layout to `(B, C, H, W)`.

### Dataset
- **Built from scratch.** No recordings of the command vocabulary existed. Positives were synthesised with Qwen TTS voice cloning, using reference speakers from multi-speaker English and Japanese corpora for speaker diversity.
- **Quality filter.** Overlong or poor-quality synthetic clips are removed automatically before training.
- **Graded negatives.** Easy, medium and hard negatives plus near-miss phrases, so the model doesn't fire on similar-sounding speech. Sources include:
  - LibriSpeech read speech
  - FMA music
  - FSD50K environmental and vocal sounds
  - ~2,000 h of precomputed general-audio features
  - microphone recordings from the deployment environment
- **Augmentation.** Room impulse responses and background audio.
- **Reproducible runs.** A stage-based layout (`samples → dataset → features → model → eval`), so every training run traces back to the data it used.

### Edge deployment
- **Export.** The classifier is exported to ONNX, quantised, and deployed on a Hailo accelerator. Calibration data is taken from the same augmented training set.
- **Streaming runtime.** Live audio is processed continuously in 10 ms frames, in synchronous or producer/consumer mode.

### Evaluation
- Scoring uses the same detection policy as the deployed runtime (thresholds plus energy gating), including false positives per hour.
- Continuous real-world audio is recorded to find false triggers that clip-level metrics miss.
- Latency benchmarks cover the front end and the end-to-end runtime.

## Results
| Metric | Value |
|---|---|
| Detection latency (continuous live audio) | **~300 ms** |
| Vocabulary | 40+ command words plus background |
| Deployment | Quantised classifier on a Hailo edge-AI accelerator |

## What I learned
Data was the bottleneck, not the model. With the two front-end stages frozen, how I designed the synthetic positives, hard negatives and near-misses mattered more than any architecture change. Testing on continuous real-world audio exposed failures that clip-level metrics hid. Building datasets carefully is a skill I expect to use directly in learning robot policies from demonstrations.

---

Full portfolio: [rahulg-infiswift.github.io](https://rahulg-infiswift.github.io/)

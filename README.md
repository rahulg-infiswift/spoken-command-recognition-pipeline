# On-Device Wake-Word and Command Detection

> **About this repository.** This is a write-up of an industry project I built as an ML / Robotics Engineer at Infiswift, for an industrial robotics client. The source code is proprietary and not included. The client's name and the actual wake phrase are omitted.

An always-on keyword spotter for a voice-controlled robot. It listens for a custom wake phrase plus short control words (*start*, *stop*) and runs on an edge device in 10 ms frames, on CPU or a Hailo NPU. This means operators don't have to press a button, and a full speech recogniser doesn't have to run all the time.

The two hard parts: **no real recordings of the wake phrase existed**, and in a noisy factory **a false trigger is costly**. Both made this mostly a data problem.

**Role:** Lead and sole developer · **Duration:** Apr 2026 – May 2026

**Stack:** Python · PyTorch · openWakeWord · ONNX Runtime · Hailo NPU · Qwen TTS · NumPy

---

## Architecture

![Training and runtime architecture](docs/architecture.svg)

## How it works

### Synthetic data generation
- **Positives.** Qwen TTS voice cloning, with reference speakers from the VCTK (English) and JVS (Japanese) corpora, plus configurable "custom voice" variations for more speaker diversity. Generation can append to or overwrite each label's samples.
- **Running on limited GPUs.** Adaptive handling of CUDA out-of-memory errors (batch backoff and retry, expandable allocator segments) keeps long generation jobs alive on modest hardware.
- **Hard negatives from three sources:**
  1. confusable spoken phrases (near-miss words and sentences)
  2. environmental sound categories from FSD50K
  3. microphone recordings from the deployment environment

### Training pipeline
- **Reproducible, stage-based layout:** `samples → dataset → features → model → eval`. Every training run is self-contained and traceable to the sample pool it was built from, so experiments can be compared directly.
- **Augmentation and features.** Room impulse responses and AudioSet/FMA background audio, then openWakeWord's mel-spectrogram and speech-embedding frontend. About 2,000 hours of precomputed ACAV100M features serve as general negatives.
- **Models.** Single-class and multi-class classifier heads. Environmental negatives are sampled separately during multi-class training to keep the classes balanced.

### Edge runtime
- **Streaming.** 10 ms audio frames, in synchronous and producer/consumer execution modes.
- **Lean dependencies.** The mel frontend is reimplemented natively from pre-extracted weights, so the production runtime no longer needs the `onnx` package.
- **Three backends from one training pipeline:** split ONNX Runtime graphs, a merged ONNX Runtime graph, and a **Hailo NPU** path. I fixed the export tensor layout to `(B, C, H, W)` so the model compiles for the accelerator.
- **Quantisation.** Packed-mel calibration data is exported from the same augmented training data, so calibration matches what the model saw in training.

### Evaluation
- Clip-level ground-truth scoring, using a detection policy (score thresholds plus RMS energy gating) that matches the deployed runtime exactly.
- **Recording of continuous audio streams** to find false positives under real conditions, which clip-level metrics miss.
- Latency benchmarks for the mel frontend and the end-to-end streaming runtime.

## Results
- **10 ms** streaming frame cadence on the edge runtime.
- **Three deployment backends** (split ONNX, merged ONNX, Hailo NPU) from a single training pipeline.
<!-- TODO: add per-class recall / true-accept rate -->
<!-- TODO: add false accepts per hour on continuous audio -->
<!-- TODO: add per-frame latency on CPU and NPU -->

## What I learned
Data was the bottleneck, not the model. The design of the synthetic positives and especially the hard negatives mattered far more than changes to the architecture. Testing on continuous real-world audio exposed failures that clip-level metrics hid.

---

Full portfolio: [rahulg-infiswift.github.io](https://rahulg-infiswift.github.io/)

<!-- TODO: link to the open reimplementation once published -->

# ⚡ VeloceNPU

<p align="center">
  <img src="https://img.shields.io/badge/AMD-Ryzen™%20AI%20NPU-red" />
  <img src="https://img.shields.io/badge/Architecture-XDNA%201%20%7C%20XDNA%202-blue" />
  <img src="https://img.shields.io/badge/C%2B%2B-20-blue" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

**VeloceNPU** is a lightweight, high-performance inference engine built for **AMD Ryzen™ AI NPUs**. It enables fast, private, and energy-efficient local execution of large language and multimodal models across **XDNA 1** and **XDNA 2** silicon without relying on discrete GPUs.

Forked from and built on the foundation of [FastFlowLM](https://github.com/ROCm/FastFlowLM) (ROCm), **VeloceNPU** introduces automated hardware autodetection for AIE2 architectures (Phoenix and Hawk Point), hybrid CPU/NPU execution fallbacks, and first-class support for multimodal model families including **Liquid Foundation Models (LFM3)**.

---

## 🚀 Key Features

- **Broad NPU Generation Support**:
  - **XDNA 1 / AIE2**: Phoenix and Hawk Point (4 compute columns + 1 memory column) with automatic PCI ID (`0x1502`) detection.
  - **XDNA 2 / AIE2P**: Strix, Strix Halo, and Krackan Point (8 columns) with full native bitstream acceleration.
- **Hybrid Execution Engine**:
  - Automatically identifies column topologies and seamlessly falls back to optimized AVX2/AVX-512 FMA vectorization when column-specific NPU bitstreams are not mapped, preventing hardware dispatch hangs.
- **Multimodal Pipeline Support**:
  - Full support for text, vision, and speech processing, including **LFM3-1.7B**, **LFM3-VL**, **LFM3-Audio**, and **LFM3-Omni**.
- **Minimal Footprint & Zero Overhead**:
  - Self-contained C++20 runtime (~17 MB) with no heavy Python runtime dependencies during inference.
- **Unified CLI & Server**:
  - Straightforward command-line interface for hardware validation, interactive chat, multimodal input processing, and OpenAI-compatible REST server serving.

---

## 🛠️ Quick Start

### 1. Prerequisites

- **Linux** (Kernel 6.10+ recommended with `amdxdna` driver) or **Windows 11** with AMD NPU driver.
- **CMake** (>= 3.22), **C++20** compatible compiler (GCC >= 12, Clang >= 15, or MSVC).
- **Boost** libraries (`boost-devel` / `brew install boost`).
- **Ninja** or GNU Make.

### 2. Clone and Build

```bash
git clone --recursive https://github.com/khoipm06/VeloceNPU.git
cd VeloceNPU

# Fetch runtime dependencies
./hrx-integration/fetch-hrx-release.sh

# Configure and build
cmake -B build -S src -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFLM_USE_HRX=ON \
  -DFLM_PORTABLE_BUILD=ON

ninja -C build flm
```

### 3. Verify Hardware

Check your NPU hardware, column topology, firmware, and driver stack:

```bash
./build/flm validate
```

Example output on an AMD Hawk Point (XDNA 1) system:
```text
[Linux]  Kernel: 7.x.x
[Linux]  NPU: /dev/accel/accel0 with 5 columns
[Linux]  NPU FW Version: 1.5.x
[Linux]  amdxdna version: 0.10
[Linux]  NPU platform: aie2
```

---

## 💬 Running Models

### List Available Models

```bash
./build/flm list
```

### Text Inference

```bash
./build/flm run lfm3:1.7b
```

### Vision, Audio & Multimodal

For vision and audio-enabled models, media files can be passed via `/input` in interactive mode:

```bash
# Vision-Language (LFM3-VL)
./build/flm run lfm3-vl:1.7b
>>> /input path/to/image.png Describe what is in this image.

# Audio (LFM3-Audio)
./build/flm run lfm3-audio:1.7b
>>> /input path/to/audio.wav Summarize this recording.

# Multimodal Omni (LFM3-Omni)
./build/flm run lfm3-omni:1.7b
>>> /input path/to/image.png Describe this scene.
```

### Serve via API

Start an OpenAI-compatible REST API server:

```bash
./build/flm serve lfm3:1.7b --port 52625
```

---

## ⚙️ Environment Variables

| Variable | Description |
| :--- | :--- |
| `VELOCE_PLATFORM` / `FLM_FORCE_PLATFORM` | Override detected platform (`aie2`, `aie2p`, `dev04`). |
| `FLM_MODEL_PATH` | Path to downloaded models directory (default: `~/.config/flm/models`). |
| `LFM3_PREFILL` | Prefill backend override (`host`, `npu`). |
| `LFM3_DECODE` | Decode backend override (`host`, `npu`). |

---

## 📄 License & Attribution

- Orchestration code, frontend interfaces, and CLI tools are licensed under the [MIT License](./LICENSE_RUNTIME.txt).
- Binary kernel runtimes are subject to the [FastFlowLM NPU Binary License](./LICENSE_BINARY.txt).
- Developer kit tools in [`flm-dev-kit/`](./flm-dev-kit/) are licensed under the [MIT License](./LICENSE_RUNTIME.txt).
- Powered by [FastFlowLM](https://github.com/ROCm/FastFlowLM).

<h1 align="center">SkyReels V3: Single Shot Video Extension (MPS)</h1>

<p align="center">
🤗 <a href="https://huggingface.co/Skywork/SkyReels-V3-Video-Extension" target="_blank">Model on Hugging Face</a> · 📑 <a href="https://arxiv.org/abs/2601.17323">Technical Report</a>
</p>

---

This is a trimmed-down fork of [SkyworkAI/SkyReels-V3](https://github.com/SkyworkAI/SkyReels-V3), stripped to just the **Single Shot Video Extension** model and adapted to run locally on Apple Silicon (MPS) instead of CUDA. The upstream repo also ships Reference-to-Video, Shot-Switching Extension, and Talking Avatar models — none of that code is included here.

Single Shot Video Extension takes an existing video and continues it with a coherent, logically consistent 5–30 second continuation, guided by a text prompt.

## Requirements

- macOS on Apple Silicon (M-series), for the MPS backend
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
git clone <this-repo-url>
cd SkyReels-V3
uv sync
```

This creates a `.venv` and installs all dependencies, including PyTorch built for MPS.

### Installing as a standalone CLI

To get a `skyreels-v3` command on your `PATH` (so it can be called from bash scripts without `uv run` or activating the venv):

```bash
uv tool install .
```

This installs an isolated environment plus a launcher script under `~/.local/bin` (make sure that's on your `PATH`). Afterwards:

```bash
skyreels-v3 --input_video my_clip.mp4 --duration 10 --prompt "..."
```

To pick up local code changes after editing the repo, reinstall with:

```bash
uv tool install . --reinstall
```

> **Note on "compiling" to a single binary:** this model depends on PyTorch, diffusers, and multi-gigabyte weights downloaded from Hugging Face, so it can't be frozen into a single self-contained executable (tools like PyInstaller aren't practical here — the resulting bundle would still need to unpack a full ML runtime and doesn't play well with MPS). `uv tool install` is the standard way to get a stable, PATH-resident CLI for a Python project like this.

## Model Download

The model is downloaded automatically from Hugging Face on first run (`Skywork/SkyReels-V3-Video-Extension`). To use a local copy instead, pass `--model_id /path/to/model`.

## Usage

```bash
uv run skyreels-v3 \
  --input_video https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4 \
  --prompt "A man is making his way forward slowly, leaning on a white cane to prop himself up." \
  --duration 10
```

(or just `skyreels-v3 ...` if installed via `uv tool install .`)

`--input_video` accepts a local path or a URL. Output is written to `result/single_shot_extension/`.

### Options

| Flag | Default | Description |
| --- | --- | --- |
| `--input_video` | example clip | Source video to extend (path or URL) |
| `--prompt` | example prompt | Text prompt describing the continuation |
| `--duration` | `5` | Output duration in seconds (5–30) |
| `--resolution` | `720P` | `480P`, `540P`, or `720P` |
| `--seed` | `42` | Random seed |
| `--dtype` | `fp16` | `fp16` or `bf16`. bf16 matches training but is less mature on MPS and can hit native matmul crashes — try `fp16` if you see an `MPSNDArrayMatrixMultiplication` assertion failure |
| `--chunk_seconds` | `2` | Seconds of video generated per roll. Attention memory scales roughly with the square of sequence length, so lowering this (e.g. to 2–3) trades speed for avoiding MPS out-of-memory errors |
| `--offload` | off | Offload models to reduce memory usage |
| `--low_vram` | off | Enable block offload for lower memory usage |
| `--model_id` | auto | Local path or HF model ID override |

### Memory tips

For constrained memory, lower `--resolution` (try `540P` or `480P`), reduce `--chunk_seconds`, and add `--offload` / `--low_vram`.

## Acknowledgements

This fork is based on [SkyworkAI/SkyReels-V3](https://github.com/SkyworkAI/SkyReels-V3), which in turn credits [Wan 2.1](https://github.com/Wan-Video/Wan2.1), [MultiTalk](https://github.com/MeiGen-AI/MultiTalk), [XDiT](https://github.com/xdit-project/xDiT), and [diffusers](https://github.com/huggingface/diffusers).

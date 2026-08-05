import gc
import logging
import os

import torch
from safetensors.torch import load_file

from .t5 import T5EncoderModel
from .transformer import WanModel
from .vae import WanVAE

# NOTE: get_image_encoder() (CLIPModel, for image-to-video conditioning) has
# been removed. It was never called by SingleShotExtensionPipeline, and
# clip.py / xlm_roberta.py aren't needed for single-shot video extension.


def _empty_cache(device: str):
    """Free cached memory on whichever backend `device` refers to.

    torch.cuda.empty_cache() is CUDA-only and does nothing useful (and isn't
    even guaranteed to exist as a no-op) on other backends. MPS has its own
    equivalent; anything else (cpu, meta) has nothing to empty.
    """
    if isinstance(device, str) and device.startswith("mps"):
        torch.mps.empty_cache()
    elif isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.empty_cache()


def download_model(model_id):
    if not os.path.exists(model_id):
        from huggingface_hub import snapshot_download

        model_id = snapshot_download(repo_id=model_id)
    return model_id


def get_vae(
    model_path, subfolder="", device="mps", weight_dtype=torch.float32
) -> WanVAE:
    model_path = os.path.join(model_path, subfolder) if subfolder else model_path
    vae = WanVAE(model_path).to(device).to(weight_dtype)
    vae.vae.requires_grad_(False)
    vae.vae.eval()
    gc.collect()
    _empty_cache(device)
    return vae


def get_transformer(
    model_path, subfolder="", device="mps", weight_dtype=torch.bfloat16, low_vram=False
) -> WanModel:
    model_path = os.path.join(model_path, subfolder) if subfolder else model_path
    config_path = os.path.join(model_path, "config.json")
    logging.info(f"loading transformer from {config_path}, model_path: {model_path}")
    transformer = WanModel.from_config(config_path).to(weight_dtype).to(device)

    for file in os.listdir(model_path):
        if file.endswith(".safetensors"):
            file_path = os.path.join(model_path, file)
            state_dict = load_file(file_path)
            transformer.load_state_dict(state_dict, strict=False)
            del state_dict
            gc.collect()
            _empty_cache(device)

    transformer.requires_grad_(False)
    transformer.eval()
    if low_vram:
        is_cuda = isinstance(device, str) and device.startswith("cuda")
        if is_cuda:
            from torchao.quantization import float8_weight_only, quantize_

            quantize_(transformer, float8_weight_only(), device=device)
            transformer.to(device)
        else:
            # torchao's FP8 weight-only quantization targets CUDA tensor
            # cores; PyTorch/MPS has no float8 support at all, so this can't
            # be ported, only skipped. On MPS the low_vram memory savings
            # come from the block_offload mechanism in transformer.py
            # instead (moving one transformer block at a time on/off the
            # device), which is already device-agnostic, so low_vram is
            # still useful here even without FP8 quantization.
            logging.info(
                f"low_vram FP8 weight quantization requires CUDA; skipping it "
                f"on device={device!r}. Memory savings will come from "
                f"block_offload instead."
            )
    gc.collect()
    _empty_cache(device)
    return transformer


def get_text_encoder(
    model_path, subfolder="", device="mps", weight_dtype=torch.bfloat16
) -> T5EncoderModel:
    model_path = os.path.join(model_path, subfolder) if subfolder else model_path
    t5_model = os.path.join(model_path, "models_t5_umt5-xxl-enc-bf16.pth")
    tokenizer_path = os.path.join(model_path, "google", "umt5-xxl")
    text_encoder = (
        T5EncoderModel(checkpoint_path=t5_model, tokenizer_path=tokenizer_path)
        .to(device)
        .to(weight_dtype)
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    gc.collect()
    _empty_cache(device)
    return text_encoder

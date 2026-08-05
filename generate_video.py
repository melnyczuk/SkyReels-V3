import argparse
import logging
import os
import random
import time

# 配置日志格式和级别，实现实时终端打印
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - skyreels_v3 - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
    handlers=[logging.StreamHandler()],  # 显式指定输出到终端
)

import imageio
import torch
import wget

from skyreels_v3.modules import download_model
from skyreels_v3.pipeline import SingleShotExtensionPipeline


def maybe_download(path_or_url: str, save_dir: str) -> str:
    """
    If `path_or_url` is already a local path, return it.
    Otherwise, download it into `save_dir` and return the downloaded local path.
    """
    if os.path.exists(path_or_url):
        return path_or_url

    url = path_or_url
    filename = url.split("/")[-1]
    local_path = os.path.join(save_dir, filename)
    logging.info(f"downloading input: {local_path}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        logging.info(f"input already exists: {local_path}")
        return local_path

    wget.download(url, local_path)
    assert os.path.exists(local_path), f"Failed to download input: {url}"
    logging.info(f"finished downloading input: {local_path}")
    return local_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SkyReels V3: Single Shot Video Extension (MPS / local)"
    )

    # ==================== Model Configuration ====================
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model path or HuggingFace model ID. Defaults to "
        "Skywork/SkyReels-V3-Video-Extension.",
    )

    # ==================== Generation Parameters ====================
    parser.add_argument(
        "--duration",
        type=int,
        default=5,
        help="Output video duration in seconds (5-30s).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A man is making his way forward slowly, leaning on a white cane to prop himself up.",
        help="Text prompt describing the desired video content.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720P",
        choices=["480P", "540P", "720P"],
        help="Output video resolution. Lower resolution recommended for low VRAM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )

    # ==================== Performance & Memory Options ====================
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Enable model offloading to reduce memory usage.",
    )
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Enable low VRAM mode with block offload.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["bf16", "fp16"],
        help="Model weight/compute dtype. bf16 is the default (matches how "
        "the model was trained), but bf16 support on the MPS backend is "
        "known to be less mature than fp16 and can hit native crashes in "
        "some matmul kernels. Try --dtype fp16 if you hit an MPS assertion "
        "failure like 'MPSNDArrayMatrixMultiplication'.",
    )

    parser.add_argument(
        "--chunk_seconds",
        type=int,
        default=2,
        help="Seconds of video generated per roll. Lower this (e.g. 2 or 3) "
        "if you hit MPS out-of-memory errors — attention memory scales "
        "roughly with the square of the sequence length, so smaller chunks "
        "help a lot at the cost of more, slightly slower rolls.",
    )

    # ==================== Video Extension Parameters ====================
    parser.add_argument(
        "--input_video",
        type=str,
        default="https://skyreels-api.oss-accelerate.aliyuncs.com/examples/video_extension/test.mp4",
        help="Input video path or URL to extend.",
    )

    args = parser.parse_args()

    args.model_id = "Skywork/SkyReels-V3-Video-Extension"
    device = "mps"

    args.model_id = download_model(args.model_id)
    print(f"args.model_id: {args.model_id}")

    if args.seed is None:
        random.seed(time.time())
        args.seed = int(random.randrange(4294967294))

    logging.info(f"input params: {args}")

    args.input_video = maybe_download(args.input_video, "input_video")

    # init pipeline
    pipe = SingleShotExtensionPipeline(
        model_path=args.model_id,
        offload=args.offload,
        low_vram=args.low_vram,
        device=device,
        weight_dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float16,
    )
    video_out = pipe.extend_video(
        args.input_video,
        args.prompt,
        args.duration,
        args.seed,
        resolution=args.resolution,
        chunk_seconds=args.chunk_seconds,
    )

    save_dir = os.path.join("result", "single_shot_extension")
    os.makedirs(save_dir, exist_ok=True)

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    video_out_file = f"{args.seed}_{current_time}.mp4"
    output_path = os.path.join(save_dir, video_out_file)
    imageio.mimwrite(
        output_path,
        video_out,
        fps=24,
        quality=8,
        output_params=["-loglevel", "error"],
    )

    print(f"saved video to: {output_path}")

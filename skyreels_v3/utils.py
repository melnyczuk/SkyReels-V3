import av
import numpy as np
import torch
from collections import deque
from PIL import Image

from .config import ASPECT_RATIO_CONFIG


def get_prefix_video(input_video_path: str, num_condition_frames: int):
    # Only the last `num_condition_frames` frames are ever used as
    # conditioning. Previously this decoded and kept EVERY frame of the
    # input video in RAM (as an unused `raw_video` array — never referenced
    # anywhere downstream), which scaled with input video length for no
    # reason. A bounded deque only ever holds the last num_condition_frames
    # decoded frames at once; older frames are dropped (and GC'd) as new
    # ones arrive, so this is roughly O(num_condition_frames) memory
    # regardless of how long the input video is.
    container = av.open(input_video_path)
    stream = container.streams.video[0]
    prefix_buffer = deque(maxlen=num_condition_frames)
    for frame in container.decode(stream):
        prefix_buffer.append(frame.to_ndarray(format="rgb24"))
    prefix_video = np.stack(list(prefix_buffer))  # [N, H, W, 3]
    return prefix_video


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(
        ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio)
    )
    return closest_ratio


def get_height_width_from_image(image: Image.Image, resolution: str = "720P"):
    assert resolution in ASPECT_RATIO_CONFIG, f"Resolution {resolution} not supported"
    aspect_ratio = ASPECT_RATIO_CONFIG[resolution]
    width, height = image.size
    closest_ratio = get_closest_ratio(height, width, aspect_ratio)
    height, width = aspect_ratio[closest_ratio]
    height = height // 8 // 2 * 2 * 8
    width = width // 8 // 2 * 2 * 8
    return height, width


def process_video(prefix_video, ASPECT_RATIO):
    # prepare for VAE
    prefix_video = (
        torch.tensor(prefix_video).permute(3, 0, 1, 2).unsqueeze(0).float()
    )  # 1, C, T, H, W
    prefix_video = prefix_video / (255.0 / 2.0) - 1.0
    # resize
    h, w = prefix_video.shape[-2:]
    height, width = ASPECT_RATIO[get_closest_ratio(h, w, ASPECT_RATIO)]
    height = height // 8 // 2 * 2 * 8
    width = width // 8 // 2 * 2 * 8
    prefix_video = torch.nn.functional.interpolate(
        prefix_video, size=(prefix_video.shape[2], height, width)
    )
    return prefix_video, height, width


def get_video_info(input_video_path: str, num_condition_frames: int, resolution: str):
    prefix_video = get_prefix_video(input_video_path, num_condition_frames)
    assert resolution in ASPECT_RATIO_CONFIG, f"Resolution {resolution} not supported"
    aspect_ratio = ASPECT_RATIO_CONFIG[resolution]
    prefix_video, height, width = process_video(prefix_video, aspect_ratio)
    return prefix_video, height, width

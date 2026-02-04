from dataclasses import dataclass
from typing import Literal


DeviceType = Literal["cpu", "cuda", "auto"]


@dataclass
class IAConfig:
    image_caption_model: str = "Salesforce/blip-image-captioning-base"
    text_rewrite_model: str = "google/flan-t5-base"
    device: DeviceType = "auto"


def resolve_device(config_device: DeviceType) -> str:
    if config_device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return config_device


DEFAULT_CONFIG = IAConfig()


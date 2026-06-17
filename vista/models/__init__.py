from ultralytics import YOLOE

from .moondream import MoonDream
from .omdetturbo import OmDetTurbo
from .sam.src.sam3_model import Sam3Model
from .yoloe import YOLOEVista
from .yolo import YOLOVista
from .rtdetr import RTDETRVista
from .locate_anything import LocateAnythingWrapper

MODEL_ZOO = {
    "moondream": MoonDream,
    "omdetturbo": OmDetTurbo,
    "sam": Sam3Model,
    "yoloe": YOLOEVista,
    "yolo": YOLOVista,
    "yolo26": YOLOVista,  # Maps the yolo26 identifier to the standard YOLO wrapper
    "rtdetr": RTDETRVista,
    "locate_anything": LocateAnythingWrapper,
}

def get_model(parameters: dict) -> object:
    model_name = parameters.get("name")
    kwargs = {k: v for k, v in parameters.items() if k != "name"}
    if model_name not in MODEL_ZOO:
        raise ValueError(f"Model '{model_name}' not found in MODEL_ZOO. Available models: {list(MODEL_ZOO.keys())}")
    return MODEL_ZOO[model_name](**kwargs)
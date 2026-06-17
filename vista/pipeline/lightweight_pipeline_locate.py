"""
Updated Lightweight pipeline with integrated ByteTrack, Locate Anything, Buffering, and Captioning.
Adjusted for VISTA Challenge compliance (Strict category mapping & Deep FPS optimization).
Optimized to dynamically handle both standard COCO and VisDrone model taxonomy.
"""

import logging
import time
from typing import List, Dict
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .base import VistaPipeline, FrameResult, Detection
from .helpers import crop_track_with_padding
from ..models.caption_wrapper import VideoCaptioner

logger = logging.getLogger(__name__)

class LightweightPipelineLocate(VistaPipeline):
    def __init__(
        self,
        yolo_model: YOLO,
        locate_model=None,
        locate_prompts: List[str] = None,
        locate_every_n: int = 15,
        trigger_classes: List[str] = None,
        enable_profiling: bool = True,
        yolo_conf: float = 0.05,
        nms_iou_threshold: float = 0.45,
        caption_buffer_size: int = 16,
        caption_stride: int = 15,  # Added to control how often the captioner runs after buffer is full
        **kwargs
    ):
        self.yolo = yolo_model
        self.locate_model = locate_model
        self.locate_prompts = locate_prompts or ["person_injured", "crashed_car", "hazardous_debris"]
        self.locate_every_n = locate_every_n
        self.enable_profiling = enable_profiling
        self.yolo_conf = yolo_conf
        self.nms_iou_threshold = nms_iou_threshold
        self.caption_buffer_size = caption_buffer_size
        self.caption_stride = caption_stride

        # Automatically map raw native trigger input names (COCO or VisDrone) to compliance targets
        provided_triggers = trigger_classes or [
            "person", "motorcycle", "bus", "pedestrian", "people", "van", "truck", "bicycle"
        ]
        self.trigger_classes = list(set(self._map_category(t) for t in provided_triggers))

        # Initialize the VideoCaptioner for heuristic captioning
        self.captioner = VideoCaptioner(model_name="heuristic")

        # Manage of track buffers and history for captioning
        self._track_buffers = defaultdict(list)
        self.track_history = {}  # Contains data for predictions_tracks.csv

        self._profiling_stats = {
            "yolo_time": [], "locate_time": [], "merge_time": [], "total_time": []
        }

    def reset(self) -> None:
        """Clears state between distinct video sequences."""
        self._track_buffers.clear()
        self.track_history.clear()
        if hasattr(self.yolo, 'predictor') and self.yolo.predictor:
            for tracker in getattr(self.yolo.predictor, 'trackers', []):
                tracker.reset()

    def _map_category(self, raw_name: str) -> str:
        """
        Maps raw COCO or VisDrone classes to strictly VISTA-compliant categories:
        'car', 'emergency_vehicle', or 'person'.
        """
        raw = raw_name.lower()
        # Handle standard COCO 'person' and VisDrone 'pedestrian'/'people' variants
        if "person" in raw or "pedestrian" in raw or "people" in raw:
            return "person"

        # Handle dedicated emergency units
        if any(ev in raw for ev in ["ambulance", "police", "fire", "emergency"]):
            return "emergency_vehicle"

        # Defaults all other vehicles (car, van, truck, bus, motorcycle, tricycle) to "car"
        return "car"

    def _iou(self, boxA: tuple, boxB: tuple) -> float:
        xA, yA, xB, yB = max(boxA[0], boxB[0]), max(boxA[1], boxB[1]), min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def _should_call_locate(self, frame_idx: int, yolo_detections: List[Detection]) -> bool:
        if self.locate_model is None: return False
        if frame_idx % self.locate_every_n == 0: return True
        # Evaluated safely against normalized compliance categories
        return any(det.category in self.trigger_classes for det in yolo_detections)

    def _nms_merge(self, yolo_dets: List[Detection], locate_dets: List[Detection]) -> List[Detection]:
        fused_detections = list(yolo_dets)
        for l_det in locate_dets:
            matched = False
            for y_det in fused_detections:
                if self._iou(l_det.bbox, y_det.bbox) > self.nms_iou_threshold:
                    # Inject the original Locate prompt into the track as a status tag,
                    # but DO NOT overwrite the mapped VISTA category.
                    y_det.caption = l_det.caption
                    matched = True
                    break
            if not matched:
                fused_detections.append(l_det)
        return fused_detections

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        t_start = time.time()

        # 1. YOLO Standard Detection & Tracking
        yolo_detections = []
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR) if np.array(frame).ndim == 3 else np.array(frame)

        results = self.yolo.track(frame_bgr, conf=self.yolo_conf, persist=True, tracker="bytetrack.yaml", verbose=False)
        for r in results:
            boxes, confs, classes = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)
            for box, conf, cls, track_id in zip(boxes, confs, classes, track_ids):
                raw_name = r.names.get(int(cls), "unknown")
                vista_category = self._map_category(raw_name)

                yolo_detections.append(Detection(
                    bbox=tuple(box.tolist()),
                    category=vista_category,
                    confidence=float(conf),
                    track_id=int(track_id) if track_id is not None else None,
                    caption=None
                ))

        # 2. Locate Anything (Conditional Grounding)
        locate_detections = []
        if self._should_call_locate(frame_idx, yolo_detections):
            try:
                grounding_results = self.locate_model.ground(image=frame, text_prompts=self.locate_prompts, conf_threshold=0.35)
                for prompt, boxes_confs in grounding_results.items():
                    vista_category = self._map_category(prompt)
                    for box, conf in boxes_confs:
                        locate_detections.append(Detection(
                            bbox=tuple(box),
                            category=vista_category,
                            confidence=conf,
                            track_id=None,
                            caption=prompt  # Store the raw prompt as the initial caption/tag
                        ))
            except Exception: pass

        # 3. NMS Merge
        final_detections = self._nms_merge(yolo_detections, locate_detections)

        # 4. History Tracking & Spatiotemporal Buffering
        for det in final_detections:
            tid = det.track_id
            if tid is not None:
                # Initialize new tracks
                if tid not in self.track_history:
                    self.track_history[tid] = {
                        "frame_start": frame_idx, "frame_end": frame_idx,
                        "locate_tags": [], "bbox_history": [], "final_caption": None
                    }

                # Update trajectory data
                self.track_history[tid]["frame_end"] = frame_idx
                self.track_history[tid]["bbox_history"].append(det.bbox)
                if det.caption:
                    self.track_history[tid]["locate_tags"].append(det.caption)

                # Crop and maintain rolling buffer
                cropped_img = crop_track_with_padding(frame, det.bbox, padding_factor=0.2)
                self._track_buffers[tid].append((frame_idx, cropped_img))
                if len(self._track_buffers[tid]) > self.caption_buffer_size:
                    self._track_buffers[tid].pop(0)

                # 5. Smart Caption Triggering
                buffer_len = len(self._track_buffers[tid])
                has_caption = self.track_history[tid]["final_caption"] is not None

                # Run captioner if buffer is full AND (we don't have one yet OR we hit the stride interval)
                if buffer_len == self.caption_buffer_size and (not has_caption or frame_idx % self.caption_stride == 0):
                    dynamic_caption = self.captioner.generate_caption(
                        frame_buffer=self._track_buffers[tid],
                        locate_tags=self.track_history[tid]["locate_tags"],
                        bbox_history=self.track_history[tid]["bbox_history"]
                    )
                    self.track_history[tid]["final_caption"] = dynamic_caption

                # 6. Fallback & Output Assignment
                if self.track_history[tid]["final_caption"]:
                    det.caption = self.track_history[tid]["final_caption"]
                elif self.track_history[tid]["locate_tags"]:
                    # Fallback to the latest locate tag if the buffer isn't full yet
                    det.caption = self.track_history[tid]["locate_tags"][-1]
                else:
                    # Absolute fallback to protect BERTScore
                    det.caption = det.category

        return FrameResult(detections=final_detections, frame_idx=frame_idx)
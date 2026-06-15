"""
Updated Lightweight pipeline with integrated ByteTrack tracking and Locate Anything.
Scope: YOLO detection + tracking + selective grounded visual understanding.
"""

import logging
import time
from typing import List, Dict

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from .base import VistaPipeline, FrameResult, Detection
# Assuming LocateAnythingWrapper is importable from vista.models or similar
# from vista.models.locate_anything import LocateAnythingWrapper

logger = logging.getLogger(__name__)

class LightweightPipelineLocate(VistaPipeline):
    """
    YOLO detector with integrated ByteTrack tracking and selective Locate Anything grounding.
    Target performance: ≥5 FPS on shallow stage (detection + tracking + selective locate).
    """

    def __init__(
        self,
        yolo_model: YOLO,
        locate_model=None,
        locate_prompts: List[str] = None,
        locate_every_n: int = 3,
        trigger_classes: List[str] = None,
        enable_profiling: bool = True,
        yolo_conf: float = 0.05,
        nms_iou_threshold: float = 0.45,
        **kwargs
    ):
        self.yolo = yolo_model
        self.locate_model = locate_model
        self.locate_prompts = locate_prompts or ["injured person", "ambulance", "debris"]
        self.locate_every_n = locate_every_n
        self.trigger_classes = trigger_classes or ["crashed_car", "person"]
        self.enable_profiling = enable_profiling
        self.yolo_conf = yolo_conf
        self.nms_iou_threshold = nms_iou_threshold

        self._profiling_stats = {
            "yolo_time": [],
            "locate_time": [],
            "merge_time": [],
            "total_time": []
        }

        logger.info(f"Pipeline initialized: YOLO={yolo_model.model_name} with Tracking & Selective Locate.")

    def reset(self) -> None:
        """Clear pipeline state and tracking history between videos."""
        if self.enable_profiling:
            self._profiling_stats = {k: [] for k in self._profiling_stats}

        # Reset the Ultralytics tracker state to avoid cross-video ID leakage
        if hasattr(self.yolo, 'predictor') and self.yolo.predictor:
            for tracker in getattr(self.yolo.predictor, 'trackers', []):
                tracker.reset()

    def _iou(self, boxA: tuple, boxB: tuple) -> float:
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def _should_call_locate(self, frame_idx: int, yolo_detections: List[Detection]) -> bool:
        """Multi-condition trigger for the Locate Anything model."""
        if self.locate_model is None:
            return False

        # Trigger 1: Temporal frequency (e.g., every 3 frames)
        if frame_idx % self.locate_every_n == 0:
            return True

        # Trigger 2: High-priority YOLO detection (e.g., accident scenario classes)
        for det in yolo_detections:
            if det.category in self.trigger_classes:
                return True

        return False

    def _nms_merge(self, yolo_dets: List[Detection], locate_dets: List[Detection]) -> List[Detection]:
        """
        Fuse YOLO and Locate detections using IoU NMS.
        Prioritizes YOLO boxes for tracker consistency, but enriches them with Locate captions.
        """
        fused_detections = list(yolo_dets)

        for l_det in locate_dets:
            matched = False
            for y_det in fused_detections:
                if self._iou(l_det.bbox, y_det.bbox) > self.nms_iou_threshold:
                    # Enrich existing YOLO track with fine-grained Locate vocabulary
                    y_det.caption = l_det.category
                    matched = True
                    break

            # If the Locate detection doesn't overlap heavily with YOLO, add it as a new box
            if not matched:
                fused_detections.append(l_det)

        return fused_detections

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        t_start_total = time.time()

        # --- 1. YOLO & Tracking Stage ---
        t_yolo_start = time.time()
        yolo_detections: List[Detection] = []
        frame_np = np.array(frame)

        if frame_np.ndim == 3 and frame_np.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_np

        try:
            results = self.yolo.track(
                frame_bgr,
                conf=self.yolo_conf,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            for r in results:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)

                for box, conf, cls, track_id in zip(boxes, confs, classes, track_ids):
                    category = r.names.get(int(cls), "unknown")
                    yolo_detections.append(
                        Detection(
                            bbox=tuple(box.tolist()),
                            category=category,
                            confidence=float(conf),
                            track_id=int(track_id) if track_id is not None else None,
                            caption=None
                        )
                    )
        except Exception as e:
            logger.error(f"YOLO tracking failed on frame {frame_idx}: {e}")

        yolo_time = time.time() - t_yolo_start

        # --- 2. Selective Locate Anything Stage ---
        t_locate_start = time.time()
        locate_detections: List[Detection] = []

        if self._should_call_locate(frame_idx, yolo_detections):
            try:
                grounding_results = self.locate_model.ground(
                    image=frame,
                    text_prompts=self.locate_prompts,
                    conf_threshold=0.3
                )
                for prompt, boxes_confs in grounding_results.items():
                    for box, conf in boxes_confs:
                        locate_detections.append(
                            Detection(
                                bbox=tuple(box),
                                category=prompt,
                                confidence=conf,
                                track_id=None,  # Locate doesn't track natively
                                caption=prompt
                            )
                        )
            except Exception as e:
                logger.error(f"Locate Anything failed on frame {frame_idx}: {e}")

        locate_time = time.time() - t_locate_start

        # --- 3. NMS Fusion Stage ---
        t_merge_start = time.time()
        final_detections = self._nms_merge(yolo_detections, locate_detections)
        merge_time = time.time() - t_merge_start

        # --- 4. Profiling ---
        total_time = time.time() - t_start_total
        if self.enable_profiling:
            self._profiling_stats["yolo_time"].append(yolo_time)
            self._profiling_stats["locate_time"].append(locate_time)
            self._profiling_stats["merge_time"].append(merge_time)
            self._profiling_stats["total_time"].append(total_time)

            # Log FPS periodically or simply log latencies
            fps = 1.0 / total_time if total_time > 0 else 0
            logger.debug(f"Frame {frame_idx} | Total: {total_time*1000:.1f}ms ({fps:.1f} FPS) | "
                         f"YOLO: {yolo_time*1000:.1f}ms | Locate: {locate_time*1000:.1f}ms | Merge: {merge_time*1000:.1f}ms")

        return FrameResult(
            detections=final_detections,
            frame_idx=frame_idx
        )
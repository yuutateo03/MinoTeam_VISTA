"""
Updated Lightweight pipeline with integrated ByteTrack tracking.
Scope: YOLO detection + tracking.
"""

import logging
import time
from typing import List

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from vista.pipeline.base import VistaPipeline, FrameResult, Detection

logger = logging.getLogger(__name__)

class LightweightPipelineLocate(VistaPipeline):
    """
    YOLO detector with integrated ByteTrack tracking.
    Target performance: ≥5 FPS on shallow stage (detection + tracking).
    """

    def __init__(
        self,
        yolo_model: YOLO,
        locate_model=None,
        enable_profiling: bool = True,
        yolo_conf: float = 0.05,
        **kwargs
    ):
        self.yolo = yolo_model
        self.enable_profiling = enable_profiling
        self.yolo_conf = yolo_conf

        self._profiling_stats = {
            "total_time": []
        }

        logger.info(f"Pipeline initialized: YOLO={yolo_model.model_name} with Tracking enabled.")

    def reset(self) -> None:
        """Clear pipeline state and tracking history between videos."""
        if self.enable_profiling:
            self._profiling_stats["total_time"] = []

        # Reset the Ultralytics tracker state to avoid cross-video ID leakage
        if hasattr(self.yolo, 'predictor') and self.yolo.predictor:
            for tracker in getattr(self.yolo.predictor, 'trackers', []):
                tracker.reset()

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        t_start = time.time()
        detections: List[Detection] = []

        # 1. Image Conversion
        frame_np = np.array(frame)
        if frame_np.ndim == 3 and frame_np.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_np

        # 2. YOLO Inference with Tracking
        try:
            # persist=True is crucial for continuous tracking across frames
            # tracker="bytetrack.yaml" forces the ByteTrack algorithm
            results = self.yolo.track(
                frame_bgr,
                conf=self.yolo_conf,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )

            for r in results:
                # Extract boxes, confidences, classes, and IDs safely
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()

                # IDs might be None if the tracker hasn't confidently assigned one yet
                track_ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else [None] * len(boxes)

                for box, conf, cls, track_id in zip(boxes, confs, classes, track_ids):
                    category = r.names.get(int(cls), "unknown")

                    detections.append(
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

        # 3. Profiling
        if self.enable_profiling:
            total_time = time.time() - t_start
            self._profiling_stats["total_time"].append(total_time)

        return FrameResult(
            detections=detections,
            frame_idx=frame_idx
        )
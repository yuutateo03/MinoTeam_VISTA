"""Lightweight pipeline with selective Locate Anything grounding."""

from __future__ import annotations
from typing import List, Dict, Optional
import logging
import time

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

from vista.pipeline.base import VistaPipeline, FrameResult, Detection
from vista.models.locate_anything import LocateAnythingWrapper

logger = logging.getLogger(__name__)


class LightweightPipelineLocate(VistaPipeline):
    """
    YOLO detector + selective Locate Anything grounding + ByteTrack + captioning.
    
    This pipeline combines fast YOLO detection with selective high-precision
    Locate Anything grounding to balance speed and accuracy on accident scene videos.
    
    Target performance: ≥5 FPS on shallow stage (detection + tracking)
    """
    
    def __init__(
        self,
        yolo_model: YOLO,
        locate_model: Optional[LocateAnythingWrapper] = None,
        tracker = None,
        caption_model = None,
        locate_every_n: int = 3,
        locate_conf_threshold: float = 0.45,
        locate_iou_merge: float = 0.4,
        yolo_conf_low_trigger: float = 0.3,
        batch_prompts: Optional[Dict[str, str]] = None,
        enable_profiling: bool = True,
    ):
        """Initialize the lightweight pipeline with optional Locate Anything grounding."""
        self.yolo = yolo_model
        self.locate = locate_model
        self.tracker = tracker
        self.captioner = caption_model
        
        # Selective grounding configuration
        self.locate_every_n = locate_every_n
        self.locate_conf_threshold = locate_conf_threshold
        self.locate_iou_merge = locate_iou_merge
        self.yolo_conf_low_trigger = yolo_conf_low_trigger
        self.batch_prompts = batch_prompts or {}
        self.enable_profiling = enable_profiling
        
        # Per-video state
        self._track_db = {}
        self._profiling_stats = {
            "yolo_time": [],
            "locate_time": [],
            "nms_time": [],
            "total_time": [],
        }
        
        logger.info(
            f"LightweightPipelineLocate initialized: "
            f"YOLO={yolo_model.model_name}, "
            f"Locate={'enabled' if locate_model else 'disabled'}, "
            f"locate_every_n={locate_every_n}"
        )
    
    def reset(self) -> None:
        """Clear per-video state (called before processing each new video)."""
        self._track_db.clear()
        if self.enable_profiling:
            self._profiling_stats = {
                "yolo_time": [],
                "locate_time": [],
                "nms_time": [],
                "total_time": [],
            }
    
    def _should_call_locate(self, frame_idx: int, yolo_dets: List[Detection]) -> bool:
        """
        Decide whether to call Locate Anything on this frame.
        
        Conditions (OR logic):
        - Frame index is multiple of locate_every_n
        - Any YOLO detection has low confidence
        - Occlusion detected (many overlapping boxes)
        """
        
        # Condition A: Every N frames
        if frame_idx % self.locate_every_n == 0:
            return True
        
        # Condition B: Low-confidence YOLO detections
        for det in yolo_dets:
            if det.confidence < self.yolo_conf_low_trigger:
                return True
        
        # Condition C: Occlusion heuristic
        if len(yolo_dets) > 1:
            overlaps = 0
            for i, det1 in enumerate(yolo_dets):
                for det2 in yolo_dets[i + 1 :]:
                    if self._iou(det1.bbox, det2.bbox) > 0.2:
                        overlaps += 1
            
            if len(yolo_dets) > 0 and overlaps > len(yolo_dets) * 0.3:
                return True
        
        return False
    
    @staticmethod
    def _iou(box1: tuple, box2: tuple) -> float:
        """Compute Intersection over Union (IoU) between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x1 = max(x1_min, x2_min)
        inter_y1 = max(y1_min, y2_min)
        inter_x2 = min(x1_max, x2_max)
        inter_y2 = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _nms_merge(self, detections: List[Detection], iou_threshold: float) -> List[Detection]:
        """Non-Maximum Suppression to remove duplicate/overlapping detections."""
        if not detections:
            return []
        
        # Sort by confidence descending
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []
        
        for det in sorted_dets:
            overlaps_with_kept = False
            for kept in keep:
                if self._iou(det.bbox, kept.bbox) > iou_threshold:
                    overlaps_with_kept = True
                    break
            
            if not overlaps_with_kept:
                keep.append(det)
        
        return keep
    
    def _map_prompt_to_category(self, prompt: str) -> str:
        """Map a Locate Anything prompt to a VISTA category."""
        prompt_lower = prompt.lower()
        
        if "person" in prompt_lower:
            return "person"
        elif (
            "ambulance" in prompt_lower
            or "emergency" in prompt_lower
            or "police" in prompt_lower
            or "fire" in prompt_lower
        ):
            return "emergency_vehicle"
        elif "car" in prompt_lower or "damage" in prompt_lower or "vehicle" in prompt_lower:
            return "car"
        
        return "unknown"
    
    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        """Process a single frame through the full pipeline."""
        t_start = time.time()
        detections = []
        
        # ──────────────────────────────────────────────────────────────────
        # Stage 1: YOLO Detection (fast, every frame)
        # ──────────────────────────────────────────────────────────────────
        
        t_yolo_start = time.time()
        
        frame_np = np.array(frame)
        if frame_np.ndim == 3 and frame_np.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame_np
        
        try:
            results = self.yolo.predict(frame_bgr, conf=0.05, verbose=False)
            
            for r in results:
                for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                    bbox = tuple(box.cpu().numpy().tolist())
                    category = r.names.get(int(cls), "unknown")
                    detections.append(
                        Detection(
                            bbox=bbox,
                            category=category,
                            confidence=float(conf),
                            track_id=None,
                            caption=None,
                        )
                    )
        except Exception as e:
            logger.error(f"YOLO detection failed on frame {frame_idx}: {e}")
            detections = []
        
        yolo_time = time.time() - t_yolo_start
        
        # ──────────────────────────────────────────────────────────────────
        # Stage 2: Optional Locate Anything Grounding (selective)
        # ──────────────────────────────────────────────────────────────────
        
        locate_time = 0.0
        
        if (
            self.locate is not None
            and not self.locate.is_dummy
            and self._should_call_locate(frame_idx, detections)
        ):
            t_locate_start = time.time()
            
            try:
                prompts = list(self.batch_prompts.values())
                
                if prompts:
                    locate_out = self.locate.ground(
                        image=frame,
                        text_prompts=prompts,
                        return_multiple=True,
                        conf_threshold=self.locate_conf_threshold,
                    )
                    
                    # Convert Locate results to Detection objects
                    for prompt, hits in locate_out.items():
                        for bbox, conf in hits:
                            category = self._map_prompt_to_category(prompt)
                            detections.append(
                                Detection(
                                    bbox=tuple(bbox),
                                    category=category,
                                    confidence=float(conf),
                                    track_id=None,
                                    caption=None,
                                )
                            )
                    
                    logger.debug(
                        f"Frame {frame_idx}: Locate grounding completed. "
                        f"Results={sum(len(h) for h in locate_out.values())}"
                    )
                
            except Exception as e:
                logger.warning(f"Locate Anything grounding failed on frame {frame_idx}: {e}")
            
            locate_time = time.time() - t_locate_start
        
        # ──────────────────────────────────────────────────────────────────
        # Stage 3: NMS Merge (deduplicate YOLO + Locate detections)
        # ──────────────────────────────────────────────────────────────────
        
        t_nms_start = time.time()
        merged_detections = self._nms_merge(detections, iou_threshold=self.locate_iou_merge)
        nms_time = time.time() - t_nms_start
        
        # ──────────────────────────────────────────────────────────────────
        # Stage 4 & 5: ByteTrack + Captioning (TODO: implement)
        # ──────────────────────────────────────────────────────────────────
        
        # Profiling & Logging
        if self.enable_profiling:
            total_time = time.time() - t_start
            self._profiling_stats["yolo_time"].append(yolo_time)
            self._profiling_stats["locate_time"].append(locate_time)
            self._profiling_stats["nms_time"].append(nms_time)
            self._profiling_stats["total_time"].append(total_time)
            
            if frame_idx % 30 == 0:
                window_size = 30
                avg_yolo = np.mean(self._profiling_stats["yolo_time"][-window_size:]) if self._profiling_stats["yolo_time"] else 0
                avg_locate = np.mean(self._profiling_stats["locate_time"][-window_size:]) if self._profiling_stats["locate_time"] else 0
                avg_total = np.mean(self._profiling_stats["total_time"][-window_size:]) if self._profiling_stats["total_time"] else 0
                fps = 1.0 / avg_total if avg_total > 0 else 0
                
                logger.info(
                    f"Frame {frame_idx:5d}: FPS={fps:6.1f}, "
                    f"YOLO={avg_yolo*1000:6.1f}ms, "
                    f"Locate={avg_locate*1000:6.1f}ms, "
                    f"Detections={len(merged_detections)}"
                )
        
        return FrameResult(detections=merged_detections, frame_idx=frame_idx)
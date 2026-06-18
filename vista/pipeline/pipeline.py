import torch
import numpy as np
from PIL import Image
import supervision as sv
from transformers import CLIPProcessor, CLIPModel

# Import base classes from your competition framework
from MinoTeam_VISTA.vista.pipeline.base import VistaPipeline, FrameResult, Detection


# ─── 1. LIGHTWEIGHT DETECTOR & TRACKER WRAPPER ────────────────────────────────

class VisDroneYOLODetector:
    def __init__(self, model_path="yolo26x_visdrone.pt"):
        """
        Loads YOLO model finetuned on the VisDrone data split.
        """
        from ultralytics import YOLO
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading VisDrone YOLO Detector: {model_path} on {self.device}...")
        self.model = YOLO(model_path).to(self.device)

        # VisDrone Class IDs to Target VISTA mappings:
        # 0: pedestrian -> person (2), 1: people -> person (2)
        # 3: car -> car (0), 4: van -> car (0), 5: truck -> car (0), 8: bus -> car (0)
        self.visdrone_to_vista_id = {
            0: 2,
            1: 2,
            3: 0,
            4: 0,
            5: 0,
            8: 0
        }

    def predict(self, frame: Image.Image):
        # Run inference using Ultralytics
        results = self.model(frame, verbose=False)[0]

        # Convert to Supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Filter classes that belong to the VISTA challenge domain
        valid_indices = [i for i, cls_id in enumerate(detections.class_id) if cls_id in self.visdrone_to_vista_id]

        if not valid_indices:
            return sv.Detections.empty()

        detections = detections[valid_indices]

        # Map VisDrone class IDs directly to VISTA class IDs
        mapped_class_ids = [self.visdrone_to_vista_id[cls_id] for cls_id in detections.class_id]
        detections.class_id = np.array(mapped_class_ids)

        return detections

# ─── 2. LIGHTWEIGHT CLIP "CAPTIONER" (More akin to a classifier) ────────────────────────────────

class CLIPCaptioner:
    def __init__(self, model_id="openai/clip-vit-base-patch16", score_threshold=0.2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_id).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.threshold = score_threshold

        # Pre‑defined attribute lists per category (no redundant category word)
        self.attributes = {
            "car": [
                "parked", "intact", "moving", "crashed", "overturned",
                "damaged", "smoke", "debris"
            ],
            "emergency_vehicle": [
                "ambulance", "police car", "fire truck", "flashing lights", "siren"
            ],
            "person": [
                "standing", "walking", "running", "sitting", "lying down",
                "injured", "helping", "waving", "crouching"
            ]
        }

        # Pre‑compute text embeddings for all labels
        self.text_embeds = {}
        for cat, texts in self.attributes.items():
            self.text_embeds[cat] = self._encode_texts(texts)

    def _extract_features(self, output):
        """Handle both plain tensors and BaseModelOutputWithPooling."""
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        return output   # already a tensor

    def _encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_out = self.model.get_text_features(**inputs)
            emb = self._extract_features(text_out)
        return emb / emb.norm(dim=-1, keepdim=True)

    def classify_batch(self, crops, categories):
        """Return a comma‑separated string of up to 3 attributes per crop."""
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            img_out = self.model.get_image_features(pixel_values=pixel_values)
            image_emb = self._extract_features(img_out)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

        results = []
        for emb, cat in zip(image_emb, categories):
            if cat not in self.text_embeds:
                results.append("intact")
                continue

            text_emb = self.text_embeds[cat]          # (num_texts, dim)
            scores = (emb @ text_emb.T).squeeze(0)    # (num_texts,)

            # Select labels above threshold
            indices = (scores > self.threshold).nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                # fallback: take the highest single label
                best_idx = scores.argmax().item()
                selected = [self.attributes[cat][best_idx]]
            else:
                sorted_indices = indices[(-scores[indices]).argsort()].tolist()
                selected = [self.attributes[cat][i] for i in sorted_indices[:3]]

                # ─── Improved contradiction solver (primary label as anchor) ───
                contradictions = {
                    "intact": ["crashed", "damaged", "overturned", "smoke"],
                    "parked": ["crashed", "overturned", "running", "moving"],
                    "moving": ["parked", "standing", "sitting", "lying down"],
                    "crashed": ["intact", "parked", "moving"],
                    "overturned": ["intact", "parked", "standing", "moving"],
                    "standing": ["sitting", "lying down", "running", "walking", "moving"],
                    "lying down": ["standing", "walking", "running", "moving"],
                    "sitting": ["standing", "running", "walking", "moving"],
                    "walking": ["sitting", "lying down", "standing", "parked"],
                    "running": ["sitting", "lying down", "standing", "parked"],
                    "injured": ["helping"],
                    "helping": ["injured"],
                }

                if len(selected) > 1:
                    primary = selected[0].lower()
                    if primary in contradictions:
                        forbidden = contradictions[primary]
                        selected = [selected[0]] + [
                            s for s in selected[1:] if s.lower() not in forbidden
                        ]

            results.append(", ".join(selected))
        return results


# ─── 3. PRODUCTION OPTIMIZED VISTA PIPELINE ───────────────────────────────────

class VISTASolutionPipeline(VistaPipeline):
    def __init__(self, yolo_model_path="yolo26x_visdrone.pt", caption_stride=30):
        super().__init__()
        # Initialize Detector & Tracker
        self.detector = VisDroneYOLODetector(model_path=yolo_model_path)
        self.tracker = sv.ByteTrack()

        # CLIP‑based captioner (lightweight, controllable)
        self.vlm = CLIPCaptioner(score_threshold=0.2)

        # Caption stride for deep semantic updates
        self.caption_stride = caption_stride

        # Persistent state across frames
        self.track_captions = {}
        self.track_categories = {}

    def reset(self) -> None:
        """Resets tracking and state databases between sequences."""
        self.tracker = sv.ByteTrack()
        self.track_captions.clear()
        self.track_categories.clear()
        print("Pipeline state successfully reset.")

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        # 1. Object Detection
        detections = self.detector.predict(frame)

        if len(detections) == 0:
            return FrameResult(detections=[], frame_idx=frame_idx)

        # 2. Update Tracking
        tracked_detections = self.tracker.update_with_detections(detections)

        id_to_class_name = {0: "car", 1: "emergency_vehicle", 2: "person"}

        # ─── Collect tracks that need caption updates ───
        tracks_to_update = []          # (track_id, crop, base_category)
        track_info = []                # (track_id, bbox, conf, cls_id, base_category)

        for i in range(len(tracked_detections)):
            bbox = tracked_detections.xyxy[i].astype(int)
            conf = float(tracked_detections.confidence[i])
            track_id = int(tracked_detections.tracker_id[i])
            cls_id = int(tracked_detections.class_id[i])
            base_category = id_to_class_name.get(cls_id, "car")

            # Store for final detection construction
            track_info.append((track_id, bbox, conf, cls_id, base_category))

            needs_update = (
                track_id not in self.track_captions or
                frame_idx % self.caption_stride == 0
            )

            if needs_update:
                pad = 15
                crop_box = (
                    max(0, bbox[0] - pad),
                    max(0, bbox[1] - pad),
                    min(frame.width, bbox[2] + pad),
                    min(frame.height, bbox[3] + pad)
                )
                crop = frame.crop(crop_box)
                tracks_to_update.append((track_id, crop, base_category))

        # ─── Batch CLIP classification ───
        if tracks_to_update:
            tids, crops, cats = zip(*tracks_to_update)
            new_captions = self.vlm.classify_batch(list(crops), list(cats))

            emergency_tokens = ["ambulance", "police", "fire truck", "flashing", "siren"]

            for tid, caption, base_cat in zip(tids, new_captions, cats):
                self.track_captions[tid] = caption

                # Use ONLY the first label (before comma) for the emergency upgrade
                primary_label = caption.split(",")[0].strip().lower()

                if base_cat == "car" and any(tok in primary_label for tok in emergency_tokens):
                    self.track_categories[tid] = "emergency_vehicle"
                else:
                    self.track_categories[tid] = base_cat

        # ─── Build final detections from cached state ───
        final_detections = []
        for tid, bbox, conf, cls_id, base_cat in track_info:
            caption = self.track_captions.get(tid, "intact")
            category = self.track_categories.get(tid, base_cat)

            final_detections.append(
                Detection(
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    category=category,
                    confidence=conf,
                    track_id=tid,
                    caption=caption
                )
            )

        return FrameResult(detections=final_detections, frame_idx=frame_idx)
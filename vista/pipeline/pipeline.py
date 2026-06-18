import torch
import numpy as np
from PIL import Image
import supervision as sv
from transformers import CLIPProcessor, CLIPModel

# Import base classes from your competition framework
from MinoTeam_VISTA.vista.pipeline.base import VistaPipeline, FrameResult, Detection


# ─── 1. YOLO DETECTORS & TRACKER WRAPPER ────────────────────────────────

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


class HybridDetector:
    def __init__(self, visdrone_path="yolo26x_visdrone.pt", general_path="yolo11s.pt", min_detections=3):
        self.visdrone = VisDroneYOLODetector(visdrone_path)
        self.general = None
        self.general_path = general_path
        self.min_detections = min_detections

        # COCO -> VISTA mapping for the general model
        # COCO: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
        self.coco_to_vista = {
            0: 2,   # person -> person
            2: 0,   # car -> car
            3: 0,   # motorcycle -> car
            5: 0,   # bus -> car
            7: 0,   # truck -> car
        }

    def _init_general(self):
        if self.general is None:
            from ultralytics import YOLO
            import supervision as sv
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading General YOLO Detector: {self.general_path} on {self.device}...")
            self.general = YOLO(self.general_path).to(self.visdrone.device)

    def predict(self, frame: Image.Image):
        # 1. VisDrone detection
        dets = self.visdrone.predict(frame)

        # 2. If too few detections, run the general model
        if len(dets) < self.min_detections:
            self._init_general()
            # Run general YOLO and convert to supervision Detections
            results = self.general(frame, verbose=False)[0]
            gen_dets = sv.Detections.from_ultralytics(results)

            # Filter to relevant COCO classes and map to VISTA IDs
            valid = [i for i, cls in enumerate(gen_dets.class_id) if cls in self.coco_to_vista]
            if valid:
                gen_dets = gen_dets[valid]
                gen_dets.class_id = np.array([self.coco_to_vista[c] for c in gen_dets.class_id])

                # Merge with VisDrone detections (if any)
                if len(dets) > 0:
                    dets = sv.Detections.merge([dets, gen_dets])
                else:
                    dets = gen_dets

        return dets

# ─── 2. CLIP STATIC CAPTIONER ────────────────────────────────

class CLIPCaptioner:
    def __init__(self, model_id="openai/clip-vit-large-patch14", score_threshold=0.10):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).eval().to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.threshold = score_threshold

        # Pre‑defined attribute lists per category (no redundant category word)
        self.attributes = {
            "car": [
                "parked", "intact", "moving", "crashed", "overturned"
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
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        pixel_values = inputs["pixel_values"]
        with torch.no_grad():
            img_out = self.model.get_image_features(pixel_values=pixel_values)
            image_emb = self._extract_features(img_out)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

        # ---- Explicit thresholds ----
        label_thresholds = {
            # Person actions (optional – only if very confident)
            "injured": 0.15,
            "helping": 0.15,
            "waving": 0.15,
            "crashed": 0.15
        }

        # ---- Conflict map ----
        conflict_map = {
            "intact": {"crashed", "damaged", "overturned"},
            "parked": {"crashed", "overturned", "running", "moving"},
            "moving": {"parked", "standing", "sitting", "lying down", "crashed"},
            "crashed": {"intact", "parked", "moving"},
            "overturned": {"intact", "parked", "standing", "moving"},
            "standing": {"sitting", "lying down", "running", "walking", "crouching", "injured"},
            "lying down": {"standing", "walking", "running", "moving"},
            "sitting": {"standing", "running", "walking", "crouching"},
            "walking": {"sitting", "lying down", "standing", "crouching", "injured", "running"},
            "running": {"sitting", "lying down", "standing", "crouching", "injured", "walking"},
            "crouching": {"standing", "running", "walking", "sitting"},
            "injured": {"helping", "standing", "walking", "running"},
            "helping": {"injured"},
        }

        # ---- Emergency vehicle types (ONLY these three) ----
        emergency_type_labels = {"ambulance", "police car", "fire truck"}
        # Threshold for deciding to reclassify as emergency_vehicle
        EMERGENCY_DETECTION_THRESHOLD = 0.15  # Adjust as needed

        results = []

        for idx, (emb, cat) in enumerate(zip(image_emb, categories)):
            # Safety check
            if cat not in self.text_embeds:
                results.append("intact")
                continue

            original_cat = cat  # Keep for embedding retrieval
            text_emb = self.text_embeds[original_cat]
            scores = (emb @ text_emb.T).squeeze(0)
            all_labels = self.attributes[original_cat]
            cat_lower = original_cat.lower()

            # -------- DETECT EMERGENCY VEHICLE --------
            is_emergency = False
            mandatory_type_idx = None

            # Find the best emergency type label and its score
            emergency_candidates = []
            for i, lbl in enumerate(all_labels):
                if lbl.lower() in emergency_type_labels:
                    emergency_candidates.append((i, scores[i]))

            if emergency_candidates:
                # Sort by score descending
                emergency_candidates.sort(key=lambda x: x[1], reverse=True)
                best_idx, best_score = emergency_candidates[0]
                if best_score > EMERGENCY_DETECTION_THRESHOLD:
                    is_emergency = True
                    mandatory_type_idx = best_idx
                    # ---- Update the object's class in the input list ----
                    categories[idx] = "emergency_vehicle"
                    # We'll keep original_cat for embedding lookups

            # -------- 1. Define tier keywords (based on original category) --------
            if cat_lower in ["person", "people", "human", "pedestrian"]:
                tier1_keys = {"walking", "crouching", "lying down", "running", "standing"}
                tier2_keys = {"helping", "waving", "injured"}
            else:  # All vehicles (including regular and now emergency)
                tier1_keys = {"parked", "moving"}
                tier2_keys = {"crashed", "intact"}

            tier1_indices = [i for i, lbl in enumerate(all_labels) if lbl in tier1_keys]
            tier2_indices = [i for i, lbl in enumerate(all_labels) if lbl in tier2_keys]
            tier3_indices = [
                i for i, lbl in enumerate(all_labels)
                if i not in tier1_indices and i not in tier2_indices
            ]

            # -------- 2. Candidates that pass their thresholds --------
            candidate_indices = []
            for i, label in enumerate(all_labels):
                thresh = label_thresholds.get(label, self.threshold)
                if scores[i] > thresh:
                    candidate_indices.append(i)

            candidate_indices.sort(key=lambda i: scores[i], reverse=True)

            # -------- 3. Fallback --------
            if not candidate_indices:
                best_idx = max(range(len(scores)), key=lambda i: scores[i])
                results.append(all_labels[best_idx])
                continue

            tier1_cands = [i for i in candidate_indices if i in tier1_indices]
            tier2_cands = [i for i in candidate_indices if i in tier2_indices]
            tier3_cands = [i for i in candidate_indices if i in tier3_indices]

            # -------- 4. Greedy selection --------
            kept_indices = []

            # --- Tier 1 ---
            if tier1_cands:
                kept_indices.append(tier1_cands[0])

            # --- Tier 2 ---
            if tier2_cands:
                for i in tier2_cands:
                    label = all_labels[i]
                    conflicts = conflict_map.get(label.lower(), set())
                    if not any(all_labels[k].lower() in conflicts for k in kept_indices):
                        kept_indices.append(i)
                        break

            # --- Tier 3 (multiple labels allowed, up to 3 total) ---
            # If emergency, force-add the mandatory type to the front of Tier 3 candidates
            if is_emergency and mandatory_type_idx is not None:
                if mandatory_type_idx not in kept_indices:
                    # Ensure it's in tier3_cands (if not, add it)
                    if mandatory_type_idx in tier3_cands:
                        tier3_cands.remove(mandatory_type_idx)
                    # Insert at front
                    tier3_cands.insert(0, mandatory_type_idx)

            if tier3_cands and len(kept_indices) < 3:
                for i in tier3_cands:
                    label = all_labels[i]
                    conflicts = conflict_map.get(label.lower(), set())
                    if any(all_labels[k].lower() in conflicts for k in kept_indices):
                        continue

                    # If this is the mandatory emergency type, add unconditionally
                    if is_emergency and i == mandatory_type_idx:
                        kept_indices.append(i)
                    else:
                        # Regular Tier 3 label: only add if it meets its threshold
                        thresh = label_thresholds.get(label, self.threshold)
                        if scores[i] > thresh:
                            kept_indices.append(i)

                    if len(kept_indices) >= 3:
                        break

            # -------- 5. Safety fallback --------
            if not kept_indices:
                kept_indices.append(candidate_indices[0])

            final_labels = [all_labels[i] for i in kept_indices]
            results.append(", ".join(final_labels))

        return results


# ─── 3. PRODUCTION OPTIMIZED VISTA PIPELINE ───────────────────────────────────

class VISTASolutionPipeline(VistaPipeline):
    def __init__(self, yolo_model_path="yolo26x_visdrone.pt", caption_stride=30):
        super().__init__()
        # Initialize Detector & Tracker
        self.detector = HybridDetector(
            visdrone_path=yolo_model_path,
            general_path="yolo11s.pt",
            min_detections=3  # switch to general model when VisDrone finds < 3 objects
        )
        self.tracker = sv.ByteTrack()

        # CLIP‑based captioner
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
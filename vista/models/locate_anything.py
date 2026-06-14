"""Locate Anything wrapper for selective grounding in VISTA pipeline."""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from PIL import Image
import logging
import torch
import time
import re

logger = logging.getLogger(__name__)


class LocateAnythingWrapper:
    """
    Wrapper around NVIDIA Locate Anything model.

    Locate Anything is a grounded visual understanding model that takes an image
    and natural language descriptions and returns bounding boxes for the described objects.

    Interface:
        ground(image, text_prompts, return_multiple=True, conf_threshold=0.3)
            → Dict[str, List[Tuple[bbox_xyxy, confidence]]]

    Example:
        >>> wrapper = LocateAnythingWrapper()
        >>> results = wrapper.ground(
        ...     image=frame,
        ...     text_prompts=["injured person", "ambulance"],
        ...     conf_threshold=0.4
        ... )
    """

    def __init__(self, model_cfg: dict | None = None, device: str = "cuda"):
        """
        Initialize Locate Anything model.

        Args:
            model_cfg: Optional configuration dict with model parameters
            device: Device to load model on ("cuda" or "cpu")
        """
        self.model_cfg = model_cfg or {}
        self.model = None
        self.device = device if torch.cuda.is_available() else "cpu"
        self._is_dummy = False

        # --- PIP INSTALL COMMANDS ---
        # 1. Native / GitHub API:
        #    pip install git+https://github.com/NVlabs/LocateAnything.git
        # 2. Hugging Face Fallback:
        #    pip install transformers accelerate pillow opencv-python bitsandbytes
        #    pip install vllm (for high-throughput inference on Edge GPUs)

        try:
            # Attempt 1: Load using the native NVIDIA package as per hints
            from locate_anything import build_locate_anything
            logger.info("Initializing native Locate Anything model...")
            self.model = build_locate_anything(
                model_name=self.model_cfg.get("model_name", "nvidia/LocateAnything-3B"),
                device=self.device
            )

        except ImportError:
            logger.warning("Native 'locate_anything' not found. Trying HuggingFace fallback.")
            try:
                # Attempt 2: Fallback to Hugging Face pipeline (LocateAnything-3B)
                from transformers import pipeline
                logger.info(f"Loading nvidia/LocateAnything-3B pipeline on {self.device}...")

                class HfLocateAnythingWrapper:
                    def __init__(self, dev):
                        self.pipe = pipeline(
                            "image-text-to-text",
                            model="nvidia/LocateAnything-3B",
                            trust_remote_code=True,
                            device=0 if dev == "cuda" else -1
                        )

                    def predict(self, img, prompt):
                        # LocateAnything-3B uses a specific conversational structure for grounding prompts
                        messages = [{
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": f"Locate all the instances that matches the following description: {prompt}</c>"}
                            ]
                        }]

                        outputs = self.pipe(img, prompt=messages, generate_kwargs={"max_new_tokens": 128})
                        generated_text = outputs[0]["generated_text"]
                        return self._parse_output(generated_text)

                    def _parse_output(self, text):
                        # Extract coordinates from VLM output text (typically [x1, y1, x2, y2])
                        boxes, confs = [], []
                        matches = re.findall(r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]', text)
                        for match in matches:
                            boxes.append([int(m) for m in match])
                            confs.append(1.0)  # Language model fallback outputs lack explicit continuous confidences

                        return {"boxes": torch.tensor(boxes) if boxes else torch.empty((0, 4)),
                                "confidences": torch.tensor(confs) if confs else torch.empty((0,))}

                self.model = HfLocateAnythingWrapper(self.device)

            except Exception as e:
                logger.error(f"HuggingFace fallback failed: {e}")
                logger.warning("Locate Anything model unavailable. Activating YOLO-only dummy mode.")
                self._is_dummy = True

    def ground(
        self,
        image: Image.Image,
        text_prompts: List[str],
        return_multiple: bool = True,
        conf_threshold: float = 0.3,
    ) -> Dict[str, List[Tuple[list, float]]]:
        """
        Ground text prompts in an image using Locate Anything.
        """
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")

        if not text_prompts:
            raise ValueError("text_prompts cannot be empty")

        if conf_threshold < 0 or conf_threshold > 1:
            raise ValueError(f"conf_threshold must be in [0, 1], got {conf_threshold}")

        results = {}
        start_time_total = time.time()

        # Try batch grounding if model supports it (preferred for efficiency)
        if hasattr(self.model, 'batch_ground') and not self._is_dummy:
            try:
                logger.debug(f"Attempting batch grounding for {len(text_prompts)} prompts")
                batch_results = self.model.batch_ground(
                    image=image,
                    text_prompts=text_prompts,
                    return_multiple=return_multiple,
                    conf_threshold=conf_threshold
                )

                total_latency_ms = (time.time() - start_time_total) * 1000
                logger.info(f"Batch grounding completed in {total_latency_ms:.2f} ms")
                return batch_results

            except Exception as e:
                logger.warning(f"Batch grounding failed: {e}. Falling back to loop.")

        # Fallback: loop over prompts sequentially
        for prompt in text_prompts:
            try:
                boxes, confs = self._ground_single(
                    image=image,
                    text_prompt=prompt,
                    conf_threshold=conf_threshold,
                    return_multiple=return_multiple
                )
                results[prompt] = [
                    (box.tolist() if hasattr(box, 'tolist') else box, float(conf))
                    for box, conf in zip(boxes, confs)
                ]
            except Exception as e:
                logger.warning(f"Failed to ground prompt '{prompt}': {e}")
                results[prompt] = []

        total_latency_ms = (time.time() - start_time_total) * 1000
        logger.info(f"Total grounding complete. Processed {len(text_prompts)} prompts in {total_latency_ms:.2f} ms")

        return results

    def _ground_single(
        self,
        image: Image.Image,
        text_prompt: str,
        conf_threshold: float,
        return_multiple: bool
    ) -> Tuple[List, List]:
        """
        Ground a single text prompt in an image.
        """
        if self._is_dummy:
            logger.debug(f"Dummy mode: Skipping grounding for prompt '{text_prompt}'.")
            return [], []

        start_time = time.time()
        filtered_boxes, filtered_confs = [], []

        try:
            # 1. Preprocess: Ensure RGB format for model compatibility
            if image.mode != "RGB":
                image = image.convert("RGB")

            # 2. Inference call
            if hasattr(self.model, 'predict'):
                # Used by our Hugging Face wrapper and generic native APIs
                outputs = self.model.predict(image, text_prompt)
                boxes = outputs.get('boxes', [])
                confs = outputs.get('confidences', [])
            else:
                # Direct method as expected by skeleton hints
                boxes, confs = self.model.ground(image, text_prompt)

            # Convert tensors to lists if necessary
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().tolist()
            if isinstance(confs, torch.Tensor):
                confs = confs.cpu().tolist()

            # 3. Filter by confidence threshold
            for box, conf in zip(boxes, confs):
                if conf >= conf_threshold:
                    filtered_boxes.append(box)
                    filtered_confs.append(conf)

            # 4. Handle return_multiple (return only the highest confidence match if False)
            if not return_multiple and filtered_confs:
                best_idx = filtered_confs.index(max(filtered_confs))
                filtered_boxes = [filtered_boxes[best_idx]]
                filtered_confs = [filtered_confs[best_idx]]

        except Exception as e:
            logger.error(f"Error during single grounding for '{text_prompt}': {e}")

        # 5. Timing and logging per call
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Locate Anything [{text_prompt}]: found {len(filtered_boxes)} objects in {latency_ms:.2f} ms")

        return filtered_boxes, filtered_confs

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set default confidence threshold for all future ground() calls."""
        if threshold < 0 or threshold > 1:
            raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
        self.model_cfg['conf_threshold'] = threshold
        logger.info(f"Confidence threshold set to {threshold}")

    def to(self, device: str) -> LocateAnythingWrapper:
        """Move model to device (cuda or cpu)."""
        if hasattr(self.model, 'to'):
            self.model.to(device)
            self.device = device
            logger.info(f"Model moved to device: {device}")
        return self

    @property
    def is_dummy(self) -> bool:
        """Check if this is a dummy/placeholder model."""
        return getattr(self, '_is_dummy', False)
"""Locate Anything wrapper for selective grounding in VISTA pipeline."""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from PIL import Image
import logging
import torch

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
        >>> for prompt, hits in results.items():
        ...     for bbox, conf in hits:
        ...         print(f"{prompt}: {bbox} @ {conf:.2f}")
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
        
        try:
            # TODO: Implement actual model loading from NVIDIA
            # Example placeholder:
            # from locate_anything import build_locate_anything
            # self.model = build_locate_anything(
            #     model_name=model_cfg.get("model_name", "locate-anything"),
            #     device=self.device
            # )
            
            logger.info(
                f"Locate Anything model initialization placeholder (device: {self.device}). "
                "Install and implement actual loading from: "
                "https://research.nvidia.com/labs/lpr/locate-anything/"
            )
            self._is_dummy = True
            
        except ImportError as e:
            logger.error(f"Failed to import Locate Anything: {e}")
            raise RuntimeError(
                "Locate Anything model not available. "
                "Install via: pip install locate-anything "
                "Or visit: https://research.nvidia.com/labs/lpr/locate-anything/"
            )
        except Exception as e:
            logger.error(f"Failed to load Locate Anything: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")
    
    def ground(
        self,
        image: Image.Image,
        text_prompts: List[str],
        return_multiple: bool = True,
        conf_threshold: float = 0.3,
    ) -> Dict[str, List[Tuple[list, float]]]:
        """
        Ground text prompts in an image using Locate Anything.
        
        Args:
            image: PIL Image (RGB). Will be converted if needed.
            text_prompts: List of text descriptions to ground
                Example: ["injured person sitting", "ambulance", "crashed car"]
            return_multiple: If True, return all matches per prompt.
                            If False, return best match only.
            conf_threshold: Minimum confidence score to include results [0, 1]
        
        Returns:
            Dict mapping prompt string → List of (bbox, confidence) tuples
            
            Each bbox is [x1, y1, x2, y2] in pixel coordinates (top-left, bottom-right)
            Confidence is a float in [0, 1]
            
            Example:
                {
                    "injured person": [[10, 20, 100, 200], 0.95], [[300, 150, 450, 350], 0.87],
                    "ambulance": [[400, 50, 550, 150], 0.92],
                }
        
        Raises:
            ValueError: If image is invalid or prompts is empty
            RuntimeError: If model inference fails
        """
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")
        
        if not text_prompts:
            raise ValueError("text_prompts cannot be empty")
        
        if conf_threshold < 0 or conf_threshold > 1:
            raise ValueError(f"conf_threshold must be in [0, 1], got {conf_threshold}")
        
        results = {}
        
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
        
        logger.debug(f"Grounding complete. Results per prompt: {[(p, len(r)) for p, r in results.items()]}")
        
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
        
        Args:
            image: PIL Image (RGB)
            text_prompt: Single text description
            conf_threshold: Minimum confidence threshold
            return_multiple: Return all matches or best match only
        
        Returns:
            Tuple of (boxes, confidences)
            boxes: List of [x1, y1, x2, y2] in pixel coordinates
            confidences: List of confidence scores [0, 1]
        
        Note:
            This is a placeholder. Implement actual Locate Anything inference here:
            
            Example (pseudocode):
            >>> image_tensor = self._preprocess(image)
            >>> outputs = self.model.predict(image_tensor, text_prompt)
            >>> boxes = outputs['boxes']  # shape (N, 4)
            >>> confs = outputs['confidences']  # shape (N,)
            >>> mask = confs >= conf_threshold
            >>> if not return_multiple:
            ...     best_idx = confs.argmax()
            ...     boxes = [boxes[best_idx]]
            ...     confs = [confs[best_idx]]
            >>> return boxes[mask], confs[mask]
        """
        
        # TODO: Implement actual Locate Anything inference
        # Placeholder returns empty
        
        logger.debug(f"_ground_single called with prompt: '{text_prompt}' (placeholder impl)")
        
        return [], []
    
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

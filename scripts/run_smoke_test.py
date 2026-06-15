"""
Smoke test for the VISTA basic YOLO detection pipeline.
Scope: 20 frames, verify FrameResult outputs, ensure zero errors.
"""

import logging
import sys
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Importing from the path specified in your scope
from vista.pipeline.project_pipeline import LightweightPipelineLocate

# Configure logging to ensure detections are visible in stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def run_smoke_test():
    logger.info("Initializing 20-frame smoke test for LightweightPipelineLocate...")

    # 1. Initialize a fast, small YOLO model (Ultralytics backend)
    try:
        # yolo11n.pt is a lightweight model suitable for quick tests
        yolo_model = YOLO("yolo11n.pt")
        logger.info(f"Loaded YOLO model: {yolo_model.model_name}")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        sys.exit(1)

    # 2. Instantiate pipeline matching the Basic YOLO scope (Locate = None)
    pipeline = LightweightPipelineLocate(
        yolo_model=yolo_model,
        locate_model=None,  # Disabled for basic stage testing
        enable_profiling=True
    )

    pipeline.reset()

    # 3. Test Execution: Process 20 dummy frames
    total_frames = 20
    successful_frames = 0

    for frame_idx in range(total_frames):
        try:
            # Generate a 640x640 blank RGB frame
            dummy_array = np.zeros((640, 640, 3), dtype=np.uint8)
            frame = Image.fromarray(dummy_array)

            # Forward pass
            result = pipeline.forward(frame, frame_idx)

            # Acceptance Criteria Validations
            assert result is not None, "Pipeline returned None instead of FrameResult."
            assert hasattr(result, 'detections'), "Result is missing 'detections' attribute."
            assert result.frame_idx == frame_idx, "Frame index mismatch in FrameResult."

            logger.info(f"Frame {frame_idx:02d} processed successfully | Detections: {len(result.detections)}")
            successful_frames += 1

        except Exception as e:
            logger.error(f"Pipeline crashed on frame {frame_idx}: {e}")
            break

    # 4. Final Validation & Teardown Summary
    if successful_frames == total_frames:
        logger.info("Smoke test passed: All 20 frames processed without errors.")

        # Manually log profiling stats since the % 30 log in forward() won't trigger for 20 frames
        if pipeline.enable_profiling and pipeline._profiling_stats["total_time"]:
            avg_time = np.mean(pipeline._profiling_stats["total_time"])
            logger.info(f"⏱️ Average Execution Time: {avg_time * 1000:.2f}ms per frame (~{1.0 / avg_time:.1f} FPS)")
    else:
        logger.error(f"Smoke test failed: Only {successful_frames}/{total_frames} frames completed.")
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
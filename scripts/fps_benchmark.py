"""
FPS Benchmark and MOT Logging Script.
Scope: Measure pipeline FPS and export tracking results to predictions_mot.csv.
"""

import logging
import time
import os
import csv
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# Adjust import based on your actual path
from vista.pipeline.project_pipeline import LightweightPipelineLocate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def generate_mock_frame(frame_idx: int) -> Image.Image:
    """Generate a mock frame with a moving box to simulate trackable motion."""
    frame_bgr = np.zeros((640, 640, 3), dtype=np.uint8)

    # Create a white moving square to stimulate the detector/tracker
    box_size = 100
    x_pos = (frame_idx * 5) % (640 - box_size)
    y_pos = (frame_idx * 3) % (640 - box_size)

    cv2.rectangle(
        frame_bgr,
        (x_pos, y_pos),
        (x_pos + box_size, y_pos + box_size),
        (255, 255, 255),
        -1
    )

    # Convert BGR back to RGB for PIL as expected by the pipeline
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def run_benchmark(video_path: str = None, num_frames: int = 100):
    logger.info("Initializing Tracker Benchmark...")

    yolo_model = YOLO("yolo11n.pt")
    pipeline = LightweightPipelineLocate(yolo_model=yolo_model, enable_profiling=True)
    pipeline.reset()

    mot_log_path = "predictions_mot.csv"

    # 1. Open the video or prepare mock generation
    cap = None
    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        logger.info(f"Using video file: {video_path}")
    else:
        logger.warning("No valid video provided. Using synthetic moving frames.")

    # 2. Open CSV for MOT logging
    with open(mot_log_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Standard MOT format header (optional, but good for readability)
        writer.writerow(["frame_id", "track_id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"])

        start_time = time.time()

        for frame_idx in range(1, num_frames + 1):
            # Fetch frame
            if cap:
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame_rgb)
            else:
                frame = generate_mock_frame(frame_idx)

            # Process frame through the tracking pipeline
            result = pipeline.forward(frame, frame_idx)

            # 3. Log to MOT Format
            for det in result.detections:
                if det.track_id is not None:
                    # Convert xyxy (bbox) to MOT format: bb_left, bb_top, bb_width, bb_height
                    x1, y1, x2, y2 = det.bbox
                    bb_width = x2 - x1
                    bb_height = y2 - y1

                    writer.writerow([
                        frame_idx,             # 1-based frame index
                        det.track_id,          # Persistent ID
                        f"{x1:.2f}",           # bb_left
                        f"{y1:.2f}",           # bb_top
                        f"{bb_width:.2f}",     # bb_width
                        f"{bb_height:.2f}",    # bb_height
                        f"{det.confidence:.4f}", # conf
                        "-1", "-1", "-1"       # x, y, z (ignored in 2D MOT)
                    ])

    # Cleanup
    if cap:
        cap.release()

    # 4. Calculate and Document FPS Acceptance
    total_duration = time.time() - start_time
    actual_frames = frame_idx
    avg_fps = actual_frames / total_duration

    logger.info(f"Benchmark Complete. Processed {actual_frames} frames.")
    logger.info(f"Results saved to: {mot_log_path}")

    # Evaluation against target threshold
    if avg_fps >= 5.0:
        logger.info(f"ACCEPTANCE MET: Average FPS is {avg_fps:.2f} (Target: ≥5 FPS).")
    else:
        logger.warning(f"ACCEPTANCE FAILED: Average FPS is {avg_fps:.2f} (Target: ≥5 FPS).")
        logger.info("Plan to improve: Use smaller YOLO variant (e.g., YOLO11n), scale down input resolution, or enable TensorRT/Half-precision (FP16) inference.")

if __name__ == "__main__":
    # You can pass a real video path here, e.g., run_benchmark("data/sequences/accident.mp4", 100)
    run_benchmark(num_frames=100)
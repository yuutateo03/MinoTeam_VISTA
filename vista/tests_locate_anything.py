import pytest
import time
import numpy as np
from PIL import Image
from unittest.mock import MagicMock

from pipeline.base import Detection
from pipeline.lightweight_pipeline_locate import LightweightPipelineLocate


def create_dummy_frame(width=640, height=480):
    """Creates a blank PIL Image."""
    array = np.zeros((height, width, 3), dtype=np.uint8)
    return Image.fromarray(array)


class MockYOLO:
    """Mocks the Ultralytics YOLO interface."""

    def __init__(self):
        self.model_name = "mock_yolo"
        self.predictor = MagicMock()

    def track(self, frame_bgr, conf, persist, tracker, verbose):
        # Simulate ~5ms processing time
        time.sleep(0.005)

        mock_result = MagicMock()
        # Fake one detection
        mock_result.boxes.xyxy.cpu().numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_result.boxes.conf.cpu().numpy.return_value = np.array([0.9])
        mock_result.boxes.cls.cpu().numpy.return_value = np.array([0])
        mock_result.boxes.id.cpu().numpy.return_value = np.array([1])
        mock_result.names = {0: "car", 1: "person"}
        return [mock_result]


class MockLocateAnything:
    """Mocks the Locate Anything model."""

    def __init__(self):
        self.call_count = 0

    def ground(self, image, text_prompts, conf_threshold):
        self.call_count += 1
        # Simulate ~50-80ms processing time
        time.sleep(0.065)

        # Return a fake box that heavily overlaps with the YOLO box (will trigger NMS merge)
        return {
            "injured person": [([12, 12, 48, 48], 0.85)]
        }


def test_lightweight_pipeline_profiling():
    """
    Test a 20-frame run measuring FPS and ensuring Locate Anything
    is called selectively based on locate_every_n.
    """
    mock_yolo = MockYOLO()
    mock_locate = MockLocateAnything()

    # Initialize pipeline expecting a trigger every 3 frames
    pipeline = LightweightPipelineLocate(
        yolo_model=mock_yolo,
        locate_model=mock_locate,
        locate_prompts=["injured person"],
        locate_every_n=3,
        trigger_classes=["crashed_car"],
        # Our mock YOLO outputs "car", so this won't trigger; only N-frames will trigger
        enable_profiling=True
    )

    pipeline.reset()

    total_frames = 20
    frames = [create_dummy_frame() for _ in range(total_frames)]

    start_time = time.time()
    for idx, frame in enumerate(frames):
        result = pipeline.forward(frame, frame_idx=idx)

        # Check that YOLO tracked the object and Locate caption was fused correctly on trigger frames
        assert len(result.detections) == 1
        det = result.detections[0]
        assert det.category == "car"

        if idx % 3 == 0:
            assert det.caption == "injured person"  # Fused successfully
        else:
            assert det.caption is None

    test_duration = time.time() - start_time
    average_fps = total_frames / test_duration

    # Validation constraints
    expected_calls = 7  # Frames 0, 3, 6, 9, 12, 15, 18
    assert mock_locate.call_count == expected_calls, f"Expected {expected_calls} calls, got {mock_locate.call_count}"

    # Stats breakdown
    avg_yolo = np.mean(pipeline._profiling_stats["yolo_time"]) * 1000
    avg_locate = (np.sum(pipeline._profiling_stats["locate_time"]) / expected_calls) * 1000

    print("\n--- Pipeline Profiling Results (20 frames) ---")
    print(f"Total Pipeline FPS: {average_fps:.2f}")
    print(f"Average YOLO + Track stage: {avg_yolo:.2f} ms")
    print(f"Average Locate Anything stage: {avg_locate:.2f} ms")

    # Verify performance threshold (>= 5 FPS)
    assert average_fps >= 5.0, f"Pipeline failed performance target. FPS: {average_fps:.2f}"
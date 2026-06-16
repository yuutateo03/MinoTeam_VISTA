import os
import csv
import yaml
import pytest
import numpy as np
from typing import List, Dict, Tuple

CONFIG_PATH = "config/detection_tuning.yaml"


class PipelinePerformanceValidator:
    """Validates CSV data schemas and performs ablation analysis comparing pipeline modes."""

    def __init__(self, config_path: str = CONFIG_PATH):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.mot_headers = self.config["evaluation"]["schemas"]["predictions_mot"]
        self.tracks_headers = self.config["evaluation"]["schemas"]["predictions_tracks"]
        self.targets = self.config["evaluation"]["targets"]

    def validate_tracking_csvs(self, mot_path: str, tracks_path: str) -> Tuple[bool, str]:
        """Validates structural integrity and column mappings of pipeline outputs."""
        for path, expected_headers in [(mot_path, self.mot_headers), (tracks_path, self.tracks_headers)]:
            if not os.path.exists(path):
                return False, f"Missing target file: {path}"

            with open(path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                actual_headers = next(reader, [])
                if actual_headers != expected_headers:
                    return False, f"Schema mismatch in {path}.\nExpected: {expected_headers}\nGot: {actual_headers}"

        return True, "All tracking CSV structures match definitions perfectly."

    def evaluate_ablation_metrics(self, tuning_summary_path: str) -> Dict:
        """Parses a tuning report to compute metrics delta between YOLO-only and Hybrid runs."""
        if not os.path.exists(tuning_summary_path):
            raise FileNotFoundError(f"Tuning summary metrics file not found: {tuning_summary_path}")

        yolo_runs = []
        hybrid_runs = []

        with open(tuning_summary_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = {
                    "mode": row["mode"].strip().lower(),
                    "f1_score": float(row["f1_score"]),
                    "fps": float(row["fps"]),
                    "latency_ms": float(row["latency_ms"])
                }
                if record["mode"] == "yolo_only":
                    yolo_runs.append(record)
                elif record["mode"] == "hybrid":
                    hybrid_runs.append(record)

        # Compute aggregate averages
        def aggregate(runs: List[Dict]) -> Dict:
            if not runs:
                return {"f1": 0.0, "fps": 0.0, "latency": 0.0, "count": 0}
            return {
                "f1": np.mean([r["f1_score"] for r in runs]),
                "fps": np.mean([r["fps"] for r in runs]),
                "latency": np.mean([r["latency_ms"] for r in runs]),
                "count": len(runs)
            }

        yolo_summary = aggregate(yolo_runs)
        hybrid_summary = aggregate(hybrid_runs)

        return {
            "yolo_only": yolo_summary,
            "hybrid": hybrid_summary,
            "comparison": {
                "f1_improvement": hybrid_summary["f1"] - yolo_summary["f1"],
                "fps_cost": yolo_summary["fps"] - hybrid_summary["fps"]
            }
        }


# --- Pytest Unit Tests & Mock Generation ---

@pytest.fixture
def setup_mock_data():
    """Generates synthetic tracker records and tuning summaries matching project schemas."""
    os.makedirs("out/test_tuning", exist_ok=True)
    mot_path = "out/test_tuning/predictions_mot.csv"
    tracks_path = "out/test_tuning/predictions_tracks.csv"
    summary_path = "out/test_tuning/tuning_summary.csv"

    # 1. Write mock predictions_mot.csv
    with open(mot_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "frame_id", "track_id", "x1", "y1", "x2", "y2", "conf", "category"])
        writer.writerow(["vid_001", 0, 1, 120.5, 200.0, 310.2, 450.1, 0.88, "person"])
        writer.writerow(["vid_001", 0, 2, 400.0, 150.5, 600.1, 390.4, 0.92, "car"])

    # 2. Write mock predictions_tracks.csv
    with open(tracks_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "track_id", "frame_start", "frame_end", "caption"])
        writer.writerow(["vid_001", 1, 0, 45, "person_injured"])
        writer.writerow(["vid_001", 2, 0, 120, "crashed_car"])

    # 3. Write mock tuning summary sheet mapping grid execution results
    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "mode", "f1_score", "fps", "latency_ms"])
        # YOLO-only runs: Ultra-fast but lower semantic F1 context resolution
        writer.writerow([1, "yolo_only", 0.68, 145.2, 6.8])
        writer.writerow([2, "yolo_only", 0.71, 138.9, 7.2])
        # Hybrid configurations: Slower but contextual VLM grounding boosts F1
        writer.writerow([3, "hybrid", 0.84, 6.4, 156.2])
        writer.writerow([4, "hybrid", 0.81, 8.1, 123.4])

    yield mot_path, tracks_path, summary_path

    # Cleanup test outputs cleanly
    for p in [mot_path, tracks_path, summary_path]:
        if os.path.exists(p):
            os.remove(p)


def test_pipeline_csv_schema_verification(setup_mock_data):
    """Guarantees tracking output schemas precisely match standard VISTA specifications."""
    mot_path, tracks_path, _ = setup_mock_data
    validator = PipelinePerformanceValidator()

    success, msg = validator.validate_tracking_csvs(mot_path, tracks_path)
    assert success is True, msg


def test_yolo_vs_hybrid_ablation(setup_mock_data):
    """Validates performance metrics and asserts that Hybrid configurations achieve SLA targets."""
    _, _, summary_path = setup_mock_data
    validator = PipelinePerformanceValidator()

    metrics = validator.evaluate_ablation_metrics(summary_path)

    # Render operational performance delta report
    print("\n\n" + "=" * 45)
    print("      VISTA DETECTOR TUNING MATRIX REPORT      ")
    print("=" * 45)
    print(
        f"YOLO-only Base -> Avg F1: {metrics['yolo_only']['f1']:.3f} | Avg Speed: {metrics['yolo_only']['fps']:.1f} FPS")
    print(f"Hybrid Mode    -> Avg F1: {metrics['hybrid']['f1']:.3f} | Avg Speed: {metrics['hybrid']['fps']:.1f} FPS")
    print("-" * 45)
    print(f"Grounding Delta -> F1 Increase:  {metrics['comparison']['f1_improvement']:+.3f}")
    print(f"Efficiency Cost -> Latency Cost: {metrics['comparison']['fps_cost']:.1f} FPS Drop")
    print("=" * 45)

    # Core target checks
    assert metrics["hybrid"]["fps"] >= validator.targets["min_hybrid_fps"], \
        f"Hybrid execution speed dropped below the real-time SLA threshold of {validator.targets['min_hybrid_fps']} FPS!"

    assert metrics["hybrid"]["f1"] > metrics["yolo_only"]["f1"], \
        "Tuning Failure: Grounded Hybrid VLM components failed to add accuracy value over standard YOLO tracker."
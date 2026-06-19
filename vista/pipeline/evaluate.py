import os
import cv2
import time
import pandas as pd
from PIL import Image
from collections import defaultdict
from glob import glob

# Import your pipeline
from pipeline import VISTASolutionPipeline


class Profiler:
    """Helper class to track execution times without modifying pipeline.py"""

    def __init__(self):
        self.shallow_time = 0.0
        self.deep_time = 0.0
        self.deep_calls = 0
        self.total_frames = 0

    def reset(self):
        self.shallow_time = 0.0
        self.deep_time = 0.0
        self.deep_calls = 0
        self.total_frames = 0


def process_and_evaluate_video(video_path: str, output_dir: str, pipeline: VISTASolutionPipeline, profiler: Profiler):
    """Processes a single video, tracks FPS metrics, and saves results."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create specific output directory for this video
    video_out_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps_native = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_video_path = os.path.join(video_out_dir, f"{video_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_native, (width, height))

    mot_records = []
    track_history = defaultdict(lambda: {"start": None, "end": None, "captions": []})

    # Reset state
    pipeline.reset()
    profiler.reset()
    profiler.total_frames = total_frames

    print(f"\nProcessing: {video_name} | Frames: {total_frames} | Res: {width}x{height}")
    print("-" * 65)

    global_start_time = time.time()
    window_start_time = time.time()

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # Pipeline inference (Timers intercept automatically!)
        result = pipeline.forward(pil_frame, frame_idx=frame_idx)

        # Record and draw detections
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox

            # MOT Record
            mot_records.append({
                "video_id": video_name,
                "frame_id": int(frame_idx),
                "track_id": int(det.track_id),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "conf": float(det.confidence),
                "category": str(det.category)
            })

            # Track History
            if track_history[det.track_id]["start"] is None:
                track_history[det.track_id]["start"] = frame_idx
            track_history[det.track_id]["end"] = frame_idx
            track_history[det.track_id]["captions"].append(det.caption)

            # Draw Annotations
            color = (0, 165, 255) if det.category == "car" else \
                (0, 0, 255) if det.category == "emergency_vehicle" else (255, 0, 0)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID:{det.track_id} {det.category} | {det.caption}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - th - 10), (int(x1) + tw, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Console logging every 50 frames
        if frame_idx > 0 and frame_idx % 50 == 0:
            elapsed_window = time.time() - window_start_time
            current_fps = 50 / elapsed_window

            avg_fps_sofar = (frame_idx + 1) / (time.time() - global_start_time)
            rem_frames = total_frames - (frame_idx + 1)
            eta_sec = int(rem_frames / avg_fps_sofar) if avg_fps_sofar > 0 else 0

            print(
                f"Progress: [{frame_idx}/{total_frames}] | Window FPS: {current_fps:.1f} | ETA: {eta_sec // 60}m {eta_sec % 60}s")
            window_start_time = time.time()

        writer.write(frame)

    cap.release()
    writer.release()

    # ---- Calculate FPS Metrics ----
    global_time = time.time() - global_start_time
    e2e_fps = total_frames / global_time

    shallow_fps = total_frames / profiler.shallow_time if profiler.shallow_time > 0 else 0
    # Deep FPS accounts for stride. If deep logic takes 1s, and stride is 30, it processes 1 frame's worth of deep data per second.
    deep_fps = profiler.deep_calls / profiler.deep_time if profiler.deep_time > 0 else 0

    print("-" * 65)
    print(f"   Video Complete: {video_name}")
    print(f"   End-to-End FPS : {e2e_fps:.2f} FPS")
    print(f"   Shallow FPS    : {shallow_fps:.2f} FPS (YOLO + Tracker)")
    print(f"   Deep FPS       : {deep_fps:.2f} Batches/sec (CLIP Captioning)")

    # ---- Aggregate Track Captions ----
    track_records = []
    for tid, data in track_history.items():
        valid_caps = [c for c in data["captions"] if c not in ["unknown"]]
        final_caption = max(set(valid_caps), key=valid_caps.count) if valid_caps else "intact"
        track_records.append({
            "video_id": video_name, "track_id": int(tid),
            "frame_start": int(data["start"]), "frame_end": int(data["end"]),
            "caption": str(final_caption)
        })

    # Save CSVs inside the specific video folder
    pd.DataFrame(mot_records).to_csv(os.path.join(video_out_dir, "predictions_mot.csv"), index=False)
    pd.DataFrame(track_records).to_csv(os.path.join(video_out_dir, "predictions_tracks.csv"), index=False)


def run_test_set_evaluation():
    TEST_DIR = r"C:\Users\yurit\GitHub\cs_1st_year\2nd Semester\CV\Project\MinoTeam_VISTA\data\VISTADataset\test"
    OUTPUT_DIR = r"C:\Users\yurit\GitHub\cs_1st_year\2nd Semester\CV\Project\MinoTeam_VISTA\vista\pipeline\out"

    print("Initializing VISTA Pipeline for Test Set Evaluation...")
    pipeline = VISTASolutionPipeline(yolo_model_path="yolo26x_visdrone.pt", caption_stride=30)
    profiler = Profiler()

    # ─── METHOD HOOKING (Intercepts pipeline calls to track exact timings) ───
    orig_detect = pipeline.detector.predict
    orig_track = pipeline.tracker.update_with_detections
    orig_classify = pipeline.vlm.classify_batch

    def timed_detect(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_detect(*args, **kwargs)
        profiler.shallow_time += time.perf_counter() - t0
        return res

    def timed_track(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_track(*args, **kwargs)
        profiler.shallow_time += time.perf_counter() - t0
        return res

    def timed_classify(*args, **kwargs):
        t0 = time.perf_counter()
        res = orig_classify(*args, **kwargs)
        profiler.deep_time += time.perf_counter() - t0
        profiler.deep_calls += 1
        return res

    # Inject the hooked methods back into the pipeline
    pipeline.detector.predict = timed_detect
    pipeline.tracker.update_with_detections = timed_track
    pipeline.vlm.classify_batch = timed_classify

    # ─── DISCOVER VIDEOS ───
    # Searches recursively through the date folders for .MP4 or .mp4 files
    search_pattern = os.path.join(TEST_DIR, "**", "*.MP4")
    video_paths = glob(search_pattern, recursive=True)

    # Fallback to lowercase .mp4 if the extension casing varies
    video_paths += glob(os.path.join(TEST_DIR, "**", "*.mp4"), recursive=True)
    video_paths = list(set(video_paths))  # Remove duplicates if any

    if not video_paths:
        print(f"No videos found in {TEST_DIR}")
        return

    print(f"Found {len(video_paths)} videos to process.")

    # Process each video sequentially
    for vid_path in video_paths:
        process_and_evaluate_video(vid_path, OUTPUT_DIR, pipeline, profiler)

    print("\nEvaluation pipeline completed successfully!")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_test_set_evaluation()
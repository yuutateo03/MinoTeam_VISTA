import os
import cv2
import time
import pandas as pd
from PIL import Image
from collections import defaultdict

# Import the pipeline from your file (assuming you named it pipeline.py)
from pipeline import VISTASolutionPipeline


def process_video(video_path: str, output_dir: str, pipeline: VISTASolutionPipeline):
    """
    Processes a single video, generates annotations, and computes track/MOT data.
    Prints execution telemetry to the console every 50 frames.
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    fps_native = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare video writer
    out_video_path = os.path.join(output_dir, f"{video_id}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_native, (width, height))

    # Data collectors
    mot_records = []
    track_history = defaultdict(lambda: {"start": None, "end": None, "captions": []})

    # Reset pipeline tracker and VLM states for the new video sequence
    pipeline.reset()

    global_start_time = time.time()
    window_start_time = time.time()

    print(f"Processing: {video_id} | Total Frames: {total_frames} | Resolution: {width}x{height}")
    print("-" * 60)

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (PIL) for the pipeline
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb_frame)

        # ─── PIPELINE INFERENCE ───
        result = pipeline.forward(pil_frame, frame_idx=frame_idx)

        # Parse results and draw annotations
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox

            # 1. Append MOT Record
            mot_records.append({
                "video_id": str(video_id),
                "frame_id": int(frame_idx),
                "track_id": int(det.track_id),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": float(det.confidence),
                "category": str(det.category)
            })

            # 2. Update Track History (for aggregation)
            if track_history[det.track_id]["start"] is None:
                track_history[det.track_id]["start"] = frame_idx
            track_history[det.track_id]["end"] = frame_idx
            track_history[det.track_id]["captions"].append(det.caption)

            # 3. Draw Annotations on Frame
            color = (0, 165, 255) if det.category == "car" else \
                (0, 0, 255) if det.category == "emergency_vehicle" else \
                    (255, 0, 0)  # Person

            # Draw Bbox & Label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID:{det.track_id} {det.category} | {det.caption}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (int(x1), int(y1) - text_height - 10), (int(x1) + text_width, int(y1)), color, -1)
            cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ─── CONSOLE LOGGING ───
        # Print to console every 50 frames
        if frame_idx > 0 and frame_idx % 50 == 0:
            elapsed_window = time.time() - window_start_time
            current_fps = 50 / elapsed_window

            # ETA calculation from overall progress
            elapsed_total = time.time() - global_start_time
            avg_fps_sofar = (frame_idx + 1) / elapsed_total
            remaining_frames = total_frames - (frame_idx + 1)
            if avg_fps_sofar > 0:
                eta_seconds = remaining_frames / avg_fps_sofar
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                eta_str = f"{eta_min}m {eta_sec}s"
            else:
                eta_str = "calculating..."

            print(f"Progress: [{frame_idx}/{total_frames}] frames | "
                  f"Window FPS: {current_fps:.2f} | "
                  f"Avg FPS: {avg_fps_sofar:.2f} | "
                  f"ETA: {eta_str}")

            window_start_time = time.time()  # Reset window timer

        # Write clean annotated frame to video
        writer.write(frame)

    cap.release()
    writer.release()

    global_end_time = time.time()
    avg_fps = total_frames / (global_end_time - global_start_time)
    print("-" * 60)
    print(f"Video '{video_id}' complete - Average End-to-End FPS: {avg_fps:.2f}\n")

    # ─── AGGREGATE TRACKS CAPTIONS (Majority Vote) ───
    track_records = []
    for tid, data in track_history.items():
        valid_caps = [c for c in data["captions"] if c not in ["unknown"]]
        final_caption = max(set(valid_caps), key=valid_caps.count) if valid_caps else "intact"

        track_records.append({
            "video_id": str(video_id),
            "track_id": int(tid),
            "frame_start": int(data["start"]),
            "frame_end": int(data["end"]),
            "caption": str(final_caption)
        })

    return pd.DataFrame(mot_records), pd.DataFrame(track_records)


def run_evaluation(input_source: str, output_dir: str):
    """
    Runs the pipeline on a single video or directory of videos, merging final CSVs.
    """
    print("Initializing VISTA Pipeline...")
    pipeline = VISTASolutionPipeline(
        yolo_model_path="yolo26x_visdrone.pt",
        caption_stride=30
    )

    all_mot_dfs = []
    all_track_dfs = []

    if os.path.isdir(input_source):
        videos = [os.path.join(input_source, f) for f in os.listdir(input_source) if
                  f.endswith(('.mp4', '.avi', '.mov'))]
    else:
        videos = [input_source]

    if not videos:
        print("No videos found to process!")
        return

    for vid_path in videos:
        mot_df, track_df = process_video(vid_path, output_dir, pipeline)
        all_mot_dfs.append(mot_df)
        all_track_dfs.append(track_df)

    print("Merging submission files...")
    final_mot = pd.concat(all_mot_dfs, ignore_index=True)
    final_tracks = pd.concat(all_track_dfs, ignore_index=True)

    mot_path = os.path.join(output_dir, "predictions_mot.csv")
    tracks_path = os.path.join(output_dir, "predictions_tracks.csv")

    final_mot.to_csv(mot_path, index=False)
    final_tracks.to_csv(tracks_path, index=False)

    print(f"🎉 Evaluation finished! Submissions saved to:")
    print(f" - {mot_path}")
    print(f" - {tracks_path}")


if __name__ == "__main__":
    # Path to a single video OR a folder containing multiple videos
    INPUT_PATH = r"C:\Users\yurit\GitHub\cs_1st_year\2nd Semester\CV\Project\MinoTeam_VISTA\data\VISTADataset\test\20251210\DJI_20251210134636_0001_S.mp4"
    OUTPUT_DIR = r"C:\Users\yurit\GitHub\cs_1st_year\2nd Semester\CV\Project\MinoTeam_VISTA\vista\pipeline\out"

    run_evaluation(INPUT_PATH, OUTPUT_DIR)
import logging
import time
import sys
from pathlib import Path
from ultralytics import YOLO

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from vista.pipeline.lightweight_pipeline_locate import LightweightPipelineLocate
from scripts.write_csv_skeletons import create_csv_skeletons, CSVWriter

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_csv():
    # 1. Define paths
    video_path = r"C:\Users\franc\Documents\GitHub\MinoTeam_VISTA\data\VISTADataset\test\20251210\DJI_20251210134636_0001_S.mp4"
    out_dir = r"C:\Users\franc\Documents\GitHub\MinoTeam_VISTA\out\run_002"
    video_id = Path(video_path).stem  # Estrae il nome: "DJI_20251210134636_0001_S"


    if not Path(video_path).exists():
        logger.error(f"ERROR: Video not found at {Path(video_path).resolve()}")
        return

    # 2. Initialize CSV structures
    logger.info(f"Initializing CSV files in directory: {out_dir}...")
    create_csv_skeletons(out_dir)
    
    writer_mot = CSVWriter(f"{out_dir}/predictions_mot.csv")

    # 3. Initialize YOLO and the tracking pipeline
    logger.info("Starting YOLO model and ByteTrack tracker...")
    pipeline = LightweightPipelineLocate(
        yolo_model=YOLO("yolov8n.pt"), 
        enable_profiling=False
    )

    logger.info("Processing video frame-by-frame and writing tracking data to CSV...")
    
    # Start the timer for FPS calculation
    start_time = time.time()
    
    writer_tracks = CSVWriter(f"{out_dir}/predictions_tracks.csv")

    # 4. Run the pipeline and save real tracking data
    for result in pipeline.process_video(video_path):
        frame_id = result.frame_idx
        
        for det in result.detections:
            # Save only high-confidence detections
            if det.confidence > 0.3:
                
                # Use ByteTrack's persistent ID, fallback to -1 if None
                track_id = det.track_id if det.track_id is not None else -1
                
                writer_mot.append_row(
                    video_id=video_id,
                    frame_id=frame_id,
                    track_id=track_id,
                    x1=det.bbox[0],
                    y1=det.bbox[1],
                    x2=det.bbox[2],
                    y2=det.bbox[3],
                    conf=det.confidence,
                    category=det.category
                )
            
        # Log progress and FPS every 30 frames
        if frame_id > 0 and frame_id % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = frame_id / elapsed_time
            logger.info(f"Processed {frame_id} frames in {elapsed_time:.2f}s (Processing Speed: {fps:.1f} FPS)")

    # New: 5. Save Final Track Captions (Captioning Finale)
    logger.info("Generating Final Track Captions and saving to predictions_tracks.csv...")
    for track_id, history in pipeline.track_history.items():
        writer_tracks.append_row(
            video_id=video_id,
            track_id=track_id,
            frame_start=history["frame_start"],
            frame_end=history["frame_end"],
            caption=history["final_caption"] if history["final_caption"] else "normal"
        )

    # Final Summary Log
    total_time = time.time() - start_time
    final_fps = frame_id / total_time if total_time > 0 else 0
    logger.info(f"COMPLETED! Processed a total of {frame_id} frames. Final Average Speed: {final_fps:.1f} FPS.")
    logger.info(f"Real tracking data has been successfully saved to: {out_dir}/predictions_mot.csv")

if __name__ == "__main__":
    generate_csv()
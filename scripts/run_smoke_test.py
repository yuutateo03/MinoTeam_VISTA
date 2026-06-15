import logging
from ultralytics import YOLO

# Assicurati che i path di import corrispondano alla struttura del tuo progetto
from vista.pipeline.lightweight_pipeline_locate import LightweightPipelineLocate

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_smoke_test():
    video_path = r"C:\Users\franc\Documents\GitHub\MinoTeam_VISTA\data\VISTADataset\test"

    logger.info("Caricamento modello YOLOv8 Nano...")
    try:
        yolo_model = YOLO("yolov8n.pt")
    except Exception as e:
        logger.error(f"Errore YOLO: {e}")
        return

    # Inizializza la pipeline
    pipeline = LightweightPipelineLocate(
        yolo_model=yolo_model,
        enable_profiling=True
    )

    logger.info(f"Avvio elaborazione usando il metodo integrato process_video() su: {video_path}")
    
    # Usiamo il metodo svelato dalla documentazione! Fa tutto lui.
    try:
        for result in pipeline.process_video(video_path):
            
            logger.info(f"Frame {result.frame_idx:02d} | Detections: {len(result.detections)}")
            
            for d in result.detections:
                if d.confidence > 0.3:
                    logger.info(f"  -> {d.category.upper()} (conf: {d.confidence:.2f})")
            
            # Il task richiede uno "smoke test 20 frames", quindi ci fermiamo al 20esimo (indice 19)
            if result.frame_idx >= 19:
                logger.info("Raggiunti i 20 frame richiesti dallo smoke test. Interruzione voluta.")
                break
                
    except Exception as e:
        logger.error(f"Errore durante l'elaborazione del video: {e}")

    logger.info("Smoke test concluso!")

if __name__ == "__main__":
    run_smoke_test()
import csv
import logging
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Define the headers for the two CSV files
MOT_HEADERS = [
    "video_id", "frame_id", "track_id", "x1", "y1", "x2", "y2", "conf", "category"
]

TRACKS_HEADERS = [
    "video_id", "track_id", "frame_start", "frame_end", "caption"
]

def create_csv_skeletons(output_dir: str = 'out/run_001') -> None:
    """
    Create both CSV files with headers in the specified directory.
    Make directory if it doesn't exist.
    """
    out_path = Path(output_dir)
    try:
        out_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create directory {output_dir}: {e}")
        return

    mot_path = out_path / "predictions_mot.csv"
    tracks_path = out_path / "predictions_tracks.csv"

    try:
        with open(mot_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(MOT_HEADERS)
        logger.info(f"Created CSV: {mot_path}")

        with open(tracks_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(TRACKS_HEADERS)
        logger.info(f"Created CSV: {tracks_path}")
    except Exception as e:
        logger.error(f"Error creating CSV skeletons: {e}")

class CSVWriter:
    """
    Helper class to append rows to an existing CSV file.
    Assumes the file already has headers.
    """
    def __init__(self, csv_path: str):
        """
        Initialize the CSVWriter.
        
        Args:
            csv_path (str): The path to the CSV file.
        """
        self.csv_path = Path(csv_path)
        self.fieldnames = []
        try:
            with open(self.csv_path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.fieldnames = next(reader)
        except Exception as e:
            logger.error(f"Failed to read headers from {self.csv_path}: {e}")

    def append_row(self, **kwargs) -> None:
        """
        Append a row with named columns.
        """
        try:
            with open(self.csv_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                # Check for any kwargs not in fieldnames
                extra_keys = set(kwargs.keys()) - set(self.fieldnames)
                if extra_keys:
                    logger.warning(f"Extra keys provided that are not in fieldnames: {extra_keys}")
                writer.writerow(kwargs)
        except Exception as e:
            logger.error(f"Failed to write row to {self.csv_path}: {e}")

if __name__ == "__main__":
    test_dir = "out/test_001"
    create_csv_skeletons(test_dir)

    # Initialize writers
    writer_mot = CSVWriter(f"{test_dir}/predictions_mot.csv")
    writer_tracks = CSVWriter(f"{test_dir}/predictions_tracks.csv")

    # Append sample rows to MOT CSV
    writer_mot.append_row(video_id='v1', frame_id=0, track_id=1, x1=100.5, y1=200.3, x2=300.2, y2=400.8, conf=0.95, category='car')
    writer_mot.append_row(video_id='v1', frame_id=0, track_id=2, x1=350.1, y1=150.2, x2=450.0, y2=350.5, conf=0.87, category='person')
    writer_mot.append_row(video_id='v1', frame_id=1, track_id=1, x1=102.3, y1=202.1, x2=302.0, y2=402.5, conf=0.94, category='car')

    # Append sample rows to Tracks CSV
    writer_tracks.append_row(video_id='v1', track_id=1, frame_start=0, frame_end=150, caption='crashed')
    writer_tracks.append_row(video_id='v1', track_id=2, frame_start=5, frame_end=80, caption='running')

    print(f"Sample rows appended for testing.")
    print(f"Ready to integrate with pipeline!")
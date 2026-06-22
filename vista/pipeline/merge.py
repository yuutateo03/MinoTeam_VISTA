import os
import pandas as pd
from glob import glob


def merge_submission_csvs():
    # Define the root output directory
    base_dir = r"C:\Users\yurit\GitHub\cs_1st_year\2nd Semester\CV\Project\MinoTeam_VISTA\vista\pipeline\out\out"

    # 1. Discover all predictions_mot.csv and predictions_tracks.csv files recursively
    all_mot_files = glob(os.path.join(base_dir, "**", "predictions_mot.csv"), recursive=True)
    all_track_files = glob(os.path.join(base_dir, "**", "predictions_tracks.csv"), recursive=True)

    # Safety Check: Exclude master files located directly in the base directory
    mot_files = [f for f in all_mot_files if os.path.dirname(f) != os.path.normpath(base_dir)]
    track_files = [f for f in all_track_files if os.path.dirname(f) != os.path.normpath(base_dir)]

    print(f"Found {len(mot_files)} MOT files and {len(track_files)} Track files in subfolders to merge.")

    # 2. Merge MOT Dataframes
    if mot_files:
        mot_dfs = []
        for f in mot_files:
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    mot_dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {f}. Error: {e}")

        if mot_dfs:
            combined_mot = pd.concat(mot_dfs, ignore_index=True)
            master_mot_path = os.path.join(base_dir, "predictions_mot.csv")
            combined_mot.to_csv(master_mot_path, index=False)
            print(f"✅ Successfully created master MOT file: {master_mot_path} ({len(combined_mot)} rows)")
    else:
        print("❌ No subdirectory predictions_mot.csv files found to merge.")

    # 3. Merge Track Dataframes
    if track_files:
        track_dfs = []
        for f in track_files:
            try:
                df = pd.read_csv(f)
                if not df.empty:
                    track_dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {f}. Error: {e}")

        if track_dfs:
            combined_tracks = pd.concat(track_dfs, ignore_index=True)
            master_track_path = os.path.join(base_dir, "predictions_tracks.csv")
            combined_tracks.to_csv(master_track_path, index=False)
            print(f"✅ Successfully created master Track file: {master_track_path} ({len(combined_tracks)} rows)")
    else:
        print("❌ No subdirectory predictions_tracks.csv files found to merge.")


if __name__ == "__main__":
    merge_submission_csvs()
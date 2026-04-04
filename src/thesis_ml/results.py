import json
import uuid
from pathlib import Path
import pandas as pd


class ResultsManager:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.root_dir / "runs_summary.csv"

    # create_run_dir - creates a unique subdirectory for a single model run.
    # The run_id combines the model name with a short random UUID suffix so that repeated executions do not overwrite one another.
    def create_run_dir(self, model_name: str):
        run_id = f"{model_name}_{uuid.uuid4().hex[:8]}"
        run_dir = self.root_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_id, run_dir

    # save_json - saves a Python object as a formatted JSON file.
    # indent=2 used for improving readability when inspecting saved outputs manually.
    def save_json(self, obj, path: Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    # save_dataframe - saves a pandas DataFrame as a CSV file without the index column.
    def save_dataframe(self, df: pd.DataFrame, path: Path):
        df.to_csv(path, index=False)

    # append_summary_row - appends a dictionary as a new row to the summary CSV file, creating the file if it does not exist.
    def append_summary_row(self, row: dict):
        row_df = pd.DataFrame([row])
        if self.summary_path.exists():
            old = pd.read_csv(self.summary_path)
            row_df = pd.concat([old, row_df], ignore_index=True)
        row_df.to_csv(self.summary_path, index=False)
import os
import json
import pandas as pd

DATA_FOLDER = "data"

# Smart flattener per file

def flatten_json_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = os.path.basename(filepath).replace(".json", "")

    # Custom flattening for deeply nested files
    if filename == "events_sample":
        df = pd.json_normalize(
            data,
            record_path=["events"],
            meta=["fixture_id", "date", "home_team", "away_team", "venue"],
            errors="ignore"
        )

    elif isinstance(data, dict):
        if "response" in data:
            data = data["response"]
        else:
            data = [data]
        df = pd.json_normalize(data)

    elif isinstance(data, list):
        df = pd.json_normalize(data)

    else:
        df = pd.DataFrame()

    # Prefix columns
    df.columns = [f"{filename}.{col}" for col in df.columns]
    df["__source_file__"] = filename
    return df

def load_all_jsons(data_folder=DATA_FOLDER):
    dfs = []
    for file in os.listdir(data_folder):
        if file.endswith(".json"):
            try:
                full_path = os.path.join(data_folder, file)
                df = flatten_json_file(full_path)
                dfs.append(df)
            except Exception as e:
                print(f"[Skipped] {file}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

import json
import os
from typing import List, Dict

LEADERBOARD_FILE = "data/coach_scores.json"

def load_leaderboard() -> List[Dict]:
    """
    Loads the saved leaderboard from disk.
    Returns an empty list if the file doesn't exist.
    """
    if not os.path.exists(LEADERBOARD_FILE):
        return []
    with open(LEADERBOARD_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_score(name: str, score: int, match: str, result: str, tactics: str) -> None:
    """
    Saves a new score entry to the leaderboard file.
    """
    leaderboard = load_leaderboard()
    leaderboard.append({
        "name": name,
        "score": score,
        "match": match,
        "result": result,
        "tactics": tactics
    })
    with open(LEADERBOARD_FILE, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

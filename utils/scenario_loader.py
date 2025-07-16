import json
from typing import Dict, List, Any, Optional


class ScenarioManager:
    def __init__(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            self.scenario: Dict[str, Any] = json.load(f)
        self.current_index = 0

    def get_match_metadata(self) -> Dict[str, str]:
        return {
            "team_1": self.scenario["team_1"],
            "team_2": self.scenario["team_2"],
            "real_result": self.scenario.get("real_result", "N/A"),
            "starting_score": self.scenario.get("starting_score", "0-0"),
            "starting_momentum": self.scenario.get("starting_momentum", self.scenario["team_1"]),
        }

    def get_next_event(self) -> Optional[Dict[str, Any]]:
        if self.current_index >= len(self.scenario["timeline"]):
            return None
        event = self._normalize_event(self.scenario["timeline"][self.current_index])
        self.current_index += 1
        return event

    def get_events_in_range(self, from_minute: int, to_minute: int) -> List[Dict[str, Any]]:
        return [
            self._normalize_event(event)
            for event in self.scenario["timeline"]
            if from_minute <= event["minute"] <= to_minute
        ]

    def get_all_events(self) -> List[Dict[str, Any]]:
        return [self._normalize_event(event) for event in self.scenario["timeline"]]

    def reset(self):
        self.current_index = 0

    def _normalize_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure every event has the fields needed for commentary generation.
        """
        return {
            "minute": event.get("minute", 0),
            "type": event.get("type", "unknown"),  # e.g., goal, card, foul
            "player": event.get("player", "Unnamed Player"),
            "team": event.get("team", "Unknown Team"),
            "description": event.get("description", ""),  # Optional free text
        }

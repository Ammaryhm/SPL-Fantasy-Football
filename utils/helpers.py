import json
import os
import random
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain


def load_json(path):
    """
    Loads a JSON file and returns its content.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detect_formation(lineup):
    from collections import Counter

    position_map = {
        "Goalkeeper": "GK",
        "Defender": "DEF",
        "Midfielder": "MID",
        "Forward": "FWD"
    }

    counts = Counter([position_map.get(p["position"], "UNK") for p in lineup])
    
    defenders = counts["DEF"]
    midfielders = counts["MID"]
    forwards = counts["FWD"]

    # Naive mapping to common formations
    if defenders == 4 and midfielders == 3 and forwards == 3:
        return "4-3-3"
    elif defenders == 4 and midfielders == 4 and forwards == 2:
        return "4-4-2"
    elif defenders == 3 and midfielders == 5 and forwards == 2:
        return "3-5-2"
    elif defenders == 5 and midfielders == 3 and forwards == 2:
        return "5-3-2"
    else:
        return f"{defenders}-{midfielders}-{forwards}"



def generate_lineup_briefing(lineup, language="english"):
    model = ChatOpenAI(temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a tactical football analyst preparing a lineup briefing for a Gen Z fantasy football coach.\n"
         "Be modern, sharp, and strategic — like you're speaking to someone who wants to dominate their match.\n"
         "Mix enthusiasm and analysis. Talk about player strengths, synergy, tactical depth, risks, and X-factors.\n"
         "Insert emojis sparingly to boost engagement (⚽🧤🛡️🎯🔥), but don’t overdo it.\n"
         "Break up the content with short paragraphs or bullet points.\n"
         "Finish with a tone like you're hyping the coach up.\n"
         f"Language: {'Arabic (Saudi)' if language == 'arabic_saudi' else 'English'}"),

        ("user", 
         "Here's the selected fantasy lineup:\n\n"
         "{lineup_text}\n\n"
         "Write a tactical, engaging pre-match briefing based on this squad.")
    ])

    lineup_text = ""
    for p in lineup:
        lineup_text += f"{p['name']} ({p['position']}, {p['age']} yrs) from {p['team']}, SAR {p['cost']:,}\n"

    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.run(lineup_text=lineup_text.strip(), language=language)

    return response



def get_team_by_name(teams, name):
    """
    Returns the team dict matching the given name.
    """
    for team in teams:
        if team["team_name"].lower() == name.lower():
            return team
    raise ValueError(f"Team not found: {name}")


def get_team_by_id(teams, team_id):
    """
    Returns the team dict matching the given ID.
    """
    for team in teams:
        if team["team_id"] == team_id:
            return team
    raise ValueError(f"Team not found with ID: {team_id}")

def load_event_pool(path="data/events_sample.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pool = []
    for match in data:
        for event in match.get("events", []):
            pool.append(event)
    return pool

def generate_fake_event_log(event_pool, num_events=15):
    sampled = random.sample(event_pool, k=min(num_events, len(event_pool)))
    lines = []

    for ev in sampled:
        time_str = ev["time"]
        team = ev["team"]
        player = ev.get("player", "Unknown")
        type_ = ev.get("type", "")
        detail = ev.get("detail", "")
        comment = f" ({ev['comments']})" if ev.get("comments") else ""

        line = f"{time_str} | {team} | {player} | {type_} | {detail}{comment}"
        lines.append(line)

    return "\n".join(lines)

def load_event_pool(path="data/events_sample.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    pool = []
    for match in data:
        for event in match.get("events", []):
            pool.append(event)
    return pool

def generate_fake_event_log(event_pool, num_events=15):
    sampled = random.sample(event_pool, k=min(num_events, len(event_pool)))
    lines = []

    for ev in sampled:
        time_str = ev["time"]
        team = ev["team"]
        player = ev.get("player", "Unknown")
        type_ = ev.get("type", "")
        detail = ev.get("detail", "")
        comment = f" ({ev['comments']})" if ev.get("comments") else ""

        line = f"{time_str} | {team} | {player} | {type_} | {detail}{comment}"
        lines.append(line)

    return "\n".join(lines)

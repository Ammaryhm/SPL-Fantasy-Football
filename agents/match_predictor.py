# utils/match_predictor.py

import random
import json
import pandas as pd
from datetime import datetime
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Optional
from streamlit.components.v1 import html  # Add this at the top of your file
import os

class MatchPrediction(BaseModel):
    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    predicted_result: Literal["Home Win", "Draw", "Away Win"] = Field(description="Predicted match outcome")
    predicted_score: str = Field(description="Predicted scoreline (e.g., '2-1')")
    confidence: int = Field(description="Confidence percentage (1-100)")
    key_factors: list[str] = Field(description="3-4 key factors influencing the prediction")
    player_to_watch: str = Field(description="Key player likely to influence the match")

@st.cache_data
def load_team_stats():
    """Load team performance data for predictions"""
    try:
        # Load standings for current form
        with open("data/standings.json", "r", encoding="utf-8") as f:
            standings = json.load(f)
        
        # Load recent match events for team analysis
        try:
            with open("data/events_sample.json", "r", encoding="utf-8") as f:
                events = json.load(f)
        except FileNotFoundError:
            events = []
        
        # Load top scorers for key players
        with open("data/top_scorers.json", "r", encoding="utf-8") as f:
            top_scorers = json.load(f)
            
        return standings, events, top_scorers
    except Exception as e:
        st.error(f"Error loading team stats: {e}")
        return [], [], []

def calculate_team_strength(team_name, standings_data):
    """Calculate team strength based on current league position and stats"""
    # Handle different team name variations
    team_variations = [
        team_name.lower(),
        team_name.lower().replace("al ", "al-"),
        team_name.lower().replace("al-", "al "),
    ]
    
    for team in standings_data:
        team_standing_name = team.get("team", "").lower()
        
        # Check if any variation matches
        if any(variation == team_standing_name for variation in team_variations):
            points = team.get("points", 0)
            goals_for = team.get("goals_for", 0)
            goals_against = team.get("goals_against", 0)
            goal_difference = goals_for - goals_against
            
            # Enhanced strength calculation
            strength = (points * 0.4) + (goal_difference * 0.3) + (goals_for * 0.2) + (goals_against * -0.1)
            return max(0, strength)  # Ensure non-negative
    
    # Fallback strength based on team reputation if not found in standings
    team_reputation = {
        "al hilal": 75,
        "al nassr": 70, 
        "al ittihad": 68,
        "al ahli": 65,
        "al ettifaq": 55,
        "al shabab": 60,
        "al taawoun": 50,
        "al fayha": 45,
        "al riyadh": 40,
        "damac": 42
    }
    
    for team_key, reputation in team_reputation.items():
        if team_key in team_name.lower():
            return reputation
    
    return 50  # Default neutral strength

def get_team_top_scorer(team_name, top_scorers_data):
    """Get the top scorer for a specific team"""
    # Handle different team name variations
    team_variations = [
        team_name.lower(),
        team_name.lower().replace("al ", "al-"),
        team_name.lower().replace("al-", "al "),
        team_name.lower().replace(" ", ""),
    ]
    
    for player in top_scorers_data:
        player_team = player.get("team", {})
        if isinstance(player_team, dict):
            player_team_name = player_team.get("english", "").lower()
        else:
            player_team_name = str(player_team).lower()
        
        # Check if any team variation matches
        if any(variation == player_team_name for variation in team_variations):
            player_name = player.get("player_name", {})
            if isinstance(player_name, dict):
                return player_name.get("english", "Key Player")
            else:
                return str(player_name) if player_name else "Key Player"
    
    # Fallback: return a realistic player name based on team
    team_star_players = {
        "al hilal": "Salem Al-Dawsari",
        "al nassr": "Cristiano Ronaldo", 
        "al ittihad": "Karim Benzema",
        "al ahli": "Roberto Firmino",
        "al ettifaq": "Georginio Wijnaldum",
        "al shabab": "Ever Banega",
        "al taawoun": "Leandro Damião",
        "al fayha": "Fashion Sakala",
        "al riyadh": "Malik Al-Yami",
        "damac": "Farouk Chafai"
    }
    
    for team_key, player_name in team_star_players.items():
        if team_key in team_name.lower():
            return player_name
    
    return "Key Player"

def generate_ai_prediction(home_team, away_team, standings_data, top_scorers_data):
    """Generate AI-powered match prediction using LLM"""
    
    # Calculate team strengths
    home_strength = calculate_team_strength(home_team, standings_data)
    away_strength = calculate_team_strength(away_team, standings_data)
    
    # Get key players
    home_top_scorer = get_team_top_scorer(home_team, top_scorers_data)
    away_top_scorer = get_team_top_scorer(away_team, top_scorers_data)
    
    # Create context for LLM
    context = f"""
    Match Analysis for Saudi Pro League:
    
    Home Team: {home_team}
    - Current Strength Score: {home_strength:.1f}
    - Key Player: {home_top_scorer}
    - Home Advantage: Yes
    
    Away Team: {away_team}
    - Current Strength Score: {away_strength:.1f}
    - Key Player: {away_top_scorer}
    - Playing Away: Yes
    
    Strength Difference: {home_strength - away_strength:.1f} (positive favors home)
    """
    
    # LLM Prediction Prompt
    prediction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Saudi Pro League match predictor with expertise in football analytics. 
         Based on the provided team data, generate a realistic match prediction.
         
         Consider these factors:
         - Team strength scores (higher is better)
         - Home advantage (worth ~3-5 points)
         - Key players and their impact
         - Typical SPL scoring patterns (1-3 goals per team)
         - Recent form and league position
         
         Provide realistic confidence levels (60-85% for clear favorites, 45-60% for close matches).
         """),
        ("user", "Analyze this upcoming match and provide a detailed prediction:\n\n{context}")
    ])
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        parser = JsonOutputParser(pydantic_object=MatchPrediction)
        
        prediction_chain = prediction_prompt | llm | parser
        
        result = prediction_chain.invoke({
            "context": context,
            "format_instructions": parser.get_format_instructions()
        })
        
        return result
        
    except Exception as e:
        # Fallback to rule-based prediction if LLM fails
        return generate_fallback_prediction(home_team, away_team, home_strength, away_strength, home_top_scorer, away_top_scorer)

def generate_fallback_prediction(home_team, away_team, home_strength, away_strength, home_top_scorer="Key Player", away_top_scorer="Key Player"):
    """Generate a simple rule-based prediction as fallback"""
    
    strength_diff = home_strength - away_strength + 3  # Home advantage
    
    if strength_diff > 8:
        result = "Home Win"
        confidence = random.randint(70, 85)
        home_goals = random.randint(2, 3)
        away_goals = random.randint(0, 1)
        score = f"{home_goals}-{away_goals}"
        key_factors = [
            f"{home_team} superior current form",
            "Strong home advantage",
            f"{home_top_scorer} key attacking threat",
            "Defensive solidity at home"
        ]
        player_to_watch = home_top_scorer
    elif strength_diff < -5:
        result = "Away Win" 
        confidence = random.randint(65, 80)
        home_goals = random.randint(0, 1)
        away_goals = random.randint(2, 3)
        score = f"{home_goals}-{away_goals}"
        key_factors = [
            f"{away_team} excellent away form",
            f"{away_top_scorer} prolific scorer",
            "Superior league position",
            "Recent head-to-head record"
        ]
        player_to_watch = away_top_scorer
    else:
        # For close matches, determine result first, then generate appropriate score
        outcomes = ["Home Win", "Draw", "Away Win"]
        weights = [0.4, 0.3, 0.3] if strength_diff > 0 else [0.3, 0.3, 0.4]
        result = random.choices(outcomes, weights=weights)[0]
        confidence = random.randint(45, 65)
        
        if result == "Draw":
            # Generate a draw score
            goals = random.randint(1, 2)
            score = f"{goals}-{goals}"
            key_factors = [
                "Evenly matched teams",
                "Similar current form",
                "Tactical battle expected",
                "Both teams need points"
            ]
            player_to_watch = random.choice([home_top_scorer, away_top_scorer])
        elif result == "Home Win":
            # Generate home win score
            home_goals = random.randint(1, 3)
            away_goals = random.randint(0, home_goals - 1)  # Ensure home wins
            score = f"{home_goals}-{away_goals}"
            key_factors = [
                "Home advantage factor",
                f"{home_top_scorer} home form",
                "Crowd support crucial",
                "Recent home record"
            ]
            player_to_watch = home_top_scorer
        else:  # Away Win
            # Generate away win score
            away_goals = random.randint(1, 3)
            home_goals = random.randint(0, away_goals - 1)  # Ensure away wins
            score = f"{home_goals}-{away_goals}"
            key_factors = [
                f"{away_team} strong away record",
                f"{away_top_scorer} clinical finishing",
                "Counter-attacking threat",
                "Motivation to climb table"
            ]
            player_to_watch = away_top_scorer
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_result": result,
        "predicted_score": score,
        "confidence": confidence,
        "key_factors": key_factors,
        "player_to_watch": player_to_watch
    }

def get_match_prediction(home_team, away_team):
    """Main function to get match prediction"""
    standings_data, events_data, top_scorers_data = load_team_stats()
    
    if not standings_data:
        return None
    
    # Generate prediction
    prediction = generate_ai_prediction(home_team, away_team, standings_data, top_scorers_data)
    
    return prediction


from streamlit.components.v1 import html  # Make sure this is at the top of your file

def display_prediction_card(prediction):
    if not prediction:
        return

    # Determine card color based on confidence
    confidence = prediction.get("confidence", 50)
    if confidence >= 70:
        card_color = "rgba(34, 197, 94, 0.15)"  # Darker green tint
        border_color = "#22c55e"
    elif confidence >= 55:
        card_color = "rgba(251, 191, 36, 0.15)"  # Darker yellow tint
        border_color = "#fbbf24"
    else:
        card_color = "rgba(239, 68, 68, 0.15)"   # Darker red tint
        border_color = "#ef4444"

    result_emoji = {
        "Home Win": "🏠",
        "Away Win": "✈️", 
        "Draw": "🤝"
    }.get(prediction.get("predicted_result", ""), "⚽")

    key_factors_html = "".join(
        f"<li style='margin: 0.2rem 0;'>{factor}</li>"
        for factor in prediction.get("key_factors", [])
    )

    complete_html = f"""
    <div style="
        background: {card_color};
        border: 2px solid {border_color};
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-family: 'Segoe UI', sans-serif;
        color: #e5e7eb;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h4 style="margin: 0; color: #f3f4f6;">
                {result_emoji} {prediction.get('predicted_result', 'Unknown')}
            </h4>
            <span style="
                background: {border_color};
                color: #000000;
                padding: 0.2rem 0.8rem;
                border-radius: 20px;
                font-size: 0.8rem;
                font-weight: 600;
            ">
                {confidence}% confidence
            </span>
        </div>

        <div style="text-align: center; margin: 1rem 0;">
            <div style="font-size: 1.6rem; font-weight: bold; color: #f9fafb;">
                {prediction.get('predicted_score', '?-?')}
            </div>
            <div style="font-size: 0.9rem; color: #9ca3af;">
                Predicted Final Score
            </div>
        </div>

        <div style="margin-bottom: 0.5rem;">
            <strong style="color: #f3f4f6;">🔍 Key Factors:</strong>
        </div>
        <ul style="margin: 0; padding-left: 1.2rem; color: #d1d5db;">
            {key_factors_html}
        </ul>

        <div style="margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.15);">
            <strong style="color: #f3f4f6;">⭐ Player to Watch:</strong>
            <span style="color: #d1d5db;">{prediction.get('player_to_watch', 'Unknown')}</span>
        </div>
    </div>
    """

    html(complete_html, height=420, scrolling=False)

# Test function
def test_prediction_system():
    """Test the prediction system"""
    print("Testing SPL Match Prediction System...")
    
    # Test prediction
    test_prediction = get_match_prediction("Al Hilal", "Al Nassr")
    
    if test_prediction:
        print(f"Prediction: {test_prediction['predicted_result']}")
        print(f"Score: {test_prediction['predicted_score']}")
        print(f"Confidence: {test_prediction['confidence']}%")
        print(f"Key Factors: {', '.join(test_prediction['key_factors'])}")
        print(f"Player to Watch: {test_prediction['player_to_watch']}")
    else:
        print("Failed to generate prediction")

if __name__ == "__main__":
    test_prediction_system()
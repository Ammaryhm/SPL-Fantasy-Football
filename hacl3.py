# ================================================
#  Importing necessary libraries and packages
# ================================================

import os
import base64
import textwrap
import json
import time
import streamlit as st
from datetime import datetime
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from agents.flatten_json import load_all_jsons
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from htmlTemplates import user_template, bot_template, css
from agents.flags import NATIONALITY_FLAGS
from agents.controlled_simulator import simulate_match_with_leaderboard, reset_match_state, display_leaderboard
from agents.match_predictor import get_match_prediction, display_prediction_card
import streamlit.components.v1 as components

st.markdown("""
<style>
html, body, .stApp, .main, .block-container,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"], [data-testid="stHeader"] {
    background-color: #0f1419 !important;
    color: white !important;
}

h1, h2, h3, h4, h5, h6, p, span, div {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# Page configuration
st.set_page_config(
    page_title="Saudi Pro League Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Force scroll to top on first render after Join SPL Hub
if st.session_state.get("just_registered"):
    components.html(
        """
        <script>
        setTimeout(function() {
            // Reset hash, then scroll to top using location change trick
            window.location.hash = '';
            window.location.hash = '#top';
            window.scrollTo({ top: 0, behavior: 'instant' });
        }, 100);
        </script>
        """,
        height=0,
    )
    st.session_state.just_registered = False

# Add this right after st.set_page_config()
st.markdown("""
<style>
/* Force page to stay at top */
html, body {
    scroll-behavior: auto !important;
}
.main > div {
    scroll-behavior: auto !important;
}
/* Prevent any automatic scrolling */
* {
    scroll-behavior: auto !important;
}
</style>
""", unsafe_allow_html=True)

def get_base64_of_bin_file(file_path):
    """
    Function to read png file and convert it to base64 string
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Image file not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error reading image file: {e}")
        return None


# ================================================
#  Data Loader
# ================================================

@st.cache_data
def load_teams_data():
    with open("data/standings.json", "r", encoding="utf-8") as f:
        standings_data = json.load(f)

    with open("data/teams.json", "r", encoding="utf-8") as f:
        teams_data = json.load(f)

    team_id_to_name = {
        team['team_id']: team['team_name']['english']
        for team in teams_data
    }

    teams = []
    for team_stat in standings_data:
        team_id = team_stat.get('team_id')
        if team_id:
            team_name = team_id_to_name.get(team_id, "Unknown")
        else:
            team_name = team_stat.get("team", "Unknown")

        teams.append({
            "name": team_name,
            "points": team_stat.get('points', 0),
            "wins": team_stat.get('win', 0),
            "draws": team_stat.get('draw', 0),
            "losses": team_stat.get('lose', 0),
            "goals_for": team_stat.get('goals_for', 0),
            "goals_against": team_stat.get('goals_against', 0)

        })

    return pd.DataFrame(teams)



@st.cache_data
def load_players_data():
    with open("data/players.json", "r", encoding="utf-8") as f:
        players_data = json.load(f)

    players = []
    for player in players_data:
        players.append({
            "name": player.get("Player", "Unknown"),
            "position": player.get("Position", "Unknown"),
            "age": player.get("Age", "N/A"),
            "nationality": player.get("Nat.", "Unknown"),
            "team": player.get("Club", "Unknown"),
            "number": player.get("Kit number", "N/A"),
            "price": round(player.get("Market value (€)", 1000000) * 4.07, -3),  # 💰 Rounded SAR
            "overall": player.get("Overall", 60),
            "goals": player.get("Goals", 0),
            "assists": player.get("Assists", 0)
        })

    return pd.DataFrame(players)


@st.cache_data
def load_standings_data():
    with open("data/standings.json", "r", encoding="utf-8") as f:
        standings_data = json.load(f)

    with open("data/teams.json", "r", encoding="utf-8") as f:
        teams_data = json.load(f)

    # Map team names to logos
    team_logo_map = {
        team["team_name"]["english"]: team["logo"]
        for team in teams_data
    }

    standings = []
    for entry in standings_data:
        team_name = entry.get("team", "Unknown")
        standings.append({
            "Rank": entry.get("position", "N/A"),
            "Team": team_name,
            "Logo": team_logo_map.get(team_name, ""),  # fallback blank
            "Played": entry.get("played", 0),
            "Wins": entry.get("won", 0),
            "Draws": entry.get("draw", 0),
            "Losses": entry.get("lost", 0),
            "Goals For": entry.get("goals_for", 0),
            "Goals Against": entry.get("goals_against", 0),
            "Goal Difference": entry.get("goals_for", 0) - entry.get("goals_against", 0),
            "Points": entry.get("points", 0)
        })

    df = pd.DataFrame(standings)
    df = df.sort_values(by="Rank").reset_index(drop=True)
    return df


def format_sar(price):
    if price >= 1_000_000:
        return f"SAR {round(price / 1_000_000)}M"
    elif price >= 1_000:
        return f"SAR {round(price / 1_000)}K"
    else:
        return f"SAR {int(price)}"

@st.cache_data
def load_top_scorers():
    with open("data/top_scorers.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Flatten name and team fields
    top_scorers = []
    for player in raw_data:
        top_scorers.append({
            "name": player["player_name"]["english"],
            "team": player["team"]["english"],
            "nationality": player.get("nationality", "-"),
            "age": player.get("age", "-"),
            "goals": player.get("goals", 0),
            "assists": player.get("assists", 0),
            "appearances": player.get("appearances", 0),
            "minutes_played": player.get("minutes_played", 0)
        })

    return pd.DataFrame(top_scorers)



# ================================================
#  Opening Page
# ================================================

# Initialize session state
if not st.session_state.get("user_registered", False):
    # ========== TEAM CONFIGURATION ==========
    # ========== SINGLE-TEAM CONFIGURATION FOR PROTOTYPE ==========
    team_config = {
        "Al-Nassr": {
            "logo": "https://media.api-sports.io/football/teams/2939.png",
            "colors": {"primary": "#ffff00", "secondary": "#2c5282"}
        }
    }

    team_names = list(team_config.keys())  # Will contain only ['Al-Nassr']

    # ========== PROFESSIONAL UI LAYOUT ==========
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="registration-card">', unsafe_allow_html=True)
    
    # Header section
    st.markdown('<div class="header-section">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">Saudi Pro League Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Join the premier destination for SPL fans and analytics</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Form section
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    
    # Username input
    st.markdown('<div class="form-group">', unsafe_allow_html=True)
    st.markdown('<label class="form-label">Username</label>', unsafe_allow_html=True)

    username_input = st.text_input(
        "", 
        placeholder="Enter your username (no spaces)", 
        key="input_username", 
        label_visibility="collapsed"
    )

    st.markdown('</div>', unsafe_allow_html=True)


    # Team selection
    st.markdown('<div class="form-group">', unsafe_allow_html=True)
    st.markdown('<label class="form-label">Favorite Team</label>', unsafe_allow_html=True)

    favorite_team = st.selectbox(
        "", 
        team_names, 
        index=0, 
        key="input_fav_team", 
        label_visibility="collapsed"
    )

    # Team preview
    if favorite_team in team_config:
        team = team_config[favorite_team]
        st.markdown(f'''
        <div class="team-preview">
            <img src="{team['logo']}" class="team-logo-small" alt="{favorite_team}" />
            <span class="team-name">{favorite_team}</span>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


    # Submit button
    st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)

    if st.button("Join SPL Hub", key="enter_button", type="primary"):
        if not username_input.strip():
            st.error("Please enter a username!")
        elif " " in username_input:
            st.error("Username must not contain spaces.")
        else:
            st.session_state.username = username_input.strip()
            st.session_state.favorite_team = favorite_team
            st.session_state.selected_team_logo = team_config[favorite_team]["logo"]
            st.session_state.team_colors = team_config[favorite_team]["colors"]
            st.session_state.user_registered = True
            # Add this line:
            st.session_state.just_registered = True
            st.rerun()


    st.markdown('</div>', unsafe_allow_html=True)


    # Close containers
    st.markdown('</div>', unsafe_allow_html=True)  # Close form-section
    st.markdown('</div>', unsafe_allow_html=True)  # Close registration-card
    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

    st.stop()



# ========== POST-REGISTRATION UI ==========
user = st.session_state.get("username", "Guest")
fav_team = st.session_state.get("favorite_team", "None")
team_logo = st.session_state.get("selected_team_logo", "")
team_colors = st.session_state.get("team_colors", {"primary": "#FFD700", "secondary": "#1E3A8A"})

# Sidebar recap
with st.sidebar:
    st.markdown("### 🎯 Your Profile")
    if team_logo:
        st.image(team_logo, width=100)
    st.markdown(f"- **Username:** `{user}`")
    st.markdown(f"- **Team:** `{fav_team}`")

    st.markdown("**Team Colors:**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div style="height: 30px; background: {team_colors['primary']}; border-radius: 4px;"></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div style="height: 30px; background: {team_colors['secondary']}; border-radius: 4px;"></div>""", unsafe_allow_html=True)




# ================================================
#  Sidebar
# ================================================

def create_professional_sidebar():
    """
    Creates a clean sidebar with a search bar, language switch (non-functional),
    and white-colored links, except for the 'Navigation' header in blue.
    """
    st.markdown("""
    <style>
    .sidebar-header {
        font-size: 26px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 25px;
        text-align: center;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 10px;
    }

    .search-box {
        margin: 20px 0 30px 0;
    }

    .nav-link {
        display: block;
        padding: 18px 24px;
        margin-bottom: 30px;
        font-size: 18px;
        font-weight: 600;
        text-align: center;
        color: white;
        text-decoration: none;
        border-bottom: 1px solid white;
        transition: all 0.3s ease;
    }

    .nav-link:hover {
        background-color: white;
        color: black;
        border-radius: 4px;
    }

    .lang-button {
        display: inline-block;
        padding: 10px 20px;
        margin: 0 10px 30px 10px;
        font-size: 14px;
        font-weight: 600;
        color: white;
        background-color: transparent;
        border: 1px solid white;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
        transition: all 0.3s ease;
    }

    .lang-button:hover {
        background-color: white;
        color: black;
    }

    .footer-text {
        font-size: 13px;
        color: white;
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid white;
    }

    .company-name {
        font-weight: 500;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)

        # Search bar (non-functional)
        with st.container():
            st.markdown('<div class="search-box">', unsafe_allow_html=True)
            st.text_input("Search", placeholder="Search this site...", key="sidebar_search")
            st.markdown('</div>', unsafe_allow_html=True)

        # Nav links
        st.markdown('<a href="#" class="nav-link">Contact Us</a>', unsafe_allow_html=True)
        st.markdown('<a href="#" class="nav-link">FAQ</a>', unsafe_allow_html=True)
        st.markdown('<a href="#" class="nav-link">About</a>', unsafe_allow_html=True)
        st.markdown('<a href="#" class="nav-link">Privacy</a>', unsafe_allow_html=True)
        st.markdown('<a href="#" class="nav-link">Terms</a>', unsafe_allow_html=True)

        # Language switch (non-functional)
        st.markdown("""
        <div style='text-align: center; margin-bottom: 40px;'>
            <a href="#" class="lang-button">English</a>
            <a href="#" class="lang-button">العربية</a>
        </div>
        """, unsafe_allow_html=True)

        current_year = datetime.now().year
        st.markdown(f"""
        <div class="footer-text">
            <span class="company-name">SPL Hub by Accenture Song</span><br>
            © {current_year} All rights reserved.
        </div>
        """, unsafe_allow_html=True)


def main():
    create_professional_sidebar()
    st.markdown("<!-- Main content goes here -->", unsafe_allow_html=True)

if __name__ == "__main__":
    main()


# ================================================
#  Header
# ================================================

header_image = "https://media.licdn.com/dms/image/v2/D4D12AQEAKW-BkO6dkQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1689759291697?e=2147483647&v=beta&t=g4ujbwKF0bkpbX_JobYfuDacN2IF8Q7eoS_SGMuBtyA"
spl_logo = "https://play-lh.googleusercontent.com/DFbRC2VtMIy3FXGo7QxEdpx225mjoHM8uF0ct3faUMWDT1rIMlYFhJXRupG5QzkV-Q=w600-h300-pc0xffffff-pd"
acc_song = "https://www.proi.com/uploaded/companies/logos/17017977352575(1).jpg"


# Header
st.markdown(f"""
<style>
.header-container {{
   width: 100%;
   height: 160px;
   background-image: url("{header_image}");
   background-size: cover;
   background-position: center bottom;
   background-repeat: no-repeat;
   background-blend-mode: overlay;
   background-color: rgba(0, 0, 0, 0.7);
   border-radius: 15px;
   padding: 1.5rem 2rem;
   color: white;
   display: flex;
   flex-direction: row;
   align-items: center;
   position: relative;
}}

.header-section {{
   flex: 1;
   display: flex;
   justify-content: center;
   align-items: center;
}}

.left-section {{
   justify-content: flex-start;
}}

.center-section {{
   justify-content: center;
}}

.right-section {{
   justify-content: flex-end;
}}

.header-logo {{
   height: 150px;
}}

.header-subtitle-large {{
    font-size: 2rem;
    font-weight: 500;
    margin: 0 !important;
    padding: 0;
    text-shadow: 1px 1px 3px black;
    line-height: 1.1;
    display: block;
}}

.header-subtitle-sub {{
    font-size: 1.1rem;
    font-weight: 400;
    opacity: 0.85;
    margin-top: 0.4rem;
    text-shadow: 1px 1px 2px black;
}}


.powered-by {{
   position: absolute;
   top: 8px;
   right: 20px;
   font-size: 1.4rem;
   font-weight: 500;
   text-shadow: 1px 1px 3px black;
   display: flex;
   align-items: center;
   gap: 0.4rem;
}}

.powered-by img {{
   height: 40px;
   vertical-align: middle;
}}
</style>

<div class="header-container">
   <div class="header-section left-section">
       <div>
   <div class="header-subtitle-large">Your Ultimate Destination for Saudi Professional Football</div>
   <div class="header-subtitle-sub">Don't just watch the league. Be apart of it.</div>
</div>
   </div>
   <div class="header-section center-section">
       <img src="{spl_logo}" alt="SPL Logo" class="header-logo" />
   </div>
   <div class="header-section right-section">
       <!-- Reserved space for balance -->
   </div>
   <div class="powered-by">Powered by <img src="{acc_song}" alt="Accson Logo" /></div>
</div>
""", unsafe_allow_html=True)


# ================================================
# Summary Cards
# ================================================

total_fans_url = "https://resources.saudi-pro-league.pulselive.com/photo-resources/2024/08/04/eb7745ea-8979-48dc-ae7f-fbac260b9080/GettyImages-1695856463.jpg?width=1868&height=1136"
matches_played_url = "https://stadiumastro-kentico.s3.amazonaws.com/stadiumastro/media/perform-article/2018/may/16/fahadalmirdasi-cropped_tpznlq2wpya1qn3ipshxkpp4.jpg?ext=.jpg"
attendance_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Entire_King_Saud_University_Stadium.jpg/1920px-Entire_King_Saud_University_Stadium.jpg"

ittihad_logo = "https://upload.wikimedia.org/wikipedia/en/thumb/8/87/Al-Ittihad_Club_%28Jeddah%29_logo.svg/800px-Al-Ittihad_Club_%28Jeddah%29_logo.svg.png"
hilal_logo = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Al_Hilal_SFC_Logo.svg/464px-Al_Hilal_SFC_Logo.svg.png"

st.markdown("""
    <div style="margin-top: 1rem;"></div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# Total SPL Fanbase
with col1:
    st.markdown(f"""
    <div style="
        position: relative;
        background-image: url('{attendance_url}');
        background-size: cover;
        background-position: center;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        overflow: hidden;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
    ">
        <div style="
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.5);
            z-index: 0;
        "></div>
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 0.9rem; font-weight: 600; text-transform: uppercase; text-shadow: 1px 1px 3px black;">Total SPL Fanbase</div>
            <div style="font-size: 2rem; font-weight: 700; text-shadow: 1px 1px 3px black;">28.5K</div> 
            <div style="font-size: 0.85rem; font-weight: 500; text-shadow: 1px 1px 3px black;">↗ +2.1K</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Matches Played / Live Match
with col2:
    st.markdown(f"""
    <div style="position: relative;
        background-image: url('{matches_played_url}');
        background-size: cover;
        background-position: center;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        overflow: hidden;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
    ">
        <div style="position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.6);
            z-index: 0;
        "></div>
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 0.85rem; font-weight: 700; color: white; letter-spacing: 0.5px; margin-bottom: 0.6rem;
                text-shadow: 1px 1px 3px black;">
                🔴 LIVE MATCH
            </div>
            <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap;">
                <img src="{ittihad_logo}" alt="Ittihad" style="height: 36px; margin-right: 8px;" />
                <span style="font-size: 1rem; margin-right: 12px; text-shadow: 1px 1px 3px black;">Al Ittihad</span>
                <span style="font-size: 1.3rem; font-weight: 800; text-shadow: 1px 1px 3px black;">1 - 0</span>
                <span style="font-size: 1rem; margin-left: 12px; text-shadow: 1px 1px 3px black;">Al Hilal</span>
                <img src="{hilal_logo}" alt="Hilal" style="height: 36px; margin-left: 8px;" />
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Active SPL Fanbase
with col3:
    st.markdown(f"""
    <div style="
        position: relative;
        background-image: url('{total_fans_url}');
        background-size: cover;
        background-position: center;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.6);
        overflow: hidden;
        height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
    ">
        <div style="
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.6);
            z-index: 0;
        "></div>
        <div style="position: relative; z-index: 1;">
            <div style="font-size: 0.9rem; font-weight: 600; text-transform: uppercase; text-shadow: 1px 1px 3px black;">Active SPL Fanbase</div>
            <div style="font-size: 2rem; font-weight: 700; text-shadow: 1px 1px 3px black;">13.4K</div>
            <div style="font-size: 0.85rem; font-weight: 500; text-shadow: 1px 1px 3px black;">↗ +203</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ================================================
# Tabs
# ================================================

st.markdown("""
<style>
/* Tab container styling */
.stTabs [data-baseweb="tab-list"] {
   flex-wrap: nowrap;
   justify-content: space-between;
   border-radius: 12px;
   padding: 8px;
   margin-bottom: 2rem;
   background: rgba(0,0,0,0.1);
   box-shadow: 0 2px 10px rgba(0,0,0,0.1);
   backdrop-filter: blur(10px);
}

/* Shared tab styling */
.stTabs [data-baseweb="tab"] {
   position: relative;
   flex-grow: 1;
   justify-content: center;
   font-weight: 600;
   font-size: 1rem;
   padding: 12px 16px;
   border-radius: 8px;
   transition: all 0.3s ease;
   background: transparent;
   border: none;
   color: white;
   display: flex;
   align-items: center;
   text-align: center;
}

/* Divider between tabs */
.stTabs [data-baseweb="tab"]:not(:last-child)::after {
   content: "";
   position: absolute;
   right: -8px;
   top: 50%;
   transform: translateY(-50%);
   height: 60%;
   width: 2px;
   background-color: rgba(255, 255, 255, 0.3);
}

/* Tab 1: Green */
.stTabs [data-baseweb="tab"]:nth-child(1) {
   background: linear-gradient(135deg, rgba(0, 122, 61, 0.5), rgba(0, 90, 45, 0.5));
}
.stTabs [data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {
   background: linear-gradient(135deg, #007A3D, #005A2D);
}

/* Tab 2: Blue */
.stTabs [data-baseweb="tab"]:nth-child(2) {
   background: linear-gradient(135deg, rgba(35, 101, 174, 0.5), rgba(26, 76, 133, 0.5));
}
.stTabs [data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {
   background: linear-gradient(135deg, #2365AE, #1A4C85);
}

/* Tab 3: Gold */
.stTabs [data-baseweb="tab"]:nth-child(3) {
   background: linear-gradient(135deg, rgba(255, 216, 107, 0.5), rgba(204, 154, 0, 0.5));
}
.stTabs [data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {
   background: linear-gradient(135deg, #FFD86B, #CC9A00);
}

/* Tab 4: Orange */
.stTabs [data-baseweb="tab"]:nth-child(4) {
   background: linear-gradient(135deg, rgba(255, 153, 0, 0.5), rgba(204, 122, 0, 0.5));
}
.stTabs [data-baseweb="tab"]:nth-child(4)[aria-selected="true"] {
   background: linear-gradient(135deg, #FF9900, #CC7A00);
}

/* Tab 5: Red */
.stTabs [data-baseweb="tab"]:nth-child(5) {
   background: linear-gradient(135deg, rgba(220, 53, 69, 0.5), rgba(172, 41, 55, 0.5));
}
.stTabs [data-baseweb="tab"]:nth-child(5)[aria-selected="true"] {
   background: linear-gradient(135deg, #DC3545, #AC2937);
}

/* Tab 6: Purple (Fan Zone) */
.stTabs [data-baseweb="tab"]:nth-child(6) {
   background: linear-gradient(135deg, rgba(111, 66, 193, 0.5), rgba(85, 40, 158, 0.5));
}
.stTabs [data-baseweb="tab"]:nth-child(6)[aria-selected="true"] {
   background: linear-gradient(135deg, #6F42C1, #55289E);
}

/* Explicitly remove divider from last tab */
.stTabs [data-baseweb="tab"]:nth-child(6)::after {
   display: none !important;
}

/* Remove focus outline */
.stTabs [data-baseweb="tab"]:focus {
   outline: none;
   box-shadow: 0 0 0 2px rgba(255,255,255,0.3);
}

/* Tab content area */
.stTabs [data-baseweb="tab-panel"] {
   padding-top: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# Update your tab definition with the 6th one added
st.markdown('<div id="main-tabs">', unsafe_allow_html=True)
tabs = st.tabs(["🏠 Home", "🏆 League Table", "⚽ Fantasy Football", "📊 Statistics", "📅 Fixtures", "🧑‍🤝‍🧑 Fan Zone"])
st.markdown('</div>', unsafe_allow_html=True)



# ================================================
#   Home tab
# ================================================

featured_article_image = "https://www.vbetnews.com/uploads/uploads-new/posts/hrachoff22/1686122369.jpg"

match_report_url = "https://i2-prod.manchestereveningnews.co.uk/article26229946.ece/ALTERNATES/s1200b/1_CR7-Al-Nassr.jpg"
team_news_url = "https://telegrafi.com/wp-content/uploads/2024/02/17071612748874-e1707206093814-780x439.jpg"
league_news_url = "https://www.arabianbusiness.com/cloud/2023/07/25/Alittiahad-alshabab.jpg"


# Determine which tab is active
page_mapping = {
    0: "🏠 Home",
    1: "🏆 League Table", 
    2: "⚽ Fantasy Football",
    3: "📊 Statistics",
    4: "📅 Fixtures",
    5: "🧑‍🤝‍🧑 Fan Zone"
}

# Main content based on selected tab
with tabs[0]:  # Home Tab
   
    # News & Headlines Section
    st.markdown("## 📰 News & Headlines")
    
    # Featured Article
    st.markdown(f"""
    <div style="background-image: url('{featured_article_image}');
                background-size: cover;
                background-position: center 30%;
                border-radius: 15px;
                padding: 2rem;
                margin: 1.5rem 0;
                color: white;
                position: relative;
                overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0;
                    width: 100%; height: 100%;
                    background: linear-gradient(135deg, rgba(0,0,0,0.4), rgba(0,0,0,0.8));
                    z-index: 0;"></div>
        <div style="position: relative; z-index: 1;">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem;
                            border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                    BREAKING NEWS
                </span>
                <span style="font-size: 0.9rem; opacity: 0.8;">30 minutes ago</span>
            </div>
            <h2 style="margin: 0 0 1rem 0; font-size: 2rem; font-weight: 700;">
                Al Riyadh Secure Top Spot with Stunning 4-0 Victory
            </h2>
            <p style="font-size: 1.1rem; line-height: 1.6; margin: 0; opacity: 0.9;">
                Akhmat Grozny's side dominated Al Shabab in a commanding display at Kingdom Arena,
                extending their lead at the summit of the Saudi Pro League table.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Latest Articles Grid
    col1, col2, col3 = st.columns(3)

    articles = [
        {
            "title": "Ronaldo's Free-Kick Masterclass Seals Derby Victory",
            "summary": "The Portuguese superstar's 89th-minute stunner secured all three points for Al Nassr in the Riyadh Derby thriller.",
            "time": "2 hours ago",
            "category": "Match Report",
            "bg": match_report_url,
            "position": "center 90%"
        },
        {
            "title": "Benzema Returns to Training Ahead of Crucial Fixture",
            "summary": "Al Ittihad's French striker is back in full training and ready to face Al Ahli in this weekend's showdown.",
            "time": "4 hours ago",
            "category": "Team News",
            "bg": team_news_url,
            "position": "center 0%"
        },
        {
            "title": "Saudi Pro League Sets New Attendance Record",
            "summary": "Over 580,000 fans attended matches this weekend, marking the highest weekly attendance in league history.",
            "time": "6 hours ago",
            "category": "League News",
            "bg": league_news_url,
            "position": "center 100%"
        }
    ]


    for i, article in enumerate(articles):
        with [col1, col2, col3][i]:
            html = f"""<div style="
        position: relative;
        background-image: url('{article['bg']}');
        background-size: cover;
        background-position: {article['position']};
        border-radius: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        overflow: hidden;
        height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        padding: 1rem;
        text-align: left;
        color: white;">
        <div style="
            position: absolute;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: rgba(0,0,0,0.7);
            z-index: 0;"></div>
        <div style="position: relative; z-index: 1; width: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <span style="background: #007A3D80; color: white; padding: 0.2rem 0.8rem; border-radius: 15px; font-size: 0.75rem; font-weight: 600;">
                    {article['category']}
                </span>
                <span style="font-size: 0.8rem; color: white;">{article['time']}</span>
            </div>
            <h4 style="margin: 0 0 0.8rem 0; font-size: 1.1rem; font-weight: 600; line-height: 1.3; color: white;">
                {article['title']}
            </h4>
            <p style="margin: 0; font-size: 0.9rem; line-height: 1.4; flex-grow: 1; color: white;">
                {article['summary']}
            </p>
        </div>
    </div>"""
            st.markdown(textwrap.dedent(html), unsafe_allow_html=True)

    # More News Section
    st.markdown("### 📋 More Headlines")
    
    more_news = [
        {"title": "Al Ettifaq Sign Brazilian Midfielder in January Window", "time": "8 hours ago", "category": "Transfer News"},
        {"title": "VAR Decision Sparks Controversy in Al Ahli vs Al Taawoun", "time": "12 hours ago", "category": "Match Report"},
        {"title": "Young Saudi Talents Shine in Youth Development Program", "time": "1 day ago", "category": "Development"},
        {"title": "New Stadium Technology Enhances Fan Experience", "time": "1 day ago", "category": "Infrastructure"},
        {"title": "Saudi Pro League Partners with Global Broadcasting Network", "time": "2 days ago", "category": "Business"}
    ]
    
    for news in more_news:
        st.markdown(f"""
        <div style="border-left: 3px solid #2a5298; padding: 1rem; margin: 0.8rem 0; background: #2365AE80; border-radius: 0 8px 8px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                <span style="background: #007A3D80; color: white; padding: 0.2rem 0.6rem; border-radius: 12px; font-size: 0.7rem; font-weight: 600;">{news['category']}</span>
                <span style="color: white; font-size: 0.8rem;">{news['time']}</span>
            </div>
            <h5 style="margin: 0; font-size: 1rem; font-weight: 600; color: white;">{news['title']}</h5>
        </div>
        """, unsafe_allow_html=True)


# ================================================
#   League Table tab
# ================================================

with tabs[1]:  # League Table Tab
    st.markdown("## 📋 Current League Standings")

    # Load data
    standings_df = load_standings_data()

    # Team logos as HTML
    standings_df["Logo"] = standings_df["Logo"].apply(
        lambda url: f"<img src='{url}' width='28'>" if url else ""
    )

    # Reorder columns
    ordered_cols = ["Rank", "Logo", "Team", "Played", "Wins", "Draws", "Losses",
                    "Goals For", "Goals Against", "Goal Difference", "Points"]
    standings_df = standings_df[ordered_cols]

    # Dropdown (multi-select) filter
    unique_teams = standings_df["Team"].unique().tolist()
    selected_teams = st.multiselect(
        label="🔍 Select one or more teams to filter",
        options=unique_teams,
        default=None,
        placeholder="Choose team(s)..."
    )

    # Apply filter (if any)
    if selected_teams:
        filtered_df = standings_df[standings_df["Team"].isin(selected_teams)]
    else:
        filtered_df = standings_df

    # Centered HTML table
    table_html = filtered_df.to_html(escape=False, index=False)

    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <div style="width: 100%; max-width: 900px;">
                {table_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("⚽ Rank is based on final league standings in Football-API.")


# ================================================
#   Fantasy Football tab
# ================================================

POSITION_COLOR_CODES = {
    "green": "#28a745",
    "orange": "#ffc107",
    "red": "#dc3545"
}

# Replace the fantasy football tab section (around line 350-450) with this functional version

# In your hacl3.py file, replace the Fantasy Football tab section with this:

# In your hacl3.py file, replace the Fantasy Football tab section with this:

# Add this line near the top of your Fantasy Football tab section (around line 350)
# Right after the tab definition and before the session state initialization

with tabs[2]:  # Fantasy Football Tab
    # Load players data for this tab
    players_df = load_players_data()  # Add this line
    
    st.markdown("""
    <style>
        /* Neutralize tab colors */
        section[data-testid="stTabs"] div[role="tablist"] > div {
            background-color: transparent !important;
            color: inherit !important;
            border: none !important;
            border-bottom: 2px solid #DDD !important;
        }

        section[data-testid="stTabs"] div[role="tablist"] > div[aria-selected="true"] {
            font-weight: 600 !important;
            border-bottom: 3px solid black !important;
            background-color: transparent !important;
        }

        section[data-testid="stTabs"] div[role="tablist"] > div:hover {
            background-color: rgba(0, 0, 0, 0.04) !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <h2 style="text-align: left; margin-bottom: 0;">🏆 Fantasy Football Manager</h2>
    <p style="text-align: left; font-size: 1.1rem; margin-top: 0.2rem;">
    Build your dream team and compete with friends!
    </p>
    """, unsafe_allow_html=True)

    # Initialize session state for fantasy team and budget
    if "fantasy_team" not in st.session_state:
        st.session_state.fantasy_team = []
    if "budget" not in st.session_state:
        st.session_state.budget = 100_000_000

    # Create nested tabs with neutral styling
    fantasy_tabs = st.tabs(["👥 Build Your Team", "🎮 Match Day"])

    # === TAB 1: Build Your Team ===
    with fantasy_tabs[0]:  # Build Your Team tab
        col1, col2 = st.columns([2, 1])  # Add this line to create the columns
        
        with col1:  # Add this line
            if len(st.session_state.fantasy_team) == 0:
                st.markdown("### 👥 No Players Selected")
                st.info("🎯 Start building your squad! Use the 'Add Random Players' button below to get your randomized team.")

                if st.button("🎲 Add My Starting 11", type="primary", key="add_starting_11"):
                    predefined_names = [
                        "Nawaf Al-Aqidi", "Ahmed Hegazy", "Bruno Henrique", "Paulo Victor",
                        "Sanousi Al-Hawsawi", "Abdullah Al-Salem", "Otabek Shukurov", "Pedro Rebocho", "Saeed Al-Hamsl", "Abdulrahman Ghareeb",
                        "Ali Al-Hassan", "Cristiano Ronaldo"
                    ]

                    selected_players = []
                    for name in predefined_names:
                        player_row = players_df[players_df['name'] == name]
                        if not player_row.empty:
                            selected_players.append(player_row.iloc[0].to_dict())

                    st.session_state.fantasy_team = []
                    st.session_state.budget = 100_000_000

                    for player in selected_players:
                        st.session_state.fantasy_team.append({
                            "name": player["name"],
                            "position": player["position"],
                            "team": player["team"],
                            "price": player["price"],
                            "overall": player.get("overall", 60),
                            "age": player.get("age", "N/A"),
                            "nationality": player.get("nationality", "Unknown"),
                            "goals": player.get("goals", 0),
                            "assists": player.get("assists", 0)
                        })
                        st.session_state.budget -= player["price"]


            else:
                st.markdown(f"### 👥 Your Squad ({len(st.session_state.fantasy_team)}/15)")

                # Sort controls for your squad
                sort_col1, sort_col2 = st.columns(2)
                with sort_col1:
                    sort_by = st.selectbox("Sort by", ["name", "price", "overall", "position"])
                with sort_col2:
                    sort_order = st.selectbox("Order", ["Ascending", "Descending"])

                # Filter controls for your squad
                if len(st.session_state.fantasy_team) > 0:
                    squad_positions = list(set([p["position"] for p in st.session_state.fantasy_team]))
                    squad_teams = list(set([p["team"] for p in st.session_state.fantasy_team]))
                    
                    filter_col1, filter_col2 = st.columns(2)
                    with filter_col1:
                        position_filter = st.selectbox("Filter by Position", ["All"] + sorted(squad_positions))
                    with filter_col2:
                        team_filter = st.selectbox("Filter by Team", ["All"] + sorted(squad_teams))

                    # Apply filters to YOUR SQUAD ONLY
                    filtered_players = st.session_state.fantasy_team.copy()
                    
                    if position_filter != "All":
                        filtered_players = [p for p in filtered_players if p["position"] == position_filter]
                    if team_filter != "All":
                        filtered_players = [p for p in filtered_players if p["team"] == team_filter]
                    
                    # Sort your squad
                    ascending = sort_order == "Ascending"
                    filtered_players = sorted(filtered_players, key=lambda x: x[sort_by], reverse=not ascending)

                    # Display your filtered squad
                    st.markdown("---")
                    st.markdown(f"**Showing {len(filtered_players)} of your players**")
                    
                    # Display your squad players with remove buttons only
                    for idx, player in enumerate(filtered_players):
                        with st.container():
                            player_col1, player_col2, player_col3 = st.columns([3, 1, 1])

                            nationality = player.get("nationality", "")
                            flag_url = NATIONALITY_FLAGS.get(nationality, "")
                            flag_img = f"<img src='{flag_url}' width='20' style='margin-left: 8px; vertical-align: middle;'/>" if flag_url else ""
                            
                            position_emojis = {"Goalkeeper": "🧤", "Defender": "🛡️", "Midfielder": "🎯", "Forward": "⚽"}
                            emoji = position_emojis.get(player["position"], "👤")

                            with player_col1:
                                st.markdown(f"""
                                <div style="padding-left: 0.5rem; border-left: 3px solid #28a745; margin-bottom: 1.2rem;">
                                    <strong>{emoji} {player['name']}{flag_img}</strong><br>
                                    <span style="color: #666;">{player['team']} • {player['position']} • Age: {player.get('age', '-')}</span>
                                </div>
                                """, unsafe_allow_html=True)

                            with player_col2:
                                overall = player.get("overall", 60)
                                price = player["price"]
                                st.markdown(f"""
                                <div style='text-align: center; font-weight: bold; font-size: 0.95rem; margin-top: 0.4rem; color: inherit;'>
                                    {overall} | {format_sar(price)}
                                </div>
                                """, unsafe_allow_html=True)

                            with player_col3:
                                # Only show remove button since these are YOUR players
                                if st.button("Remove", key=f"remove_{idx}_{player['name']}", type="secondary"):
                                    # Remove player from team
                                    st.session_state.fantasy_team = [p for p in st.session_state.fantasy_team if p["name"] != player['name']]
                                    st.session_state.budget += player["price"]
                                    st.rerun()

        with col2:
            team_size = len(st.session_state.fantasy_team)
            total_budget = 100_000_000
            remaining_budget = st.session_state.budget
            used_budget = total_budget - remaining_budget

            used_m = round(used_budget / 1_000_000, 2)
            remaining_m = round(remaining_budget / 1_000_000, 2)

            # Budget Summary Box
            st.markdown(f"""
                <div style='position: relative; padding: 1rem; background: rgba(40, 167, 69, 0.1); border-radius: 10px; margin-bottom: 1rem;'>
                    <div style='text-align: center;'>
                        <div style='font-weight: 600; font-size: 1.1rem; color: white;'>Team Budget</div>
                        <div style='font-size: 1.8rem; font-weight: bold; color: #28a745;'>SAR {used_m:.1f}M / SAR 100M</div>
                        <div style='font-size: 1rem; font-weight: 500; color: #ccc;'>
                            SAR {remaining_m:.1f}M remaining
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Team Average Overall Box
            if team_size > 0:
                team_avg = sum(p.get("overall", 60) for p in st.session_state.fantasy_team) / team_size
                st.markdown(f"""
                    <div style='position: relative; padding: 1rem; background: rgba(255, 216, 107, 0.5); border-radius: 10px; margin-bottom: 1rem;'>
                        <div style='text-align: center;'>
                            <div style='font-weight: 600; font-size: 1.1rem; color: #333;'>Team Overall</div>
                            <div style='font-size: 1.8rem; font-weight: bold; color: white;'>{team_avg:.1f}</div>
                            <div style='font-size: 1rem; font-weight: 500; color: #333;'>
                                across {team_size} players
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            # Position Requirements
            st.markdown("#### 📋 Squad Requirements")
            limits = {"Goalkeeper": 2, "Defender": 5, "Midfielder": 5, "Forward": 3}
            position_emojis = {"Goalkeeper": "🧤", "Defender": "🛡️", "Midfielder": "🎯", "Forward": "⚽"}

            current_positions = {}
            for p in st.session_state.fantasy_team:
                pos = p["position"]
                current_positions[pos] = current_positions.get(pos, 0) + 1

            for pos, limit in limits.items():
                current = current_positions.get(pos, 0)
                emoji = position_emojis.get(pos, "")

                if current == 0:
                    color = "#dc3545"  # Red - missing entirely
                elif current < limit:
                    color = "#ffc107"  # Yellow - needs more
                elif current == limit:
                    color = "#28a745"  # Green - complete
                else:
                    color = "#dc3545"  # Red - too many

                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem; margin: 0.3rem 0; background: rgba(0,0,0,0.05); border-radius: 5px;">
                    <span>{emoji} {pos}</span>
                    <span style="color: {color}; font-weight: bold;">{current}/{limit}</span>
                </div>
                """, unsafe_allow_html=True)

            # Current Squad display
            st.markdown("---")
            if team_size == 11:
                st.markdown(f"#### 👥 Current Squad (11/11)")
            else:
                st.markdown(f"#### 👥 Current Squad ({team_size}/11)")

            if team_size > 0:
                # Captain / Vice Captain selection
                player_names = [p["name"] for p in st.session_state.fantasy_team]
                captain = st.selectbox("Select Captain", player_names, key="captain_select")
                vice_captain = st.selectbox("Select Vice Captain", [name for name in player_names if name != captain], key="vice_captain_select")

                st.session_state.captain = captain
                st.session_state.vice_captain = vice_captain

                st.markdown(f"**Captain:** {captain}  \n**Vice Captain:** {vice_captain}")

                # Player List
                for i, player in enumerate(st.session_state.fantasy_team):
                    emoji = position_emojis.get(player["position"], "👤")
                    st.markdown(f"""
                    <div style="font-size: 0.9rem; padding: 0.3rem; margin: 0.2rem 0; background: rgba(180, 180, 180, 0.15); border-radius: 3px;">
                        {emoji} <strong>{player['name']}</strong><br>
                        <span style="color: #666; font-size: 0.8rem;">{player['team']} • {format_sar(player['price'])}</span>
                    </div>
                    """, unsafe_allow_html=True)

                if st.button("🗑️ Clear Team", type="secondary"):
                    st.session_state.fantasy_team = []
                    st.session_state.budget = 100_000_000
                    st.rerun()
            else:
                st.info("No players selected yet. Start building your dream team!")

    # === TAB 2: Match Day ===
    with fantasy_tabs[1]:
        st.markdown("### 🎮 Match Day Simulation")

        if len(st.session_state.fantasy_team) == 11:
            # Load players data for opponent team
            players_df = load_players_data()
            
            # Create player lookup
            player_lookup = {p["name"]: p for p in st.session_state.fantasy_team}
            
            # Add all other players to lookup for opponent selection
            for _, player in players_df.iterrows():
                if player["name"] not in player_lookup:
                    player_lookup[player["name"]] = {
                        "name": player["name"],
                        "overall": player.get("overall", 60),
                        "position": player.get("position", "-"),
                        "team": player.get("team", "-"),
                        "nationality": player.get("nationality", "-"),
                        "age": player.get("age", "-")
                    }

            # Function to display player in match day format
            def display_player_minimal(player, index):
                name = player.get("name", "Unknown")
                ovr = player.get("overall", 60)
                
                if ovr == "-" or ovr is None:
                    ovr = 60
                
                kit_number = f"{index:02}"

                html = f"""
                <div style="font-family: monospace; font-size: 15px; margin-bottom: 6px; color: white; background: rgba(0,0,0,0.1); padding: 5px; border-radius: 3px;">
                    [<span style="width: 2ch; display: inline-block;">{kit_number}</span>] 
                    <span style="width: 24ch; display: inline-block;">{name}</span> 
                    (<span style="width: 2ch; display: inline-block;">{ovr}</span>)
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

            st.subheader("🏟️ Match Day: Your XI vs Opponent XI")
            
            col1, col2 = st.columns(2)

            # User XI
            with col1:
                st.markdown("### 🟢 Your Starting XI")
                
                for i, player in enumerate(st.session_state.fantasy_team, start=1):
                    display_player_minimal(player, i)

            # Opponent XI
            with col2:
                st.markdown("### 🔴 Opponent XI")

                # Get names of user's selected players to avoid duplicates
                user_player_names = {p["name"] for p in st.session_state.fantasy_team}
                
                # Create opponent pool
                available_opponents = []
                for player in player_lookup.values():
                    if player["name"] not in user_player_names:
                        available_opponents.append(player)
                
                # Generate or retrieve opponent XI
                if "opponent_xi" not in st.session_state:
                    import random
                    opponent_xi = random.sample(available_opponents, k=min(11, len(available_opponents)))
                    st.session_state.opponent_xi = opponent_xi
                else:
                    opponent_xi = st.session_state.opponent_xi

                for i, player in enumerate(opponent_xi, start=1):
                    display_player_minimal(player, i)
                    
            # Team statistics
            user_xi = st.session_state.fantasy_team
            opponent_xi = st.session_state.get("opponent_xi", [])
            
            st.divider()
            st.subheader("📊 Team Statistics")
            
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                if len(user_xi) == 11:
                    user_ratings = []
                    for player in user_xi:
                        rating = player.get("overall", 60)
                        try:
                            rating = float(rating) if rating != "-" else 60.0
                        except (ValueError, TypeError):
                            rating = 60.0
                        user_ratings.append(rating)

                    if user_ratings:
                        user_avg = sum(user_ratings) / len(user_ratings)
                        st.markdown(f"""
                            <div style="text-align: center;">
                                <h1 style="color: #16C60C; font-size: 48px; margin-bottom: 0;">{user_avg:.1f}</h1>
                                <p style="font-size: 18px; color: white;">Team Overall Rating</p>
                                <p style="font-size: 14px; color: white;">
                                    Highest: {max(user_ratings):.1f} | 
                                    Lowest: {min(user_ratings):.1f} | 
                                    Spread: {(max(user_ratings) - min(user_ratings)):.1f}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
            
            with col_stats2:
                if opponent_xi:
                    opponent_ratings = []
                    for player in opponent_xi:
                        rating = player.get("overall", 60)
                        try:
                            rating = float(rating) if rating != "-" else 60.0
                        except (ValueError, TypeError):
                            rating = 60.0
                        opponent_ratings.append(rating)

                    if opponent_ratings:
                        opponent_avg = sum(opponent_ratings) / len(opponent_ratings)
                        st.markdown(f"""
                            <div style="text-align: center;">
                                <h1 style="color: red; font-size: 48px; margin-bottom: 0;">{opponent_avg:.1f}</h1>
                                <p style="font-size: 18px; color: white;">Team Overall Rating</p>
                                <p style="font-size: 14px; color: white;">
                                    Highest: {max(opponent_ratings):.1f} | 
                                    Lowest: {min(opponent_ratings):.1f} | 
                                    Spread: {(max(opponent_ratings) - min(opponent_ratings)):.1f}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)

            # Match prediction
            if len(user_xi) == 11 and opponent_xi:
                st.divider()
                st.subheader("⚔️ Match Prediction")
                
                user_ratings = []
                opponent_ratings = []
                
                # Calculate averages
                for player in user_xi:
                    rating = player.get("overall", 60)
                    try:
                        rating = float(rating) if rating != "-" else 60.0
                    except (ValueError, TypeError):
                        rating = 60.0
                    user_ratings.append(rating)
                
                for player in opponent_xi:
                    rating = player.get("overall", 60)
                    try:
                        rating = float(rating) if rating != "-" else 60.0
                    except (ValueError, TypeError):
                        rating = 60.0
                    opponent_ratings.append(rating)
                
                if user_ratings and opponent_ratings:
                    user_avg = sum(user_ratings) / len(user_ratings)
                    opponent_avg = sum(opponent_ratings) / len(opponent_ratings)
                    
                    difference = user_avg - opponent_avg
                    
                    if abs(difference) < 2:
                        prediction = "🤝 Evenly Matched! This should be a close game."
                        color = "orange"
                    elif difference > 0:
                        prediction = f"🟢 Your Team Favoured! You have a {difference:.1f} point advantage."
                        color = "green"
                    else:
                        prediction = f"🔴 Opponent Favoured! They have a {abs(difference):.1f} point advantage."
                        color = "red"
                    
                    st.markdown(f"<div style='color: {color}; font-size: 16px; text-align: center; padding: 10px; border: 2px solid {color}; border-radius: 10px;'>{prediction}</div>", unsafe_allow_html=True)

                    # Simulate Match button
                    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([5, 2, 1])
                    with col2:
                        username = st.session_state.get("username", None)
                        simulate_match_with_leaderboard(user_xi, opponent_xi, username)

            
            # Regenerate opponent team functionality
            if "opponent_regeneration_count" not in st.session_state:
                st.session_state.opponent_regeneration_count = 0
            
            max_regenerations = 3
            remaining_regenerations = max_regenerations - st.session_state.opponent_regeneration_count
            
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

            if remaining_regenerations > 0:
                col1, col2, col3 = st.columns([6, 1, 1])
                with col3:
                    button_text = f"🔄 Generate New Opponent XI ({remaining_regenerations} left)"
                    if st.button(button_text):
                        import random
                        available_opponents = [
                            player for player in player_lookup.values()
                            if player["name"] not in user_player_names
                        ]
                        
                        if len(available_opponents) >= 11:
                            st.session_state.opponent_xi = random.sample(available_opponents, k=11)
                        else:
                            st.session_state.opponent_xi = random.sample(list(player_lookup.values()), k=11)
                        
                        st.session_state.opponent_regeneration_count += 1
                        st.rerun()
            else:
                st.info("🚫 Maximum opponent regenerations reached (3/3). Opponent XI is locked for this session.")
                
                col1, col2, col3 = st.columns([6, 1, 1])
                with col3:
                    if st.button("🔓 Reset Regeneration Counter"):
                        st.session_state.opponent_regeneration_count = 0
                        st.rerun()

            # Display leaderboard
            display_leaderboard()

        else:
            st.warning("⚠️ Please select 11 players in the Fantasy Team tab first!")



# ================================================
#   Statistics tab
# ================================================

with tabs[3]:  # Statistics Tab
    st.markdown("## 📊 League Statistics")
    
    # Load data with error handling
    try:
        teams_df = load_teams_data()
        top_scorers_df = load_top_scorers()
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        st.stop()
        
    # Check if data is available
    if teams_df.empty and players_df.empty:
        st.warning("📭 No statistics data available. Please ensure data files are loaded properly.")
        st.stop()
    
    # Team Statistics Section
    st.markdown("### 🏆 Team Performance")
    
    if not teams_df.empty:
        # Team selection filters
        col_team_filter, col_count, col_metric = st.columns([2, 1, 1])

        with col_team_filter:
            if 'name' in teams_df.columns:
                all_teams = ["All Teams"] + sorted(teams_df['name'].unique().tolist())
                selected_teams = st.multiselect(
                    "Filter by specific teams:",
                    options=all_teams,
                    default=["All Teams"],
                    key="team_filter_select"
                )
            else:
                selected_teams = ["All Teams"]

        with col_count:
            team_count = st.selectbox(
                "Number of teams:",
                options=[5, 10, 15, 20, "All"],
                index=1,
                key="team_count_select"
            )

        # Apply team filter
        if "All Teams" not in selected_teams and selected_teams:
            filtered_teams = teams_df[teams_df['name'].isin(selected_teams)]
        else:
            filtered_teams = teams_df

        # Compute goal difference BEFORE slicing or visualizing
        if 'goals_for' in filtered_teams.columns and 'goals_against' in filtered_teams.columns:
            filtered_teams['goal_difference'] = filtered_teams['goals_for'] - filtered_teams['goals_against']

        # Determine number of teams to show
        if team_count == "All":
            display_teams = filtered_teams
        else:
            display_teams = filtered_teams.head(int(team_count))
        
        # Select columns to display
        available_columns = teams_df.columns.tolist()
        default_columns = ['name', 'points', 'goals_for', 'goals_against']
        display_columns = [col for col in default_columns if col in available_columns]

        if len(display_columns) < 2:
            display_columns = available_columns[:min(6, len(available_columns))]

        # Add goal difference if both goals columns exist
        if 'goals_for' in teams_df.columns and 'goals_against' in teams_df.columns:
            teams_df['goal_difference'] = teams_df['goals_for'] - teams_df['goals_against']
            if 'goal_difference' not in display_columns:
                display_columns.append('goal_difference')

        
        # Sort by points if available, otherwise by first numeric column
        if 'points' in teams_df.columns:
            sorted_teams = teams_df.sort_values('points', ascending=False)
        else:
            numeric_cols = teams_df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                sorted_teams = teams_df.sort_values(numeric_cols[0], ascending=False)
            else:
                sorted_teams = teams_df
        
        # Display league table with ranking
        sorted_teams = sorted_teams.reset_index(drop=True)
        sorted_teams.index = sorted_teams.index + 1
        sorted_teams.index.name = "Rank"
        
        st.dataframe(
            sorted_teams[display_columns].head(int(team_count) if team_count != "All" else None),
            use_container_width=True,
            height=400
        )
    
    else:
        st.warning("⚠️ No team data available for statistics.")
    
    # Player Statistics Section
    st.markdown("### ⚽ Player Performance")
    
    if not top_scorers_df.empty:
        # Player statistics controls
        col_team_filter, col_metric, col_count = st.columns([2, 1, 1])
        
        with col_team_filter:
            if 'team' in top_scorers_df.columns:
                all_player_teams = ["All Teams"] + sorted(top_scorers_df['team'].unique().tolist())
                selected_player_teams = st.multiselect(
                    "Filter players by team:",
                    options=all_player_teams,
                    default=["All Teams"],
                    key="player_team_filter_select"
                )
            else:
                selected_player_teams = ["All Teams"]
        
        with col_metric:
            metric_options = ["Goals", "Assists", "Appearances", "Minutes Played"]
            available_metrics = []
            
            if 'goals' in top_scorers_df.columns:
                available_metrics.append("Goals")
            if 'assists' in top_scorers_df.columns:
                available_metrics.append("Assists")
            if 'appearances' in top_scorers_df.columns:
                available_metrics.append("Appearances")
            if 'minutes_played' in top_scorers_df.columns:
                available_metrics.append("Minutes Played")
            
            if available_metrics:
                selected_metric = st.selectbox(
                    "Select metric to highlight:",
                    options=available_metrics,
                    index=0,
                    key="player_metric_select"
                )
            else:
                selected_metric = "Goals"
        
        with col_count:
            player_count = st.selectbox(
                "Players to display:",
                options=[10, 20, 30, 50, "All"],
                index=1,
                key="player_count_select"
            )
        
        # Top players chart
        if available_metrics:
            metric_column = selected_metric.lower().replace(' ', '_')
            
            # Apply team filter for players
            if "All Teams" not in selected_player_teams and selected_player_teams and 'team' in top_scorers_df.columns:
                filtered_players = top_scorers_df[top_scorers_df['team'].isin(selected_player_teams)]
            else:
                filtered_players = top_scorers_df
            
            display_players = filtered_players.head(int(player_count) if player_count != "All" else None)
            
            fig_players = px.bar(
                display_players.sort_values(metric_column, ascending=False).head(15),
                x='name',
                y=metric_column,
                title=f'🏆 Top Players - {selected_metric}',
                color=metric_column,
                color_continuous_scale='plasma',
                labels={metric_column: selected_metric, 'name': 'Player'},
                text=metric_column
            )
            fig_players.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family='Inter',
                title_font_size=16,
                xaxis_tickangle=-45,
                height=500,
                showlegend=False,
                yaxis_visible=False,
                xaxis_showgrid=False,
                yaxis_showgrid=False
            )
            fig_players.update_traces(
                hovertemplate='<b>%{x}</b><br>' + selected_metric + ': %{y}<extra></extra>',
                texttemplate='%{text}',
                textposition='outside'
            )
            st.plotly_chart(fig_players, use_container_width=True)
        
        # Player statistics table
        st.markdown("### 📊 Player Statistics Table")
        
        # Select available columns for display
        player_columns = ["name", "team", "goals", "assists", "appearances", "minutes_played"]
        available_player_columns = [col for col in player_columns if col in top_scorers_df.columns]
        
        if not available_player_columns:
            available_player_columns = top_scorers_df.columns.tolist()[:6]
        
        # Add calculated metrics if possible
        if 'goals' in top_scorers_df.columns and 'appearances' in top_scorers_df.columns:
            top_scorers_df['goals_per_game'] = (top_scorers_df['goals'] / top_scorers_df['appearances']).round(2)
            available_player_columns.append('goals_per_game')
        
        if 'minutes_played' in top_scorers_df.columns and 'goals' in top_scorers_df.columns:
            top_scorers_df['goals_per_90'] = ((top_scorers_df['goals'] / top_scorers_df['minutes_played']) * 90).round(2)
            available_player_columns.append('goals_per_90')
        
        # Sort players by selected metric
        if available_metrics and selected_metric.lower().replace(' ', '_') in top_scorers_df.columns:
            # Apply team filter before sorting
            if "All Teams" not in selected_player_teams and selected_player_teams and 'team' in top_scorers_df.columns:
                sorted_players = top_scorers_df[top_scorers_df['team'].isin(selected_player_teams)].sort_values(
                    selected_metric.lower().replace(' ', '_'), 
                    ascending=False
                )
            else:
                sorted_players = top_scorers_df.sort_values(
                    selected_metric.lower().replace(' ', '_'), 
                    ascending=False
                )
        else:
            # Apply team filter without sorting
            if "All Teams" not in selected_player_teams and selected_player_teams and 'team' in top_scorers_df.columns:
                sorted_players = top_scorers_df[top_scorers_df['team'].isin(selected_player_teams)]
            else:
                sorted_players = top_scorers_df
        
        # Display player table
        display_count = int(player_count) if player_count != "All" else len(sorted_players)
        
        st.dataframe(
            sorted_players[available_player_columns].head(display_count),
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Summary statistics
        if len(available_metrics) > 1:
            st.markdown("### 📈 Summary Statistics")
            
            summary_cols = st.columns(len(available_metrics))
            
            for i, metric in enumerate(available_metrics):
                with summary_cols[i]:
                    metric_col = metric.lower().replace(' ', '_')
                    if metric_col in top_scorers_df.columns:
                        total = top_scorers_df[metric_col].sum()
                        avg = top_scorers_df[metric_col].mean()
                        top_player = top_scorers_df.loc[top_scorers_df[metric_col].idxmax(), 'name']
                        top_value = top_scorers_df[metric_col].max()
                        
                        st.metric(
                            label=f"Total {metric}",
                            value=f"{total:,.0f}",
                            delta=f"Avg: {avg:.1f}"
                        )
                        st.write(f"**Top:** {top_player} ({top_value})")
    
    else:
        st.warning("⚠️ No player data available for statistics.")


# ================================================
#   Fixtures tab
# ================================================

with tabs[4]:  # Fixtures Tab
    st.markdown("## 📅 Fixtures & Results")
    
    # Enhanced fixtures data with more details
    fixtures = [
        {"date": "2025-08-27", "time": "20:00", "home": "Al Fayha", "away": "Al Shabab", "venue": "Al Majma'ah Stadium", "matchday": 5, "status": "upcoming", "home_form": "WLDLL", "away_form": "DWWLW", "prediction": ""},
        {"date": "2025-08-28", "time": "21:00", "home": "Al Riyadh", "away": "Damac", "venue": "Prince Faisal bin Fahd Stadium", "matchday": 5, "status": "upcoming", "home_form": "LLWDL", "away_form": "WDLWL", "prediction": ""},
        # Recent results
        {"date": "2025-08-17", "time": "19:00", "home": "Al Nassr", "away": "Al Hilal", "venue": "Mrsool Park", "matchday": 4, "status": "completed", "home_score": 2, "away_score": 3, "home_form": "WLWWW", "away_form": "WWDWL"},
        {"date": "2025-08-18", "time": "20:30", "home": "Al Ahli", "away": "Al Ittihad", "venue": "King Abdullah Stadium", "matchday": 4, "status": "completed", "home_score": 1, "away_score": 1, "home_form": "WWLWL", "away_form": "LWWDW"},
        {"date": "2025-08-19", "time": "18:00", "home": "Al Shabab", "away": "Al Ettifaq", "venue": "Al Shabab Stadium", "matchday": 4, "status": "completed", "home_score": 0, "away_score": 2, "home_form": "DWWLW", "away_form": "DLWWL"},
    ]
    
    # Convert to DataFrame for easier filtering
    import pandas as pd
    from datetime import datetime, timedelta
    
    fixtures_df = pd.DataFrame(fixtures)
    fixtures_df['date'] = pd.to_datetime(fixtures_df['date'])
    
    # Filter and view controls - ADD THE 4TH COLUMN FOR PREDICTIONS
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        view_filter = st.selectbox(
            "📊 View:",
            options=["All Fixtures", "Upcoming Only", "Recent Results", "This Week", "Next Week"],
            index=0,
            key="fixture_view_filter"
        )
    
    with col2:
        team_filter = st.selectbox(
            "🏆 Team:",
            options=["All Teams"] + sorted(list(set(fixtures_df['home'].tolist() + fixtures_df['away'].tolist()))),
            index=0,
            key="fixture_team_filter"
        )
    
    with col3:
        sort_option = st.selectbox(
            "📅 Sort by:",
            options=["Date (Newest First)", "Date (Oldest First)", "Matchday"],
            index=0,
            key="fixture_sort_option"
        )
    
    # NEW: Add AI predictions toggle
    with col4:
        show_ai_predictions = st.checkbox("🔮 AI Predictions", value=True, key="show_ai_predictions")
    
    # Apply filters
    filtered_fixtures = fixtures_df.copy()
    
    # Date-based filtering
    today = datetime.now()
    if view_filter == "Upcoming Only":
        filtered_fixtures = filtered_fixtures[filtered_fixtures['status'] == 'upcoming']
    elif view_filter == "Recent Results":
        filtered_fixtures = filtered_fixtures[filtered_fixtures['status'] == 'completed']
    elif view_filter == "This Week":
        week_start = today - timedelta(days=today.weekday())
        week_end = week_start + timedelta(days=6)
        filtered_fixtures = filtered_fixtures[
            (filtered_fixtures['date'] >= week_start) & 
            (filtered_fixtures['date'] <= week_end)
        ]
    elif view_filter == "Next Week":
        next_week_start = today + timedelta(days=(7 - today.weekday()))
        next_week_end = next_week_start + timedelta(days=6)
        filtered_fixtures = filtered_fixtures[
            (filtered_fixtures['date'] >= next_week_start) & 
            (filtered_fixtures['date'] <= next_week_end)
        ]
    
    # Team filtering
    if team_filter != "All Teams":
        filtered_fixtures = filtered_fixtures[
            (filtered_fixtures['home'] == team_filter) | 
            (filtered_fixtures['away'] == team_filter)
        ]
    
    # Apply sorting
    if sort_option == "Date (Newest First)":
        filtered_fixtures = filtered_fixtures.sort_values('date', ascending=False)
    elif sort_option == "Date (Oldest First)":
        filtered_fixtures = filtered_fixtures.sort_values('date', ascending=True)
    elif sort_option == "Matchday":
        filtered_fixtures = filtered_fixtures.sort_values('matchday', ascending=True)

    # Convert scores to whole numbers (safe even if some rows are missing values)
    filtered_fixtures['home_score'] = filtered_fixtures['home_score'].fillna(0).astype(int)
    filtered_fixtures['away_score'] = filtered_fixtures['away_score'].fillna(0).astype(int)

    # Custom CSS for enhanced fixture cards
    st.markdown("""
    <style>
    .fixture-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .fixture-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
    }
    
    .result-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.2);
    }
    
    .team-name {
        font-size: 1.4rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    
    .match-details {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0;
        font-size: 0.95rem;
    }
    
    .match-time {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 0;
    }
    
    .match-date {
        color: rgba(255,255,255,0.8);
        margin: 0;
        font-size: 0.9rem;
    }
    
    .form-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin: 0 2px;
        text-align: center;
        font-size: 12px;
        font-weight: bold;
        line-height: 20px;
    }
    
    .form-w { background: #4CAF50; color: white; }
    .form-d { background: #FF9800; color: white; }
    .form-l { background: #F44336; color: white; }
    
    .prediction-badge {
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .score-display {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin: 0 15px;
    }
    
    .matchday-badge {
        background: rgba(255,255,255,0.2);
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display fixtures
    if len(filtered_fixtures) == 0:
        st.info("🔍 No fixtures found matching your filter criteria.")
    else:
        # Summary statistics
        upcoming_count = len(filtered_fixtures[filtered_fixtures['status'] == 'upcoming'])
        completed_count = len(filtered_fixtures[filtered_fixtures['status'] == 'completed'])
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)

        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 5rem; margin-top: 2rem;">
            <div style="text-align: center;">
                <div style="font-size: 1rem;">🗓️ Total Fixtures</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1rem;">⏳ Upcoming</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1rem;">✅ Completed</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{}</div>
            </div>
        </div>
        """.format(len(filtered_fixtures), upcoming_count, completed_count), unsafe_allow_html=True)

        # Add AI predictions info banner
        if show_ai_predictions and upcoming_count > 0:
            st.info("🤖 AI Match Predictions powered by current league standings, team form, and key player analysis")

        st.markdown("---")
        
        # Function to create form indicators
        def create_form_html(form_string):
            if pd.isna(form_string) or form_string == "":
                return ""
            
            form_html = ""
            for char in form_string:
                if char == 'W':
                    form_html += '<span class="form-indicator form-w">W</span>'
                elif char == 'D':
                    form_html += '<span class="form-indicator form-d">D</span>'
                elif char == 'L':
                    form_html += '<span class="form-indicator form-l">L</span>'
            return form_html
        
        # Display each fixture
        for _, fixture in filtered_fixtures.iterrows():
            fixture_date = fixture['date'].strftime('%a, %b %d, %Y')
            
            if fixture['status'] == 'upcoming':
                # Upcoming fixture card
                prediction_html = f'<span class="prediction-badge">🎯 {fixture.get("prediction", "TBD")}</span>' if 'prediction' in fixture and pd.notna(fixture.get('prediction')) else ''
                
                home_form_html = create_form_html(fixture.get('home_form', ''))
                away_form_html = create_form_html(fixture.get('away_form', ''))
                
                st.markdown(f"""
                <div class="fixture-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <span class="matchday-badge">Matchday {fixture['matchday']}</span>
                        {prediction_html}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div style="flex: 1;">
                            <h3 class="team-name">{fixture['home']}</h3>
                            <div style="margin-top: 5px;">{home_form_html}</div>
                        </div>
                        <div style="text-align: center; padding: 0 20px;">
                            <div style="font-size: 1.5rem; color: white; font-weight: bold;">VS</div>
                            <div style="font-size: 0.9rem; color: rgba(255,255,255,0.8);">{fixture['time']}</div>
                        </div>
                        <div style="flex: 1; text-align: right;">
                            <h3 class="team-name">{fixture['away']}</h3>
                            <div style="margin-top: 5px; text-align: right;">{away_form_html}</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; text-align: center;">
                        <p class="match-details">📍 {fixture['venue']}</p>
                        <p class="match-details">📅 {fixture_date}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # ENHANCED AI PREDICTION SECTION
                if show_ai_predictions:
                    with st.expander(f"🔮 Detailed AI Prediction: {fixture['home']} vs {fixture['away']}", expanded=False):
                        with st.spinner("🤖 Analyzing teams and generating prediction..."):
                            try:
                                # Import the prediction function here to avoid circular imports
                                from agents.match_predictor import get_match_prediction, display_prediction_card
                                
                                prediction = get_match_prediction(fixture['home'], fixture['away'])
                                
                                if prediction:
                                    display_prediction_card(prediction)
                                    
                                    # Add user's team insight if applicable
                                    user_fav_team = st.session_state.get("favorite_team", "")
                                    if user_fav_team in [fixture['home'], fixture['away']]:
                                        if prediction.get('predicted_result') == 'Home Win' and fixture['home'] == user_fav_team:
                                            st.success(f"🎉 Great news for {user_fav_team} fans! The prediction favors your team!")
                                        elif prediction.get('predicted_result') == 'Away Win' and fixture['away'] == user_fav_team:
                                            st.success(f"🎉 Great news for {user_fav_team} fans! The prediction favors your team!")
                                        elif prediction.get('predicted_result') == 'Draw':
                                            st.info(f"🤝 A draw is predicted - {user_fav_team} will need to fight for all 3 points!")
                                        else:
                                            st.warning(f"⚠️ The prediction doesn't favor {user_fav_team}, but anything can happen in football!")
                                else:
                                    st.warning("⚠️ AI prediction temporarily unavailable")
                                    
                            except ImportError:
                                st.error("🚫 Prediction system not available. Please ensure match_predictor.py is in the utils/ folder.")
                            except Exception as e:
                                st.error(f"Error generating prediction: {e}")
                                # Fallback simple prediction based on existing data
                                if 'prediction' in fixture and pd.notna(fixture.get('prediction')):
                                    st.info(f"📊 Simple prediction based on current form: **{fixture['prediction']}**")
                
            else:
                # Completed fixture (result) card
                if fixture['status'] == 'completed':
                    home_score = fixture.get('home_score', 0)
                    away_score = fixture.get('away_score', 0)

                    if pd.notna(home_score) and pd.notna(away_score):
                        if home_score > away_score:
                            result_text = f"{fixture['home']} Won"
                        elif away_score > home_score:
                            result_text = f"{fixture['away']} Won"
                        else:
                            result_text = "Draw"
                    else:
                        result_text = "Result Unknown"

                    home_form_html = create_form_html(fixture.get('home_form', ''))
                    away_form_html = create_form_html(fixture.get('away_form', ''))

                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                            <span class="matchday-badge">Matchday {fixture['matchday']}</span>
                            <span class="prediction-badge">✅ {result_text}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="flex: 1;">
                                <h3 class="team-name">{fixture['home']}</h3>
                                <div style="margin-top: 5px;">{home_form_html}</div>
                            </div>
                            <div style="text-align: center; padding: 0 20px;">
                                <div class="score-display">{home_score} - {away_score}</div>
                                <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8);">FULL TIME</div>
                            </div>
                            <div style="flex: 1; text-align: right;">
                                <h3 class="team-name">{fixture['away']}</h3>
                                <div style="margin-top: 5px; text-align: right;">{away_form_html}</div>
                            </div>
                        </div>
                        <div style="margin-top: 15px; text-align: center;">
                            <p class="match-details">📍 {fixture['venue']}</p>
                            <p class="match-details">🗓️ {fixture_date}</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Additional Information
    st.markdown("### ℹ️ Legend")
    
    col_legend1, col_legend2 = st.columns(2)

    with col_legend1:
        st.markdown("""
        **Card Colors:**  
        - 🔵 **Blue** = Upcoming Fixture  
        - 🟢 **Green** = Completed Result  
        
        **Form Indicators:**  
        - 🟢 **W** = Win  
        - 🟡 **D** = Draw  
        - 🔴 **L** = Loss  
        """)
    
    with col_legend2:
        if show_ai_predictions:
            st.markdown("""
            **AI Predictions:**  
            - Based on current league standings
            - Team form and goal statistics  
            - Key player analysis  
            - Home advantage factors  
            
            *Click "Detailed AI Prediction" for in-depth analysis*
            """)
        else:
            st.markdown("""
            **Enable AI Predictions:**  
            - Toggle "🔮 AI Predictions" above  
            - Get detailed match analysis  
            - View confidence scores  
            - See key factors and players to watch  
            """)

# ================================================
#   Fan Zone tab with 4 Subtabs
# ================================================

with tabs[5]:  # 🧑‍🤝‍🧑 Fan Zone Tab
    st.markdown("""
    <h2 style="text-align: left;">🧑‍🤝‍🧑 Fan Zone</h2>
    <p style="text-align: left; font-size: 1.1rem; margin-top: 0.2rem;">
        Explore interactive fan features like your assistant, avatar, chants, and trivia!
    </p>
    """, unsafe_allow_html=True)

    fan_tabs = st.tabs(["🖼️ Fan Avatar", "🔊 Chant Maker", "🧠 SPL Trivia"])


    # === Tab 1: Fan Avatar Generator ===
    with fan_tabs[0]:
        from utils.img2img_utils import setup_image_pipeline
        from utils.prompt_templates import IMAGE_GENERATION_PROMPT, NEGATIVE_PROMPT
        from PIL import Image
        import torch
        import numpy as np

        st.subheader("Share your SPL avatar with your friends!")
        pipe = setup_image_pipeline()
        uploaded_file = st.file_uploader("Upload your image here (PNG or JPG)", type=["png", "jpg", "jpeg"])
        prompt_suffix = st.text_input("Saudi Pro League Fan Prompt", value="Saudi Pro League jersey")

        if uploaded_file:
            try:
                initial_image = Image.open(uploaded_file).convert("RGB").resize((512, 512))
                st.image(initial_image, caption="Your Uploaded Character Image", use_container_width=True)
                if st.button("Generate Your Avatar"):
                    full_prompt = IMAGE_GENERATION_PROMPT.format(prompt_suffix=prompt_suffix)
                    output = pipe(prompt=full_prompt, image=initial_image, negative_prompt=NEGATIVE_PROMPT, output_type="pil")
                    generated_image = output[0]
                    st.success("Image generated successfully!")
                    st.image(generated_image, caption="Generated SPL Avatar", use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

    # === Tab 3: Chant Generator ===
    with fan_tabs[1]:
        from utils.chants_utils import setup_chant_chain
        from utils.prompt_templates import CHANT_PROMPT

        st.subheader("Generate a Football Chant! 📣")
        chant_chain = setup_chant_chain(chant_prompt_template_str=CHANT_PROMPT)
        user_input_chant = st.text_input("Enter a team name or theme:")

        if st.button("Generate Chant"):
            if user_input_chant:
                with st.spinner("Crafting your chant..."):
                    try:
                        chant = chant_chain.invoke(user_input_chant)
                        st.markdown(f"```\n{chant}\n```")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # === Tab 4: Trivia Game ===
    with fan_tabs[2]:
        from utils.trivia_utils import generate_trivia_question_from_fact, validate_answer

        st.subheader("SPL Trivia Game! 🧠")

        if "current_question" not in st.session_state:
            st.session_state.current_question = None
        if "score" not in st.session_state:
            st.session_state.score = 0
        if "answered" not in st.session_state:
            st.session_state.answered = False
        if "selected_option" not in st.session_state:
            st.session_state.selected_option = None
        if "feedback" not in st.session_state:
            st.session_state.feedback = ""
        if "total_questions" not in st.session_state:
            st.session_state.total_questions = 0

        def start_new_question():
            st.session_state.current_question = generate_trivia_question_from_fact()
            st.session_state.answered = False
            st.session_state.selected_option = None
            st.session_state.feedback = ""
            st.session_state.total_questions += 1

        def reset_trivia_game():
            st.session_state.current_question = None
            st.session_state.score = 0
            st.session_state.answered = False
            st.session_state.selected_option = None
            st.session_state.feedback = ""
            st.session_state.total_questions = 0

        if st.session_state.current_question is None:
            if st.button("Start New Trivia Game"):
                start_new_question()
                st.rerun()

        if st.session_state.current_question:
            q = st.session_state.current_question
            st.write(f"**Question:** {q['question']}")

            user_selection = st.radio("Choose your answer:", q['options'], index=0, disabled=st.session_state.answered)

            if not st.session_state.answered and st.button("Submit Answer"):
                st.session_state.answered = True
                selected_index = q['options'].index(user_selection)
                st.session_state.selected_option = selected_index
                if validate_answer(selected_index, q['correct_answer_index']):
                    st.session_state.feedback = "✅ Correct!"
                    st.session_state.score += 1
                else:
                    correct = q['options'][q['correct_answer_index']]
                    st.session_state.feedback = f"❌ Incorrect. Correct answer: **{correct}**"
                st.rerun()

            if st.session_state.answered:
                st.markdown(st.session_state.feedback)
                st.write(f"Current Score: {st.session_state.score} / {st.session_state.total_questions}")
                if st.button("Next Question"):
                    start_new_question()
                    st.rerun()
                if st.button("Restart Game"):
                    reset_trivia_game()
                    st.rerun()






# ================================================
#   Chatbot
# ================================================

DATA_FOLDER = "data"

if "team_logo" not in st.session_state:
    st.session_state.team_logo = "https://cdn-icons-png.flaticon.com/512/847/847969.png"

user_team_logo = st.session_state["team_logo"]


def generate_column_guide(df):
    guide = "Column Reference (from combined JSON data):\n\n"
    grouped = {}

    for col in df.columns:
        if col.startswith("__"):
            continue
        prefix, field = col.split(".", 1) if "." in col else ("misc", col)
        grouped.setdefault(prefix, []).append(field)

    for source, fields in grouped.items():
        guide += f"\U0001F4C4 {source}.json\n"
        for field in fields:
            guide += f"- {source}.{field}\n"
        guide += "\n"

    return guide.strip()

@st.cache_resource
def create_agent_from_jsons():
    df = load_all_jsons()
    if df.empty:
        st.stop()
    temp_csv = "temp_combined_data.csv"
    df.to_csv(temp_csv, index=False)
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4"),
        temp_csv,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )
    os.remove(temp_csv)  # Clean up after agent is built
    st.session_state.json_qa_df = df  # Store for later use if needed
    return agent

players_df = load_players_data()

# ==== Streamlit UI ====
st.title("🤖 Your Saudi Pro League Sports Analyst")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

agent = create_agent_from_jsons()
df = st.session_state.get("json_qa_df")
column_guide = generate_column_guide(df) if df is not None else ""

# Replace this section in your main.py (around line 580-590)

# ===== Chat Display =====
for role, content in st.session_state.chat_history:
    if role == "user":
        rendered = user_template.replace("{MSG}", content).replace("{USER_AVATAR}", user_team_logo)
    else:
        rendered = bot_template.replace("{MSG}", content)
    st.write(rendered, unsafe_allow_html=True)

# ===== Chat Input =====
user_input = st.chat_input("Ask your analyst something...")
if user_input:
    username = st.session_state.get("username", "Guest")
    fav_team = st.session_state.get("favorite_team", "SPL")
    
    st.session_state.chat_history.append(("user", user_input))

    try:
        # Enhanced prompt with user context
        context_prompt = f"""
        You are a Saudi Pro League sports analyst assistant. 
        The user's name is {username} and they support {fav_team}.
        Address them by name and reference their team when relevant.
        
        {column_guide}
        
        User Question: {user_input}
        """
        answer = agent.run(context_prompt)
    except Exception as e:
        answer = f"⚠️ Sorry {username}, I ran into an error: {str(e)}"

    st.session_state.chat_history.append(("assistant", answer))

    # Render the conversation
    rendered_user = user_template.replace("{MSG}", user_input).replace("{USER_AVATAR}", user_team_logo)
    rendered_bot = bot_template.replace("{MSG}", answer)
    st.write(rendered_user, unsafe_allow_html=True)
    st.write(rendered_bot, unsafe_allow_html=True)






























































# Footer
st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid #e5e7eb;">
        <p style="color: #666; margin: 0;">© 2025 Saudi Pro League Hub | Brought to you by Accenture Song</p>
    </div>
    """, unsafe_allow_html=True)

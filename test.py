# ================================================
#  Enhanced Saudi Pro League Hub - Integrated Version
# ================================================

import os
import base64
import textwrap
import json
import streamlit as st
from datetime import datetime
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from dotenv import load_dotenv
import time

# Import existing modules
from agents.flatten_json import load_all_jsons
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from htmlTemplates import user_template, bot_template, css
from agents.flags import NATIONALITY_FLAGS
from agents.controlled_simulator import simulate_match

# Import new utility modules (you'll need to create these)
try:
    from utils.rag_utils import setup_rag_components
    from utils.img2img_utils import setup_image_pipeline
    from utils.prompt_templates import RAG_PROMPT, IMAGE_GENERATION_PROMPT, NEGATIVE_PROMPT, CHANT_PROMPT
    from utils.chants_utils import setup_chant_chain
    from utils.trivia_utils import generate_trivia_question_from_fact, validate_answer
except ImportError:
    st.warning("⚠️ Some utility modules are missing. RAG, Image Generation, and Trivia features will be disabled.")
    RAG_AVAILABLE = False
else:
    RAG_AVAILABLE = True

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Saudi Pro League Hub",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================
#  Background and Styling Functions
# ================================================

def get_base64_of_bin_file(file_path):
    """Function to read png file and convert it to base64 string"""
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

def set_background_image(image_path):
    """Function to set background image for white boxes/cards in Streamlit app"""
    if not os.path.exists(image_path):
        st.warning(f"Background image not found at: {image_path}")
        return
    
    bin_str = get_base64_of_bin_file(image_path)
    if bin_str is None:
        return
    
    page_bg_img = f'''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {{
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
    }}
    
    .main {{
        font-family: 'Inter', sans-serif;
        padding: 2rem;
        margin: 1rem;
    }}
    
    .header-container {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-blend-mode: overlay;
        background-color: rgba(30, 60, 114, 0.8);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }}
    
    .metric-card {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.25);
    }}
    
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def get_flag(nationality):
    return NATIONALITY_FLAGS.get(nationality, "🏳️")

# ================================================
#  Data Loading Functions
# ================================================

@st.cache_data
def load_teams_data():
    """Load teams data from JSON files"""
    try:
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
    except Exception as e:
        st.error(f"Error loading teams data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_players_data():
    """Load players data from JSON files"""
    try:
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
                "price": round(player.get("Market value (€)", 1000000) * 4.07, -3),
                "overall": player.get("Overall", 60),
                "goals": player.get("Goals", 0),
                "assists": player.get("Assists", 0)
            })

        return pd.DataFrame(players)
    except Exception as e:
        st.error(f"Error loading players data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_standings_data():
    """Load standings data with team logos"""
    try:
        with open("data/standings.json", "r", encoding="utf-8") as f:
            standings_data = json.load(f)

        with open("data/teams.json", "r", encoding="utf-8") as f:
            teams_data = json.load(f)

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
                "Logo": team_logo_map.get(team_name, ""),
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
    except Exception as e:
        st.error(f"Error loading standings data: {e}")
        return pd.DataFrame()

@st.cache_data
def load_top_scorers():
    """Load top scorers data"""
    try:
        with open("data/top_scorers.json", "r", encoding="utf-8") as f:
            raw_data = json.load(f)

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
    except Exception as e:
        st.error(f"Error loading top scorers data: {e}")
        return pd.DataFrame()

def format_sar(price):
    """Format price in Saudi Riyals"""
    if price >= 1_000_000:
        return f"SAR {round(price / 1_000_000)}M"
    elif price >= 1_000:
        return f"SAR {round(price / 1_000)}K"
    else:
        return f"SAR {int(price)}"

# ================================================
#  Sidebar Function
# ================================================

def create_professional_sidebar():
    """Creates a clean sidebar with navigation and branding"""
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

    .footer-text {
        font-size: 13px;
        color: white;
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid white;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)

        # Search bar
        st.text_input("Search", placeholder="Search this site...", key="sidebar_search")

        # Nav links
        st.markdown('<a href="#" class="nav-link">Contact Us</a>', unsafe_allow_html=True)
        st.markdown('<a href="#" class="nav-link">FAQ</a>', unsafe_allow_html=True)
        st.markdown('<a href="#" class="nav-link">About</a>', unsafe_allow_html=True)
        st.markdown('<a href="#" class="nav-link">Privacy</a>', unsafe_allow_html=True)
        st.markdown('<a href="#" class="nav-link">Terms</a>', unsafe_allow_html=True)

        # Language switch
        st.markdown("""
        <div style='text-align: center; margin: 40px 0;'>
            <button style='margin: 0 5px; padding: 8px 16px; background: transparent; color: white; border: 1px solid white; border-radius: 5px;'>English</button>
            <button style='margin: 0 5px; padding: 8px 16px; background: transparent; color: white; border: 1px solid white; border-radius: 5px;'>العربية</button>
        </div>
        """, unsafe_allow_html=True)

        current_year = datetime.now().year
        st.markdown(f"""
        <div class="footer-text">
            <strong>SPL Hub by Accenture Song</strong><br>
            © {current_year} All rights reserved.
        </div>
        """, unsafe_allow_html=True)

# ================================================
#  Main Application
# ================================================

def main():
    # Initialize sidebar
    create_professional_sidebar()
    
    # Header with logos
    header_image = "https://media.licdn.com/dms/image/v2/D4D12AQEAKW-BkO6dkQ/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1689759291697?e=2147483647&v=beta&t=g4ujbwKF0bkpbX_JobYfuDacN2IF8Q7eoS_SGMuBtyA"
    spl_logo = "https://play-lh.googleusercontent.com/DFbRC2VtMIy3FXGo7QxEdpx225mjoHM8uF0ct3faUMWDT1rIMlYFhJXRupG5QzkV-Q=w600-h300-pc0xffffff-pd"
    acc_song = "https://www.proi.com/uploaded/companies/logos/17017977352575(1).jpg"

    # Header
    st.markdown(f"""
    <div style="
        width: 100%;
        height: 160px;
        background-image: url('{header_image}');
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
        margin-bottom: 2rem;
    ">
        <div style="flex: 1;">
            <div style="font-size: 2rem; font-weight: 500; text-shadow: 1px 1px 3px black;">
                Your Ultimate Destination for Saudi Professional Football
            </div>
            <div style="font-size: 1.1rem; font-weight: 400; opacity: 0.85; margin-top: 0.4rem; text-shadow: 1px 1px 2px black;">
                Love the League. Live the Game.
            </div>
        </div>
        <div style="flex: 0 0 auto;">
            <img src="{spl_logo}" alt="SPL Logo" style="height: 120px;" />
        </div>
        <div style="position: absolute; top: 8px; right: 20px; font-size: 1.2rem; display: flex; align-items: center; gap: 0.4rem;">
            Powered by <img src="{acc_song}" alt="Accenture Song Logo" style="height: 30px;" />
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary Cards
    display_summary_cards()

    # Main Navigation Tabs
    tabs = st.tabs([
        "🏠 Home", 
        "🏆 League Table", 
        "⚽ Fantasy Football", 
        "📊 Statistics", 
        "📅 Fixtures",
        "🎮 Fan Zone"
    ])

    # Tab Contents
    with tabs[0]:  # Home
        display_home_tab()
    
    with tabs[1]:  # League Table
        display_league_table_tab()
    
    with tabs[2]:  # Fantasy Football
        display_fantasy_football_tab()
    
    with tabs[3]:  # Statistics
        display_statistics_tab()
    
    with tabs[4]:  # Fixtures
        display_fixtures_tab()
    
    with tabs[5]:  # Fan Zone
        display_fan_zone_tab()

    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid #e5e7eb;">
        <p style="color: #666; margin: 0;">© 2025 Saudi Pro League Hub | Brought to you by Accenture Song</p>
    </div>
    """, unsafe_allow_html=True)

def display_summary_cards():
    """Display the summary metric cards"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%); border-radius: 12px; padding: 1.5rem; color: white; text-align: center;">
            <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Total SPL Fanbase</div>
            <div style="font-size: 2rem; font-weight: 700;">28.5K</div>
            <div style="font-size: 0.85rem; color: #a8d8ea;">↗ +2.1K</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dc3545 0%, #ac2937 100%); border-radius: 12px; padding: 1.5rem; color: white; text-align: center;">
            <div style="font-size: 0.85rem; font-weight: 700; margin-bottom: 0.6rem;">🔴 LIVE MATCH</div>
            <div style="font-size: 1rem;">Al Ittihad 1 - 0 Al Hilal</div>
            <div style="font-size: 0.8rem; margin-top: 0.5rem;">75' - Kingdom Arena</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%); border-radius: 12px; padding: 1.5rem; color: white; text-align: center;">
            <div style="font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">Active SPL Fanbase</div>
            <div style="font-size: 2rem; font-weight: 700;">13.4K</div>
            <div style="font-size: 0.85rem; color: #a8d8ea;">↗ +203</div>
        </div>
        """, unsafe_allow_html=True)

def display_fan_zone_tab():
    """Display the Fan Zone with sub-tabs for all AI features"""
    st.markdown("## 🎮 Fan Zone")
    st.markdown("### Experience the ultimate SPL fan features!")
    
    # Fan Zone Sub-tabs
    fan_tabs = st.tabs([
        "🤖 AI Assistant", 
        "🎨 Avatar Generator", 
        "📣 Chant Generator", 
        "🧠 Trivia"
    ])
    
    with fan_tabs[0]:  # AI Assistant
        display_ai_assistant_tab()
    
    with fan_tabs[1]:  # Avatar Generator
        display_avatar_generator_tab()
    
    with fan_tabs[2]:  # Chant Generator
        display_chant_generator_tab()
    
    with fan_tabs[3]:  # Trivia
        display_trivia_tab()

def display_home_tab():
    """Display the home tab content"""
    st.markdown("## 📰 News & Headlines")
    
    # Featured Article
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(42, 82, 152, 0.9) 0%, rgba(30, 60, 114, 0.9) 100%); 
                border-radius: 15px; padding: 2rem; margin: 1.5rem 0; color: white;">
        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
            <span style="background: rgba(255,255,255,0.2); padding: 0.3rem 1rem; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                BREAKING NEWS
            </span>
            <span style="font-size: 0.9rem; opacity: 0.8;">30 minutes ago</span>
        </div>
        <h2 style="margin: 0 0 1rem 0; font-size: 2rem; font-weight: 700;">
            Al Hilal Secure Top Spot with Stunning 4-0 Victory
        </h2>
        <p style="font-size: 1.1rem; line-height: 1.6; margin: 0; opacity: 0.9;">
            Jorge Jesus's side dominated Al Shabab in a commanding display at Kingdom Arena,
            extending their lead at the summit of the Saudi Pro League table.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Latest Articles Grid
    col1, col2, col3 = st.columns(3)
    
    articles = [
        {
            "title": "Ronaldo's Free-Kick Masterclass Seals Derby Victory",
            "summary": "The Portuguese superstar's 89th-minute stunner secured all three points for Al Nassr in the Riyadh Derby thriller.",
            "time": "2 hours ago",
            "category": "Match Report"
        },
        {
            "title": "Benzema Returns to Training Ahead of Crucial Fixture", 
            "summary": "Al Ittihad's French striker is back in full training and ready to face Al Ahli in this weekend's showdown.",
            "time": "4 hours ago",
            "category": "Team News"
        },
        {
            "title": "Saudi Pro League Sets New Attendance Record",
            "summary": "Over 580,000 fans attended matches this weekend, marking the highest weekly attendance in league history.",
            "time": "6 hours ago", 
            "category": "League News"
        }
    ]
    
    for i, article in enumerate(articles):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        border-radius: 15px; padding: 1.5rem; margin: 1rem 0; color: white; min-height: 200px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.8rem; border-radius: 15px; font-size: 0.75rem; font-weight: 600;">
                        {article['category']}
                    </span>
                    <span style="font-size: 0.8rem;">{article['time']}</span>
                </div>
                <h4 style="margin: 0 0 1rem 0; font-size: 1.1rem; font-weight: 600; line-height: 1.3;">
                    {article['title']}
                </h4>
                <p style="margin: 0; font-size: 0.9rem; line-height: 1.4; opacity: 0.9;">
                    {article['summary']}
                </p>
            </div>
            """, unsafe_allow_html=True)

def display_league_table_tab():
    """Display the league table"""
    st.markdown("## 📋 Current League Standings")
    
    standings_df = load_standings_data()
    
    if not standings_df.empty:
        # Team logos as HTML
        standings_df["Logo"] = standings_df["Logo"].apply(
            lambda url: f"<img src='{url}' width='28'>" if url else ""
        )
        
        # Team filter
        unique_teams = standings_df["Team"].unique().tolist()
        selected_teams = st.multiselect(
            label="🔍 Select teams to filter",
            options=unique_teams,
            default=None,
            placeholder="Choose team(s)..."
        )
        
        # Apply filter
        if selected_teams:
            filtered_df = standings_df[standings_df["Team"].isin(selected_teams)]
        else:
            filtered_df = standings_df
        
        # Display table
        st.write(filtered_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.error("No standings data available")

def display_fantasy_football_tab():
    """Display fantasy football management"""
    st.markdown("## 🏆 Fantasy Football Manager")
    st.markdown("Build your dream team and compete with friends!")
    
    # Initialize session state
    if "fantasy_team" not in st.session_state:
        st.session_state.fantasy_team = []
    if "budget" not in st.session_state:
        st.session_state.budget = 100_000_000  # 100M SAR
    
    players_df = load_players_data()
    
    if not players_df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 👥 Available Players")
            
            # Player filters
            positions = ["All"] + sorted(players_df["position"].unique().tolist())
            teams = ["All"] + sorted(players_df["team"].unique().tolist())
            
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                position_filter = st.selectbox("Position", positions)
            with filter_col2:
                team_filter = st.selectbox("Team", teams)
            
            # Apply filters
            filtered_players = players_df.copy()
            if position_filter != "All":
                filtered_players = filtered_players[filtered_players["position"] == position_filter]
            if team_filter != "All":
                filtered_players = filtered_players[filtered_players["team"] == team_filter]
            
            # Display players
            for idx, player in filtered_players.head(20).iterrows():
                with st.container():
                    player_col1, player_col2, player_col3 = st.columns([3, 1, 1])
                    
                    with player_col1:
                        st.write(f"**{player['name']}** - {player['team']}")
                        st.caption(f"{player['position']} • Age: {player['age']}")
                    
                    with player_col2:
                        st.write(f"Overall: {player['overall']}")
                        st.write(format_sar(player['price']))
                    
                    with player_col3:
                        if len(st.session_state.fantasy_team) < 15:
                            if st.button("Add", key=f"add_{idx}"):
                                if st.session_state.budget >= player['price']:
                                    st.session_state.fantasy_team.append(player.to_dict())
                                    st.session_state.budget -= player['price']
                                    st.rerun()
                                else:
                                    st.error("Insufficient budget!")
        
        with col2:
            st.markdown("### 💰 Team Management")
            
            # Budget display
            used_budget = 100_000_000 - st.session_state.budget
            st.metric("Budget Used", format_sar(used_budget), f"Remaining: {format_sar(st.session_state.budget)}")
            
            # Team display
            st.markdown(f"### 👥 Your Squad ({len(st.session_state.fantasy_team)}/15)")
            
            for i, player in enumerate(st.session_state.fantasy_team):
                col_player, col_remove = st.columns([3, 1])
                with col_player:
                    st.write(f"**{player['name']}** - {player['position']}")
                    st.caption(format_sar(player['price']))
                with col_remove:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.budget += player['price']
                        st.session_state.fantasy_team.pop(i)
                        st.rerun()
            
            if st.button("🗑️ Clear Team"):
                st.session_state.fantasy_team = []
                st.session_state.budget = 100_000_000
                st.rerun()
    else:
        st.error("No player data available")

def display_statistics_tab():
    """Display league and player statistics"""
    st.markdown("## 📊 League Statistics")
    
    teams_df = load_teams_data()
    top_scorers_df = load_top_scorers()
    
    if not teams_df.empty:
        st.markdown("### 🏆 Team Performance")
        
        # Team statistics chart
        fig = px.bar(
            teams_df.sort_values('points', ascending=False).head(10),
            x='name',
            y='points',
            title='Top 10 Teams by Points',
            color='points',
            color_continuous_scale='blues'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.dataframe(teams_df, use_container_width=True)
    
    if not top_scorers_df.empty:
        st.markdown("### ⚽ Top Scorers")
        
        # Top scorers chart
        fig_scorers = px.bar(
            top_scorers_df.head(10),
            x='name',
            y='goals',
            title='Top 10 Goalscorers',
            color='goals',
            color_continuous_scale='reds'
        )
        fig_scorers.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_scorers, use_container_width=True)
        
        # Display table
        st.dataframe(top_scorers_df, use_container_width=True)

def display_fixtures_tab():
    """Display fixtures and results"""
    st.markdown("## 📅 Fixtures & Results")
    
    # Sample fixtures data
    fixtures = [
        {"date": "2025-08-24", "time": "19:00", "home": "Al Hilal", "away": "Al Nassr", "status": "upcoming"},
        {"date": "2025-08-25", "time": "20:00", "home": "Al Ittihad", "away": "Al Ahli", "status": "upcoming"},
        {"date": "2025-08-17", "time": "19:00", "home": "Al Nassr", "away": "Al Hilal", "status": "completed", "home_score": 2, "away_score": 3},
    ]
    
    for fixture in fixtures:
        if fixture['status'] == 'upcoming':
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                        border-radius: 15px; padding: 1.5rem; margin: 1rem 0; color: white;">
                <div style="text-align: center;">
                    <h3>{fixture['home']} vs {fixture['away']}</h3>
                    <p>📅 {fixture['date']} • ⏰ {fixture['time']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                        border-radius: 15px; padding: 1.5rem; margin: 1rem 0; color: white;">
                <div style="text-align: center;">
                    <h3>{fixture['home']} {fixture.get('home_score', 0)} - {fixture.get('away_score', 0)} {fixture['away']}</h3>
                    <p>📅 {fixture['date']} • ✅ FULL TIME</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_ai_assistant_tab():
    """Display the AI Assistant (existing chatbot functionality)"""
    st.markdown("## 🤖 SPL AI Assistant")
    st.markdown("Ask me anything about the Saudi Pro League!")
    
    # Initialize session state for chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "team_logo" not in st.session_state:
        st.session_state.team_logo = "https://cdn-icons-png.flaticon.com/512/847/847969.png"
    
    # Load existing agent functionality
    try:
        agent = create_agent_from_jsons()
        df = st.session_state.get("json_qa_df")
        
        if df is not None:
            column_guide = generate_column_guide(df)
        else:
            column_guide = ""
        
        # Display chat history
        for role, content in st.session_state.chat_history:
            if role == "user":
                rendered = user_template.replace("{MSG}", content).replace("{USER_AVATAR}", st.session_state.team_logo)
            else:
                rendered = bot_template.replace("{MSG}", content)
            st.write(rendered, unsafe_allow_html=True)
        
        # Chat input
        user_input = st.chat_input("Ask your analyst something...")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            
            try:
                full_prompt = f"{column_guide}\n\nUser Question: {user_input}"
                answer = agent.run(full_prompt)
            except Exception as e:
                answer = f"⚠️ Sorry, I ran into an error: {str(e)}"
            
            st.session_state.chat_history.append(("assistant", answer))
            st.rerun()
            
    except Exception as e:
        st.error(f"AI Assistant unavailable: {e}")
        st.info("Please ensure all required data files and dependencies are available.")

def display_avatar_generator_tab():
    """Display the Avatar Generator"""
    st.markdown("## 🎨 Avatar Generator")
    st.markdown("Create your personalized SPL avatar!")
    
    if not RAG_AVAILABLE:
        st.warning("⚠️ Avatar Generator requires additional setup. Please install the required dependencies.")
        st.info("Missing modules: utils.img2img_utils, diffusers, torch")
        return
    
    uploaded_file = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])
    prompt_suffix = st.text_input(
        "Customization prompt:",
        value="Add details to your image or click Generate for a randomized look!"
    )
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Your uploaded image", use_container_width=True)
        
        if st.button("🎨 Generate Avatar"):
            with st.spinner("Generating your SPL avatar..."):
                try:
                    # This would use the image generation pipeline
                    st.success("Avatar generation feature coming soon!")
                    st.info("This feature requires Stable Diffusion setup.")
                except Exception as e:
                    st.error(f"Error generating avatar: {e}")

def display_chant_generator_tab():
    """Display the Chant Generator"""
    st.markdown("## 📣 Chant Generator")
    st.markdown("Generate exciting football chants for your team!")
    
    user_input = st.text_input("Enter a team name, player name, or keyword:")
    
    if st.button("🎵 Generate Chant"):
        if user_input:
            with st.spinner("Creating your chant..."):
                try:
                    if RAG_AVAILABLE:
                        # Use the actual chant generation
                        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                        if OPENAI_API_KEY:
                            chant_chain = setup_chant_chain(CHANT_PROMPT)
                            generated_chant = chant_chain.invoke(user_input)
                            st.markdown(f"```\n{generated_chant}\n```")
                            st.success("Your chant is ready! 🎉")
                        else:
                            st.error("OpenAI API key not found.")
                    else:
                        # Fallback simple chant generation
                        sample_chant = f"""
{user_input.upper()}! {user_input.upper()}!
We are the best, we'll pass the test!
From Riyadh to Jeddah, we sing with pride,
{user_input} forever by our side!

Olé, olé, olé, olé!
{user_input} will win the day!
Green and gold, our hearts are bold,
Victory stories will be told!
                        """
                        st.markdown(f"```\n{sample_chant.strip()}\n```")
                        st.success("Your chant is ready! 🎉")
                        st.info("Enhanced AI chant generation available with full setup.")
                        
                except Exception as e:
                    st.error(f"Error generating chant: {e}")
        else:
            st.warning("Please enter a keyword to generate a chant.")

def display_trivia_tab():
    """Display the Trivia Game"""
    st.markdown("## 🧠 SPL Trivia Challenge")
    st.markdown("Test your knowledge of the Saudi Pro League!")
    
    # Initialize trivia session state
    if "current_question" not in st.session_state:
        st.session_state.current_question = None
    if "score" not in st.session_state:
        st.session_state.score = 0
    if "total_questions" not in st.session_state:
        st.session_state.total_questions = 0
    if "answered" not in st.session_state:
        st.session_state.answered = False
    
    def start_new_question():
        # Sample trivia questions
        sample_questions = [
            {
                "question": "Which team has won the most Saudi Pro League titles?",
                "options": ["Al Hilal", "Al Nassr", "Al Ittihad", "Al Ahli"],
                "correct_answer_index": 0
            },
            {
                "question": "In which city is the King Fahd International Stadium located?",
                "options": ["Jeddah", "Dammam", "Riyadh", "Mecca"],
                "correct_answer_index": 2
            },
            {
                "question": "Who is known as 'CR7' and plays for Al Nassr?",
                "options": ["Lionel Messi", "Cristiano Ronaldo", "Neymar", "Karim Benzema"],
                "correct_answer_index": 1
            }
        ]
        
        import random
        st.session_state.current_question = random.choice(sample_questions)
        st.session_state.answered = False
        st.session_state.total_questions += 1
    
    # Display score
    st.markdown(f"**Score: {st.session_state.score}/{st.session_state.total_questions}**")
    
    if st.session_state.current_question is None:
        if st.button("🎯 Start Trivia Game"):
            start_new_question()
            st.rerun()
    else:
        question = st.session_state.current_question
        st.markdown(f"**Question:** {question['question']}")
        
        if not st.session_state.answered:
            user_answer = st.radio("Choose your answer:", question['options'])
            
            if st.button("Submit Answer"):
                selected_index = question['options'].index(user_answer)
                if selected_index == question['correct_answer_index']:
                    st.success("✅ Correct!")
                    st.session_state.score += 1
                else:
                    correct_answer = question['options'][question['correct_answer_index']]
                    st.error(f"❌ Incorrect. The correct answer was: **{correct_answer}**")
                
                st.session_state.answered = True
                st.rerun()
        else:
            # Show correct answer and next question button
            if st.button("Next Question"):
                start_new_question()
                st.rerun()
            
            if st.button("Restart Game"):
                st.session_state.current_question = None
                st.session_state.score = 0
                st.session_state.total_questions = 0
                st.session_state.answered = False
                st.rerun()

# Helper functions for AI Assistant
def generate_column_guide(df):
    """Generate column guide for the AI assistant"""
    guide = "Column Reference (from combined JSON data):\n\n"
    grouped = {}

    for col in df.columns:
        if col.startswith("__"):
            continue
        prefix, field = col.split(".", 1) if "." in col else ("misc", col)
        grouped.setdefault(prefix, []).append(field)

    for source, fields in grouped.items():
        guide += f"📄 {source}.json\n"
        for field in fields:
            guide += f"- {source}.{field}\n"
        guide += "\n"

    return guide.strip()

@st.cache_resource
def create_agent_from_jsons():
    """Create the CSV agent from JSON data"""
    df = load_all_jsons()
    if df.empty:
        st.error("No data available for AI assistant")
        return None
        
    temp_csv = "temp_combined_data.csv"
    df.to_csv(temp_csv, index=False)
    
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4"),
        temp_csv,
        verbose=False,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )
    
    os.remove(temp_csv)
    st.session_state.json_qa_df = df
    return agent
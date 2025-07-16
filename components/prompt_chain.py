import json
import streamlit as st
from openai import OpenAI 
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from config.env_loader import load_environment
from utils.helpers import generate_lineup_briefing
from config.env_loader import load_environment
from components.rag_engine import build_vectorstore, load_documents, retriever

env = load_environment()
docs = load_documents()
vectorstore = build_vectorstore(docs)
retriever = vectorstore.as_retriever()

# Prompt templates per assistant style
PROMPT_TEMPLATES = { "Saudi Pro League Analyst": """
You are an advanced Saudi Pro League assistant trained on structured Football API data.

Knowledge Source: Local Football API JSON files (structured from https://www.api-football.com/)
Coverage Scope: Saudi Pro League (SPL) only
Data Endpoints Available:
- Countries
- Seasons
- Leagues
- Standings
- Teams
- Fixtures
- Livescore
- Head 2 Head
- Events
- Line Ups
- Top Scorers
- Players & Coaches
- Players Transfers
- Trophies
- Sidelined
- Injuries
- In-play Odds
- Pre-match Odds
- Statistics
- Predictions

Your capabilities:
✅ Answer factual questions using ONLY the available local JSON data from the Football API  
✅ Support tactical and analytical queries using pre-processed SPL data  
✅ Provide fan engagement features (chants, commentary tone, match highlight phrasing)  
✅ Translate to/from Arabic where applicable (if enabled by system)

Your boundaries:
❌ Do NOT hallucinate statistics or events not provided in context  
❌ Do NOT reference any league outside of the Saudi Pro League  
❌ Do NOT use any third-party sources, search results, or knowledge beyond the provided local data  
❌ Do NOT generate predictions or simulations unless explicitly requested and supported

Tone and Style Guidelines:
- Match the tone to the user intent:
  - Analyst-style → factual, structured, concise
  - Commentary-style → vivid, energetic, emoji-supported (e.g., ⚽🔥🟥)
  - Fan mode → informal, fun, engaging
- Use bullet points and spacing for clarity
- Support emojis to improve readability or excitement — avoid overuse
- If user data is missing or unclear, respond honestly (e.g., "I couldn’t find that info in the current match data.")
- Never use overly formal or robotic phrasing. Be helpful and clear.

Tool Integrations:
- Match Simulation Agent (optional): simulates SPL matches based on API data and user decisions
- Prompt Routing: directs user queries to the correct format or endpoint (e.g., Top Scorer → Statistics)
- Translation Router: switches between English and Saudi Arabic if user preference is detected

<API Data Context>
{context}
</API Data Context>

Now respond appropriately to the user's request using ONLY the data provided and following the role, rules, and tone defined above.
"""
}

router_prompt = ChatPromptTemplate.from_template("""
You are a query classification assistant for a football chatbot that only covers the Saudi Pro League.

Based on the user's message, classify it into ONE of the following categories:

- top_scorer (questions about who scored the most goals, leading scorers, golden boot)
- standings (questions about rankings, league table, team positions)
- live_score (questions about ongoing matches or real-time scores)
- fixtures (questions about upcoming matches, schedules)
- match_events (questions about what happened in a match — goals, cards, substitutions)
- lineups (questions about starting eleven, formations, players in the lineup)
- player_stats (questions about individual player performance — goals, assists, cards)
- injuries (questions about which players are injured or sidelined)
- transfers (questions about player movements between clubs)
- trophies (questions about team honors or titles won)

User message: "{query}"

Return ONLY the category name (e.g., "top_scorer"). Do not explain or format anything else.
""")

router_chain = LLMChain(
    llm=ChatOpenAI(temperature=0, openai_api_key=env["OPENAI_API_KEY"]),
    prompt=router_prompt
)

def classify_query_llm(query: str, llm=None) -> str:
    """Classifies user query into one Football API category using the provided LLM."""
    router = LLMChain(llm=llm, prompt=router_prompt)
    result = router.invoke({"query": query})
    return result["text"].strip() if isinstance(result, dict) else result.strip()



def load_and_format_top_scorers():
    try:
        with open("data/top_scorers.json", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            return "No top scorers data found."

        context = "Saudi Pro League – Top Scorers:\n"
        for i, player in enumerate(data[:5], 1):  # limit to top 5
            name = player["player_name"]["english"]
            team = player["team"]["english"]
            nationality = player["nationality"]
            age = player["age"]
            goals = player["goals"]
            assists = player["assists"]
            appearances = player["appearances"]
            minutes = player["minutes_played"]

            context += (
                f"{i}. {name} ({nationality}, {age} yrs) – {team} – "
                f"{goals} goals, {assists} assists in {appearances} games "
                f"({minutes} mins played)\n"
            )

        return context
    except Exception as e:
        return f"⚠️ Failed to load top scorers: {e}"



def load_and_format_standings(lang="english"):
    try:
        with open("data/standings.json", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or not data:
            return "No standings data found."

        context = "Current SPL Standings:\n\n"
        for team in data[:10]:
            rank = team.get("position", "-")
            name = team.get("team", "Unknown Team")
            points = team.get("points", 0)
            wins = team.get("won", 0)
            draws = team.get("draw", 0)
            losses = team.get("lost", 0)

            context += f"{rank}. {name} – {points} pts (W:{wins} D:{draws} L:{losses})\n"

        return context

    except Exception as e:
        return f"⚠️ Failed to load standings: {e}"



def load_and_format_fixtures():
    try:
        with open("data/fixtures.json", encoding="utf-8") as f:
            data = json.load(f)

        fixtures = data.get("response", [])
        if not fixtures:
            return "No upcoming fixture data found."

        context = "Upcoming SPL Fixtures:\n\n"
        for fixture in fixtures[:5]:  # Show top 5 upcoming
            home = fixture.get("teams", {}).get("home", {}).get("name", "Home Team")
            away = fixture.get("teams", {}).get("away", {}).get("name", "Away Team")
            date = fixture.get("fixture", {}).get("date", "")[:10]
            venue = fixture.get("fixture", {}).get("venue", {}).get("name", "Unknown Venue")
            context += f"{home} vs {away} – {date} at {venue}\n"

        return context

    except Exception as e:
        return f"⚠️ Failed to load fixtures: {e}"

def load_and_format_transfers():
    try:
        with open("data/transfers.json", encoding="utf-8") as f:
            data = json.load(f)

        transfers = data.get("response", [])
        if not transfers:
            return "No recent transfers found."

        context = "Recent SPL Transfers:\n\n"
        for t in transfers[:5]:  # Top 5 transfers
            player = t.get("player", {}).get("name", "Unknown Player")
            from_team = t.get("transfers", [{}])[0].get("teams", {}).get("out", {}).get("name", "Unknown")
            to_team = t.get("transfers", [{}])[0].get("teams", {}).get("in", {}).get("name", "Unknown")
            date = t.get("transfers", [{}])[0].get("date", "-")
            type_ = t.get("transfers", [{}])[0].get("type", "N/A")
            context += f"{player}: {from_team} → {to_team} on {date} ({type_})\n"

        return context

    except Exception as e:
        return f"⚠️ Failed to load transfers: {e}"


def load_and_format_lineups():
    try:
        with open("data/lineups_latest_per_team.json", encoding="utf-8") as f:
            data = json.load(f)

        lineups = data.get("response", [])
        if not lineups:
            return "No lineup data found."

        context = "Recent SPL Lineups:\n\n"
        for item in lineups[:2]:  # Show 2 teams (or 2 lineups)
            team_name = item.get("team", {}).get("name", "Unknown Team")
            formation = item.get("formation", "N/A")
            coach = item.get("coach", {}).get("name", "Unknown Coach")
            players = item.get("startXI", [])

            context += f"🔷 {team_name} – Formation: {formation}, Coach: {coach}\n"
            for p in players[:5]:  # Show top 5 starters only
                name = p.get("player", {}).get("name", "Unnamed")
                number = p.get("player", {}).get("number", "-")
                pos = p.get("player", {}).get("pos", "N/A")
                context += f"  • #{number} {name} ({pos})\n"
            context += "\n"

        return context

    except Exception as e:
        return f"⚠️ Failed to load lineups: {e}"


def load_and_format_player_stats():
    try:
        with open("data/stats_latest_per_team.json", encoding="utf-8") as f:
            data = json.load(f)

        teams = data.get("response", [])
        if not teams:
            return "No player statistics found."

        context = "Latest Player Stats by Team:\n\n"
        for team in teams[:2]:  # Show 2 teams
            team_name = team.get("team_name", "Unknown Team")
            players = team.get("players", [])
            context += f"🔷 {team_name}:\n"

            for p in players[:3]:  # Top 3 players only
                name = p.get("name", "Unnamed")
                pos = p.get("position", "N/A")
                goals = p.get("goals", 0)
                assists = p.get("assists", 0)
                yellow = p.get("yellow_cards", 0)
                red = p.get("red_cards", 0)

                context += f"  • {name} ({pos}) – Goals: {goals}, Assists: {assists}, 🟨 {yellow}, 🟥 {red}\n"
            context += "\n"

        return context

    except Exception as e:
        return f"⚠️ Failed to load player stats: {e}"


def load_and_format_match_events():
    try:
        with open("data/events_sample.json", encoding="utf-8") as f:
            data = json.load(f)

        events = data.get("response", [])
        if not events:
            return "No match event data found."

        context = "Recent SPL Match Events:\n\n"
        for event in events[:10]:  # Top 10 events
            time = event.get("time", {}).get("elapsed", "-")
            team = event.get("team", {}).get("name", "Unknown Team")
            player = event.get("player", {}).get("name", "Unknown Player")
            type_ = event.get("type", "Event")
            detail = event.get("detail", "")

            context += f"⏱ {time}' – {type_} – {player} ({team})"
            if detail and detail != type_:
                context += f" [{detail}]"
            context += "\n"

        return context

    except Exception as e:
        return f"⚠️ Failed to load match events: {e}"

def load_and_format_match_events():
    try:
        with open("data/events_sample.json", encoding="utf-8") as f:
            data = json.load(f)

        events = data.get("response", [])
        if not events:
            return "No match event data found."

        context = "Recent SPL Match Events:\n\n"
        for event in events[:10]:  # Show up to 10 events
            time = event.get("time", {}).get("elapsed", "-")
            team = event.get("team", {}).get("name", "Unknown Team")
            player = event.get("player", {}).get("name", "Unknown Player")
            type_ = event.get("type", "Event")
            detail = event.get("detail", "")

            context += f"⏱ {time}' – {type_} – {player} ({team})"
            if detail and detail != type_:
                context += f" [{detail}]"
            context += "\n"

        return context

    except Exception as e:
        return f"⚠️ Failed to load match events: {e}"


def load_and_format_teams():
    try:
        with open("data/teams.json", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            return "No team data found."

        context = "Saudi Pro League Teams Overview:\n\n"
        for team in data[:5]:  # Limit to top 5
            name_en = team.get("team_name", {}).get("english", "Unknown Team")
            name_ar = team.get("team_name", {}).get("arabic_saudi", "")
            founded = team.get("founded", "-")
            stadium = team.get("stadium_name", "Unknown Stadium")
            city = team.get("stadium_city", "Unknown City")
            capacity = team.get("stadium_capacity", "-")

            context += f"🏟️ {name_en} ({name_ar})\n"
            context += f"  • Founded: {founded}\n"
            context += f"  • Stadium: {stadium} – {city} ({capacity} seats)\n\n"

        return context

    except Exception as e:
        return f"⚠️ Failed to load team data: {e}"


def load_and_format_players():
    try:
        with open("data/players.json", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            return "No player data available."

        context = "Squad Overviews (from local players.json):\n\n"
        for team in data[:3]:  # Limit to 3 teams for brevity
            team_name = team.get("team_name", "Unknown Team")
            players = team.get("players", [])

            context += f"👥 {team_name}:\n"
            for player in players[:5]:  # First 5 players per team
                name = player.get("name", "Unnamed")
                age = player.get("age", "-")
                position = player.get("position", "N/A")
                nationality = player.get("nationality", "Unknown")
                foot = player.get("foot", "N/A")
                height = player.get("height", "-")
                context += (
                    f"  • {name} – {position}, Age: {age}, "
                    f"{nationality}, {foot}-footed, {height} tall\n"
                )
            context += "\n"

        return context

    except Exception as e:
        return f"⚠️ Failed to load players.json: {e}"


def build_tactical_context(language, lineup, full_lineup, formation="Unknown"):
    if not lineup:
        return "No starting XI has been selected yet."

    summary_lines = [f"Formation: {formation}", "Starting XI:"]
    for player_str in lineup:
        name = player_str.split(" (")[0]
        player_data = next((p for p in full_lineup if p["name"] == name), {})
        summary_lines.append(
            f"- {player_data.get('name', 'Unknown')} ({player_data.get('position', '-')}, "
            f"{player_data.get('age', '-')} yrs, {player_data.get('team', '-')}, "
            f"Ovr: {player_data.get('overall', '-')}, Cost: SAR {player_data.get('cost', 0):,})"
        )

    # Optional: Include info on remaining bench
    if len(full_lineup) > len(lineup):
        summary_lines.append("\nBench Options:")
        bench_players = [p for p in full_lineup if p["name"] not in [s.split(" (")[0] for s in lineup]]
        for p in bench_players:
            summary_lines.append(
                f"- {p['name']} ({p['position']}, {p['age']} yrs, {p['team']}, Ovr: {p['overall']})"
            )

    return "\n".join(summary_lines)


def handle_user_query(user_prompt, language="english", lineup=[], full_lineup=[], formation="Unknown"):
    # 🔍 RAG context from JSON/CSV
    rag_docs = retriever.get_relevant_documents(user_prompt)
    rag_context = "\n".join([doc.page_content for doc in rag_docs])

    # 🧠 Tactical context
    tactical_lines = [f"Formation: {formation}", "Starting XI:"]
    for p in lineup:
        player_name = p.split(" (")[0]
        match = next((x for x in full_lineup if x["name"] == player_name), {})
        if match:
            tactical_lines.append(
                f"- {match.get('name')} ({match.get('position')}, {match.get('age')} yrs, {match.get('team')}, Ovr: {match.get('overall')})"
            )
    tactical_context = "\n".join(tactical_lines)

    # 🧩 Full context (tactical + retrieved)
    full_context = f"{tactical_context}\n\n📚 Reference Material:\n{rag_context}"

    # 🔗 Prompt setup
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a tactical football assistant grounded in data from the Saudi Pro League."),
        ("user", "Context:\n{context}\n\nQuestion: {user_prompt}")
    ])
    
    llm = ChatOpenAI(temperature=0.7, model="gpt-4")
    chain = LLMChain(llm=llm, prompt=prompt_template)

    return chain.run({"context": full_context, "user_prompt": user_prompt})



def get_custom_chain(temperature: float = 0.7) -> LLMChain:
    """Returns a simplified LLMChain for a unified SPL assistant."""
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["Saudi Pro League Analyst"])
    llm = ChatOpenAI(temperature=temperature, openai_api_key=env["OPENAI_API_KEY"])
    return LLMChain(llm=llm, prompt=prompt)


def render_lineup_summary(lineup, full_lineup, language="english"):
    lines = []

    for p in lineup:
        # Extract name from "Name (Club)" format
        player_name = p.split(" (")[0]
        player_data = next((x for x in full_lineup if x["name"] == player_name), {})

        name = player_data.get("name", "-")
        position = player_data.get("position", "-")
        club = player_data.get("team", "-")
        age = player_data.get("age", "-")
        cost = player_data.get("cost", 0)
        formatted_cost = f"SAR {cost:,}"

        if language == "english":
            lines.append(f"- **{name}** ({position}, {age} yrs) from {club} — {formatted_cost}")
        else:
            lines.append(f"- **{name}** ({position}، {age} سنة) من {club} — {formatted_cost}")

    return "\n".join(lines)

def render_chat_message(role, message):
    if role == "user":
        st.markdown(f"**🗣️ You:** {message}")
    else:
        if st.session_state.language == "arabic_saudi":
            st.markdown(f"<div style='direction: rtl; text-align: right;'>🧠 {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: #f0f2f6; padding: 8px; border-radius: 8px;'>🧠 {message}</div>", unsafe_allow_html=True)

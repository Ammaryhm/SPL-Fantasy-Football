import streamlit as st
import random
from datetime import datetime
import json
import os

def update_leaderboard(username, result, user_score, opponent_score):
    """Update leaderboard with match result"""
    leaderboard = load_leaderboard()
    
    if username not in leaderboard:
        leaderboard[username] = {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "matches_played": 0,
            "last_played": None
        }
    
    stats = leaderboard[username]
    stats["matches_played"] += 1
    stats["goals_for"] += user_score
    stats["goals_against"] += opponent_score
    stats["last_played"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    if result == "win":
        stats["wins"] += 1
    elif result == "draw":
        stats["draws"] += 1
    else:
        stats["losses"] += 1
    
    save_leaderboard(leaderboard)

def load_leaderboard():
    """Load leaderboard from JSON file"""
    try:
        if os.path.exists("leaderboard.json"):
            with open("leaderboard.json", "r") as f:
                return json.load(f)
        return {}
    except:
        return {}

def save_leaderboard(leaderboard):
    """Save leaderboard to JSON file"""
    try:
        with open("leaderboard.json", "w") as f:
            json.dump(leaderboard, f, indent=2)
    except:
        pass


def display_leaderboard():
    """Display the leaderboard"""
    st.divider()
    st.subheader("🏆 Leaderboard")
    
    leaderboard = load_leaderboard()
    
    if not leaderboard:
        st.info("No matches played yet. Be the first to play!")
        return
    
    # Calculate additional stats and sort by wins, then by goal difference
    leaderboard_data = []
    for username, stats in leaderboard.items():
        goal_diff = stats["goals_for"] - stats["goals_against"]
        points = stats["wins"] * 3 + stats["draws"] * 1  # 3 points for win, 1 for draw
        win_rate = (stats["wins"] / stats["matches_played"] * 100) if stats["matches_played"] > 0 else 0
        
        leaderboard_data.append({
            "Player": username,
            "Matches": stats["matches_played"],
            "Wins": stats["wins"],
            "Draws": stats["draws"],
            "Losses": stats["losses"],
            "Goals For": stats["goals_for"],
            "Goals Against": stats["goals_against"],
            "Goal Diff": f"{goal_diff:+d}",
            "Points": points,
            "Win Rate": f"{win_rate:.1f}%",
            "Last Played": stats["last_played"] or "Never"
        })
    
    # Sort by points (descending), then by goal difference (descending), then by wins (descending)
    leaderboard_data.sort(key=lambda x: (-x["Points"], -int(x["Goal Diff"]), -x["Wins"]))
    
    # Display top 10
    st.markdown("#### 🥇 Top Players")
    for i, player_data in enumerate(leaderboard_data[:10], 1):
        # Medal emojis for top 3
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        
        col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
        with col1:
            st.markdown(f"**{medal}**")
        with col2:
            st.markdown(f"**{player_data['Player']}**")
        with col3:
            st.markdown(f"{player_data['Wins']}-{player_data['Draws']}-{player_data['Losses']} ({player_data['Points']} pts)")
        with col4:
            st.markdown(f"GD: {player_data['Goal Diff']}, WR: {player_data['Win Rate']}")
    
    # Show detailed stats in expander
    with st.expander("📊 Detailed Statistics"):
        st.dataframe(leaderboard_data, use_container_width=True, hide_index=True)


def simulate_match_with_leaderboard(user_xi, opponent_xi, username=None):
    """
    Simulates a match and updates leaderboard with results.
    """
    if len(user_xi) != 11 or len(opponent_xi) != 11:
        st.error("Both teams must have exactly 11 players!")
        return

    st.divider()
    st.subheader("⚽ Live Match Simulation")

    # Initialize match state if not already done
    if "match_started" not in st.session_state:
        st.session_state.match_started = False
        st.session_state.match_events = []
        st.session_state.match_minute = 0
        st.session_state.user_score = 0
        st.session_state.opponent_score = 0
        st.session_state.match_finished = False
        st.session_state.ronaldo_goal_scored = False

    if not st.session_state.match_started:
        if st.button("🚀 Start Match Simulation"):
            st.session_state.match_started = True
            st.session_state.match_events = ["⚽ Match begins! Both teams looking sharp."]
            st.rerun()

    if st.session_state.match_started and not st.session_state.match_finished:
        # Match progress bar
        progress = min(st.session_state.match_minute / 90, 1.0)
        st.progress(progress, text=f"⏱️ {st.session_state.match_minute}' - Score: {st.session_state.user_score}-{st.session_state.opponent_score}")

        if st.button("⏭️ Next Event", key="next_event_btn"):
            ronaldo_in_user = any(
                player.get("name", "").lower() == "cristiano ronaldo"
                for player in user_xi
            )

            scripted_event_triggered = False
            target_minute = min(st.session_state.match_minute + 15, 90)

            while st.session_state.match_minute < target_minute:
                st.session_state.match_minute += 1

                # Scripted Ronaldo event
                if (ronaldo_in_user and
                    not st.session_state.ronaldo_goal_scored and
                    st.session_state.match_minute == 80):
                    st.session_state.match_events.append("⚽ 80' - Foul on the edge of the box...")
                    scripted_event_triggered = True
                    break

                elif (ronaldo_in_user and
                      not st.session_state.ronaldo_goal_scored and
                      st.session_state.match_minute == 81):
                    st.session_state.match_events.append("🎉 81' - GOAL! Cristiano Ronaldo curls a magnificent free kick into the top corner! Unstoppable!")
                    st.session_state.user_score = 1
                    st.session_state.opponent_score = 0
                    st.session_state.ronaldo_goal_scored = True
                    scripted_event_triggered = True
                    break

                elif not (ronaldo_in_user and 80 <= st.session_state.match_minute <= 81):
                    if random.random() < 0.4:
                        generic_events = [
                            "Corner kick goes wide", "Offside called on the attack",
                            "Tactical foul in midfield", "Throw-in near halfway line",
                            "Missed chance from close range", "Routine save by the goalkeeper",
                            "Cross deflected out for a corner", "Scrappy play in midfield continues",
                            "Coach shouting tactical instructions", "Yellow card for dissent",
                            "Header over the bar", "Shot blocked by defender",
                            "Keeper claims the cross", "Play stopped for injury"
                        ]

                        # Only include substitution in second half
                        if st.session_state.match_minute >= 46:
                            generic_events.append("Substitution: Fresh legs come on")

                        event = random.choice(generic_events)
                        st.session_state.match_events.append(f"⚽ {st.session_state.match_minute}' - {event}")

            if st.session_state.match_minute >= 90:
                st.session_state.match_finished = True
                st.session_state.match_events.append(f"🏁 90' - Full time! Final score: {st.session_state.user_score}-{st.session_state.opponent_score}")

            st.rerun()

    # Show recent match events
    if st.session_state.match_events:
        st.subheader("📰 Match Events")
        for event in st.session_state.match_events[-5:]:
            st.write(event)

    # Final result and leaderboard
    if st.session_state.match_finished:
        st.divider()
        st.subheader("🏆 Final Result")
        final_score = f"{st.session_state.user_score} - {st.session_state.opponent_score}"

        if st.session_state.user_score > st.session_state.opponent_score:
            result_text = f"🎉 Victory! You won {final_score}"
            result_color = "green"
            result_type = "win"
        elif st.session_state.user_score < st.session_state.opponent_score:
            result_text = f"😔 Defeat! You lost {final_score}"
            result_color = "red"
            result_type = "loss"
        else:
            result_text = f"🤝 Draw! Final score {final_score}"
            result_color = "orange"
            result_type = "draw"

        st.markdown(
            f"<div style='color: {result_color}; font-size: 20px; text-align: center; padding: 15px; border: 3px solid {result_color}; border-radius: 15px; font-weight: bold;'>"
            f"{result_text}</div>",
            unsafe_allow_html=True
        )

        if username:
            try:
                update_leaderboard(username, result_type, st.session_state.user_score, st.session_state.opponent_score)
                st.success(f"🏆 Leaderboard updated for {username}!")
            except Exception as e:
                st.warning(f"⚠️ Leaderboard update failed: {str(e)}")

        if st.button("🔄 Play Again"):
            for key in ["match_started", "match_events", "match_minute", "user_score", "opponent_score", "match_finished", "ronaldo_goal_scored"]:
                st.session_state[key] = False if isinstance(st.session_state[key], bool) else 0 if isinstance(st.session_state[key], int) else []
            st.rerun()



def reset_match_state():
    """Helper function to reset match state"""
    keys_to_reset = [
        "match_started", "match_events", "match_minute", 
        "user_score", "opponent_score", "match_finished", 
        "ronaldo_goal_scored"
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
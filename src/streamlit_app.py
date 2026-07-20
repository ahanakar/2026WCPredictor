import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title = "2026 World Cup Predictor", page_icon = "⚽", layout = 'centered')

st.title("2026 World Cup Match Predictor")
st.markdown("---------")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_groupstage_data():
    path = os.path.join(BASE_DIR, "../data/processed/groupstage_predictions_2026.csv")
    return pd.read_csv(path)

@st.cache_data
def load_ro32_data():
    path = os.path.join(BASE_DIR, "../data/processed/ro32_predictions_2026.csv")
    return pd.read_csv(path)

@st.cache_data
def load_ro16_data():
    path = os.path.join(BASE_DIR, "../data/processed/ro16_predictions_2026.csv")
    return pd.read_csv(path)

@st.cache_data
def load_quarters_data():
    path = os.path.join(BASE_DIR, "../data/processed/quarterfinals_predictions_2026.csv")
    return pd.read_csv(path)

@st.cache_data
def load_semis_data():
    path = os.path.join(BASE_DIR, "../data/processed/semifinals_predictions_2026.csv")
    return pd.read_csv(path)

@st.cache_data
def load_finals_data():
    path = os.path.join(BASE_DIR, "../data/processed/finals_predictions_2026.csv")
    return pd.read_csv(path)

selected_round = st.sidebar.selectbox("Select Tournament Round:", ["Group Stage", "Round of 32", "Round of 16", "Quarter finals", "Semi finals", "Finals"])

try:
    if selected_round == "Group Stage":
        df = load_groupstage_data()
    elif selected_round == "Round of 32":
        df = load_ro32_data()
    elif selected_round == "Round of 16":
        df = load_ro16_data()
    elif selected_round == "Quarter finals":
        df = load_quarters_data()
    elif selected_round == "Semi finals":
        df = load_semis_data()
    else:
        df = load_finals_data()
except FileNotFoundError:
    st.error(f"Error 404: Prediction data file for {selected_round} not found.")
    st.stop()

st.write(df.columns)

flag_map = {
    "Mexico": "mx", "South Africa": "za", "South Korea": "kr", "Czech Republic": "cz",
    "Canada": "ca", "Bosnia and Herzegovina": "ba", "United States": "us", "Paraguay": "py",
    "Australia": "au", "Turkey": "tr", "Haiti": "ht", "Scotland": "gb-sct", "Brazil": "br",
    "Morocco": "ma", "Qatar": "qa", "Switzerland": "ch", "Germany": "de", "Curacao": "cw",
    "Ivory Coast": "ci", "Ecuador": "ec", "Netherlands": "nl", "Japan": "jp", "Spain": "es",
    "Saudi Arabia": "sa", "England": "gb-eng", "Ghana": "gh", "Cape Verde": "cv", "Argentina": "ar",
    "Jordan": "jo", "Algeria": "dz", "France": "fr", "Iraq": "iq", "Norway": "no", "Senegal": "sn",
    "Colombia": "co", "DR Congo": "cd", "Portugal": "pt", "Uzbekistan": "uz", "Panama": "pa",
    "Croatia": "hr", "Austria": "at", "Iran": "ir", "Egypt": "eg", "Tunisia": "tn", "Sweden": "se",
    "New Zealand": "nz", "Belgium": "be"
}

if selected_round == "Group Stage":
    if 'Group' in df.columns:
        groups = sorted(df['Group'].unique())
        selected_group = st.sidebar.selectbox("Select a group: ", groups)
        filtered_df = df[df['Group'] == selected_group]
    else:
        st.sidebar.warning("Error. Showing all matches")
        filtered_df = df
else:
    filtered_df = df

match_options = [f"{row['Home Team']} vs {row['Away Team']}" for _, row in filtered_df.iterrows()]
selected_match = st.selectbox("Select match: ", match_options)

home_team, away_team = selected_match.split(" vs ")
match_data = filtered_df[(filtered_df['Home Team'] == home_team) & (filtered_df['Away Team'] == away_team)].iloc[0]

col1, col2, col3 = st.columns([3, 1, 3])

with col1:
    h_code = flag_map.get(home_team, "un")
    st.image(f"https://flagcdn.com/w160/{h_code}.png", width = 110)
    st.subheader(home_team)
    st.caption(f"Expected Goals: {match_data['Pred Home Goals']:.2f}")

with col2:
    st.markdown("<h2 style = 'text-align: center; color: gray; margin-top: 25px;'>VS</h2>", unsafe_allow_html = True)

with col3:
    a_code = flag_map.get(away_team, "un")
    st.image(f"https://flagcdn.com/w160/{a_code}.png", width = 110)
    st.subheader(away_team)
    st.caption(f"Expected Goals: {match_data['Pred Away Goals']:.2f}")

st.markdown("---------")

if selected_round == "Group Stage":
    hw_p = match_data['Home Win %']
    aw_p = match_data['Away Win %']
    draw_p = match_data['Draw %']
    scoreline = match_data['Predicted Scoreline']

    st.write("### Match Outcome Probability")
    progress_html = f"""
    <div style="width: 100%; background-color: #f1f1f1; border-radius: 8px; display: flex; overflow: hidden; height: 35px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);">
      <div style="width: {hw_p}%; background-color: #2E7D32; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">{hw_p}%</div>
      <div style="width: {draw_p}%; background-color: #757575; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">{draw_p}%</div>
      <div style="width: {aw_p}%; background-color: #1565C0; color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 14px;">{aw_p}%</div>
    </div>
    <div style="display: flex; justify-content: space-between; padding: 5px 5px 0px 5px; font-weight: 500; font-size: 12px; color: #424242;">
      <span>🟢 {home_team} Win</span>
      <span>⚪ Draw</span>
      <span>🔵 {away_team} Win</span>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    score_home, score_away = map(int, scoreline.split("-"))
    if score_home > score_away:
        st.success(f"**Predicted Result:** **{home_team} Win** ({scoreline})")
    elif score_away > score_home:
        st.info(f"**Predicted Result:** **{away_team} Win** ({scoreline})")
    else:
        st.warning(f"**Predicted Result:** **Draw** ({scoreline})")

else:
    hw_p = match_data['Home Win 90m %']
    aw_p = match_data['Away Win 90m %']
    draw_p = match_data['Draw 90m %']
    scoreline = match_data['Predicted 90m Score']
    team_to_advance = match_data['Team To Advance']
    
    tot_home_adv = match_data['Total Home Advance %']
    tot_away_adv = match_data['Total Away Advance %']

    st.write("### Tournament Advancement Verdict")
    st.success(f"**{team_to_advance}** is predicted to advance to the next round (Combined Probability: **{max(tot_home_adv, tot_away_adv)}%**)")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns(2)

    with col_left:
        st.write("#### Regulation Time (90m)")
        st.write(f"**Most Likely Score:** `{scoreline}`")
        
        reg_html = f"""
        <div style="width: 100%; background-color: #f1f1f1; border-radius: 6px; display: flex; overflow: hidden; height: 24px;">
          <div style="width: {hw_p}%; background-color: #2E7D32; color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold;">{hw_p}%</div>
          <div style="width: {draw_p}%; background-color: #757575; color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold;">{draw_p}%</div>
          <div style="width: {aw_p}%; background-color: #1565C0; color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold;">{aw_p}%</div>
        </div>
        """
        st.markdown(reg_html, unsafe_allow_html=True)
        st.caption(f"🟢 Win: {hw_p}% | ⚪ Draw: {draw_p}% | 🔵 Win: {aw_p}%")

    with col_right:
        st.write("#### Penalty Tie-Breaker")
        
        so_home_raw = match_data['Home Advance Shootout %']
        so_away_raw = match_data['Away Advance Shootout %']
        total_raw = so_home_raw + so_away_raw
        
        home_pure_so = (so_home_raw / total_raw) * 100 if total_raw > 0 else 50.0
        away_pure_so = (so_away_raw / total_raw) * 100 if total_raw > 0 else 50.0
        
        st.write(f"**If match ends in a draw:**")
        
        so_html = f"""
        <div style="width: 100%; background-color: #f1f1f1; border-radius: 6px; display: flex; overflow: hidden; height: 24px;">
          <div style="width: {home_pure_so}%; background-color: #E65100; color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold;">{home_pure_so:.1f}%</div>
          <div style="width: {away_pure_so}%; background-color: #006064; color: white; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold;">{away_pure_so:.1f}%</div>
        </div>
        """
        st.markdown(so_html, unsafe_allow_html=True)
        st.caption(f"🟠 {home_team} penalty win: {home_pure_so:.1f}% | 🔵 {away_team} penalty win: {away_pure_so:.1f}%")

    st.markdown("---")
    st.caption(
        f"**How this is calculated:** Even if the single most frequent scoreline coordinate is a draw (like `1-1`), "
        f"the model assesses *all possible outcomes*. If a team has a significantly higher chance to finish the game "
        f"inside the regular 90-minute window, their overall total advancement probability will often carry them through"
    )
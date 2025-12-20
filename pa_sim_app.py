import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from itertools import product
import warnings
from streamlit_drawable_canvas import st_canvas

from markov_functions import (
    get_pitches_with_counts,
    counts_prob,
    sample_pitch,
    prob_determine,
    write_situation,
    pitch_types,
    plotting_background,
    x_bound,
    y_bound,
    determine_quadrant,
    smooth_map,
    joint_prob_map,
    get_joint_called_strike_zone,
    _swing_tokens
)

# --- Page Config ---
st.set_page_config(layout="wide", page_title="CPBL Pitching Strategy Simulator")

# --- Data Loading ---
@st.cache_data
def load_data():
    """Loads the preprocessed pitch data."""
    pa_pitches_filename = './data/paired_filtered.csv'
    if not os.path.exists(pa_pitches_filename):
        st.error(f"Data file not found: {pa_pitches_filename}. Please run the data cleaning notebook first.")
        return None, None, None
    pas = pd.read_csv(pa_pitches_filename)
    pas.drop(['pa_seq', 'bases', 'velocities_events', 'pitchCodes_events'], axis=1, inplace=True, errors='ignore')
    pitchers = sorted(pas['pitcherName'].unique())
    batters = sorted(pas['batterName'].unique())
    return pas, pitchers, batters

pas, pitchers_list, batters_list = load_data()

if pas is None:
    st.stop()

# --- Helper Functions ---
def get_batter_data(pas_df, selected_batter_option, pitcher_hand):
    if selected_batter_option == "Right-Handed Batter":
        return pas_df[pas_df['batterHand'] == 'R']
    elif selected_batter_option == "Left-Handed Batter":
        return pas_df[pas_df['batterHand'] == 'L']
    else:
        return pas_df[pas_df['batterName'] == selected_batter_option]

def is_opposite_hand(pitcher_hand, batter_hand_char):
    return pitcher_hand != batter_hand_char

def get_all_events_for_count(df, opposite_hand):
    counts_config = {'ball': [0, 1, 2, 3], 'strike': [0, 1, 2]}
    
    # This is a simplified version. In the original notebook, it iterates.
    # For performance in an app, we might need a more direct way to get all relevant events.
    # Let's prepare the data for all counts at once.
    
    all_events = {}
    for b in counts_config['ball']:
        for s in counts_config['strike']:
            count_str = f"{b}-{s}"
            all_events[count_str] = get_pitches_with_counts(df, b, s, opposite_hand=opposite_hand)
    return all_events


def run_simulation_and_get_maps(pitcher_events_all, batter_events_all, situation_params, current_count_str):
    """Runs simulation to get next-pitch probability maps."""
    
    # We don't run a full PA simulation here, but instead calculate the probability maps for the *next* pitch
    # based on the current situation. This is more aligned with the KataGo concept.
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            
            pitchtype_map, swing_map, whiff_map, inplay_map, soft_map, called_strike_zone = counts_prob(
                current_count_str, pitcher_events_all, batter_events_all, situation_params=situation_params
            )
            
            # "Good result" map for the pitcher.
            # A good result is a swing-and-miss, a called strike, or soft contact.
            # P(Good) = P(No Swing) * P(Called Strike | No Swing) + P(Swing) * [ P(Whiff | Swing) + P(Contact | Swing) * P(Soft | Contact) ]
            
            prob_swing = swing_map
            prob_no_swing = 1 - prob_swing
            
            prob_whiff = whiff_map
            prob_contact = 1 - prob_whiff
            
            prob_soft_contact = soft_map
            
            prob_called_strike = called_strike_zone
            
            good_result_prob = prob_no_swing * prob_called_strike + prob_swing * (prob_whiff + prob_contact * prob_soft_contact)
            
            return {
                "pitch_type_map": pitchtype_map,
                "good_result_prob": good_result_prob,
            }

    except RuntimeWarning as e:
        st.warning(f"Could not calculate probabilities for the full context. Trying with a simpler context. Details: {e}")
        # Fallback strategies if the context is too specific and has no data
        fallback_strategies = [
            {},
            {'coords_quadrant_last2': None},
            {'swing_last2': None, 'whiff_last2': None},
            {'pitch_type_last2': None},
        ]
        for strategy in fallback_strategies:
            try:
                params_copy = situation_params.copy()
                params_copy.update(strategy)
                # Re-run with simpler params
                # This is a simplified version of the fallback in the notebook
                return run_simulation_and_get_maps(pitcher_events_all, batter_events_all, params_copy, current_count_str)
            except RuntimeWarning:
                continue
        st.error("All fallback strategies failed. Not enough data for this situation.")
        return None


# --- Session State Initialization ---
if 'pa_history' not in st.session_state:
    st.session_state.pa_history = []
if 'current_count' not in st.session_state:
    st.session_state.current_count = {'balls': 0, 'strikes': 0}
if 'situation_params' not in st.session_state:
    st.session_state.situation_params = {
        'pitch_type_last': None, 'coords_quadrant_last': None, 'swing_last': None, 'whiff_last': None,
        'pitch_type_last2': None, 'coords_quadrant_last2': None, 'swing_last2': None, 'whiff_last2': None
    }
if 'pitch_maps' not in st.session_state:
    st.session_state.pitch_maps = None
if 'pitcher_events' not in st.session_state:
    st.session_state.pitcher_events = None
if 'batter_events' not in st.session_state:
    st.session_state.batter_events = None


# --- UI Components ---
st.title("⚾️ CPBL Pitching Strategy Simulator")
st.write("Select a pitcher and batter, then add pitches to the sequence to see suggested locations for the next pitch.")

with st.sidebar:
    st.header("1. Configuration")
    
    selected_pitcher = st.selectbox("Select Pitcher", pitchers_list, key="pitcher_select")
    
    batter_options = ["Right-Handed Batter", "Left-Handed Batter"] + batters_list
    selected_batter_option = st.selectbox("Select Batter", batter_options, key="batter_select")

    if st.button("Load Matchup & Reset PA"):
        st.session_state.pa_history = []
        st.session_state.current_count = {'balls': 0, 'strikes': 0}
        st.session_state.situation_params = {
            'pitch_type_last': None, 'coords_quadrant_last': None, 'swing_last': None, 'whiff_last': None,
            'pitch_type_last2': None, 'coords_quadrant_last2': None, 'swing_last2': None, 'whiff_last2': None
        }
        st.session_state.pitch_maps = None
        
        pitcher_df = pas[pas['pitcherName'] == selected_pitcher]
        pitcher_hand = pitcher_df['pitcherHand'].iloc[0]
        
        if selected_batter_option == "Right-Handed Batter":
            batter_df = pas[pas['batterHand'] == 'R']
            opposite = is_opposite_hand(pitcher_hand, 'R')
        elif selected_batter_option == "Left-Handed Batter":
            batter_df = pas[pas['batterHand'] == 'L']
            opposite = is_opposite_hand(pitcher_hand, 'L')
        else:
            batter_df = pas[pas['batterName'] == selected_batter_option]
            batter_hand = batter_df['batterHand'].iloc[0]
            opposite = is_opposite_hand(pitcher_hand, batter_hand)

        with st.spinner("Loading and caching event data for this matchup..."):
            st.session_state.pitcher_events = get_all_events_for_count(pitcher_df, opposite)
            st.session_state.batter_events = get_all_events_for_count(batter_df, opposite)
        st.success("Matchup data loaded!")

    st.header("2. Add a Pitch")
    pitch_type_input = st.selectbox("Pitch Type", pitch_types)
    pitch_result_input = st.selectbox("Pitch Result", ["Called Strike", "Called Ball", "Swung and Missed", "Foul", "Soft In-Play", "Hard In-Play"])
    
    # Using a button to add the pitch
    if st.button("Add Pitch to Sequence"):
        # This is a placeholder for location. For now, we use a default.
        # In a real scenario, this would come from the drawable canvas.
        x_loc, y_loc = 0, 0 # Center of the plate
        
        # Update history
        st.session_state.pa_history.append({
            "pitch_num": len(st.session_state.pa_history) + 1,
            "type": pitch_type_input,
            "result": pitch_result_input,
            "count": f"{st.session_state.current_count['balls']}-{st.session_state.current_count['strikes']}"
        })
        
        # Update situation params
        swing = "Swing" in pitch_result_input or "Foul" in pitch_result_input
        whiff = "Missed" in pitch_result_input
        st.session_state.situation_params = write_situation(st.session_state.situation_params, x_loc, y_loc, pitch_type_input, swing, whiff)

        # Update count
        if pitch_result_input == "Called Strike" or pitch_result_input == "Swung and Missed":
            if st.session_state.current_count['strikes'] < 2:
                st.session_state.current_count['strikes'] += 1
        elif pitch_result_input == "Foul":
            if st.session_state.current_count['strikes'] < 2:
                st.session_state.current_count['strikes'] += 1
        elif pitch_result_input == "Called Ball":
            if st.session_state.current_count['balls'] < 3:
                st.session_state.current_count['balls'] += 1
        
        # Trigger recalculation
        st.session_state.pitch_maps = None


# --- Main App Logic ---
if st.session_state.pitcher_events is None or st.session_state.batter_events is None:
    st.info("Please load a matchup using the sidebar to begin.")
    st.stop()

# Display current state
col1, col2 = st.columns(2)
with col1:
    st.metric("Pitch Count", len(st.session_state.pa_history))
with col2:
    st.metric("Current Count", f"{st.session_state.current_count['balls']} - {st.session_state.current_count['strikes']}")

# Display PA History
if st.session_state.pa_history:
    st.write("#### Pitch Sequence")
    history_df = pd.DataFrame(st.session_state.pa_history)
    st.dataframe(history_df, use_container_width=True)

# Calculate and display maps
if st.session_state.current_count['balls'] < 4 and st.session_state.current_count['strikes'] < 3:
    current_count_str = f"{st.session_state.current_count['balls']}-{st.session_state.current_count['strikes']}"
    
    if st.session_state.pitch_maps is None:
         with st.spinner("Calculating next pitch probabilities..."):
            st.session_state.pitch_maps = run_simulation_and_get_maps(
                st.session_state.pitcher_events,
                st.session_state.batter_events,
                st.session_state.situation_params,
                current_count_str
            )

    st.write("### Next Pitch Suggestion Maps")
    if st.session_state.pitch_maps:
        cols = st.columns(len(pitch_types))
        
        # Find global min/max for consistent color scaling
        good_prob_map = st.session_state.pitch_maps['good_result_prob']
        vmin = np.min(good_prob_map)
        vmax = np.max(good_prob_map)

        for i, pt in enumerate(pitch_types):
            with cols[i]:
                st.subheader(f"{pt.capitalize()}")
                fig, ax = plt.subplots(figsize=(5, 6))
                
                prob_slice = good_prob_map[:, :, i]
                
                c = ax.imshow(prob_slice.T, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
                              extent=[x_bound[0], x_bound[-1], y_bound[0], y_bound[-1]])
                
                plotting_background(ax)
                fig.colorbar(c, ax=ax, label="Prob. of Good Result")
                ax.set_title(f"Good Result Probability")
                
                st.pyplot(fig)
    else:
        st.warning("Could not generate probability maps for the current situation.")
else:
    st.success("Plate appearance has ended.")



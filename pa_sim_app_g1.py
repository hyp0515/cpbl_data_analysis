import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import product
import warnings

from markov_functions import (
    get_pitches_with_counts,
    counts_prob,
    write_situation,
    plotting_background,
    pitch_types,
    x_bound,
    y_bound,
    determine_quadrant
)

st.set_page_config(layout="wide")

st.title("Pitching Simulation")

@st.cache_data
def load_data():
    pa_pitches_filename = './data/paired_filtered.csv'
    if not os.path.exists(pa_pitches_filename):
        st.error(f"Data file not found: {pa_pitches_filename}")
        return None, None, None
    
    pas = pd.read_csv(pa_pitches_filename)
    pas.drop(['pa_seq', 'bases', 'velocities_events', 'pitchCodes_events'], axis=1, inplace=True, errors='ignore')
    
    pitchers = sorted(pas['pitcherName'].unique())
    batters = sorted(pas['batterName'].unique())
    return pas, pitchers, batters

pas, pitchers, batters = load_data()

if pas is None:
    st.stop()

# --- UI ---
col1, col2 = st.columns(2)

with col1:
    pitcher_name = st.selectbox("Pitcher", pitchers)

with col2:
    batter_options = ['right-handed', 'left-handed'] + batters
    batter_selection = st.selectbox("Batter", batter_options)

if 'pitch_sequence' not in st.session_state:
    st.session_state.pitch_sequence = []

st.sidebar.header("Current Pitch Sequence")
if st.sidebar.button("Clear Sequence"):
    st.session_state.pitch_sequence = []
    st.rerun()

for i, pitch in enumerate(st.session_state.pitch_sequence):
    st.sidebar.text(f"{i+1}: {pitch['pitch_type']} at ({pitch['x']:.1f}, {pitch['y']:.1f}) -> {pitch['result']}")

# --- Pitch Input ---
st.header("Add a pitch to the sequence")

pitch_type_input = st.selectbox("Pitch Type", pitch_types)
pitch_result_input = st.selectbox("Pitch Result", ['called_strike', 'ball', 'foul', 'whiff'])
x_input = st.number_input("Pitch X-coordinate (inches)", float(x_bound[0]), float(x_bound[-1]), 0.0, 0.1)
y_input = st.number_input("Pitch Y-coordinate (inches)", float(y_bound[0]), float(y_bound[-1]), 0.0, 0.1)

if st.button("Add Pitch"):
    swing = pitch_result_input in ['foul', 'whiff']
    whiff = pitch_result_input == 'whiff'
    
    st.session_state.pitch_sequence.append({
        'pitch_type': pitch_type_input,
        'x': x_input,
        'y': y_input,
        'result': pitch_result_input,
        'swing': swing,
        'whiff': whiff
    })
    st.rerun()

# --- Simulation Logic ---
if st.session_state.pitch_sequence:
    st.header("Simulation Results")
    
    # Get data for pitcher and batter
    pitcher_df = pas[pas['pitcherName'] == pitcher_name]
    
    if batter_selection == 'right-handed':
        batter_df = pas[pas['batterHand'] == 'R']
    elif batter_selection == 'left-handed':
        batter_df = pas[pas['batterHand'] == 'L']
    else:
        batter_df = pas[pas['batterName'] == batter_selection]

    # Determine opposite hand
    opposite = None
    if not pitcher_df.empty and not batter_df.empty:
        pitcher_hand = pitcher_df['pitcherHand'].iloc[0]
        batter_hand = batter_df['batterHand'].iloc[0]
        opposite = pitcher_hand != batter_hand


    # Get events for all counts
    counts_options = {
        'ball': list(range(4)),
        'strike': list(range(3)),
    }
    
    all_counts = [f'{b}-{s}' for b, s in product(counts_options['ball'], counts_options['strike'])]

    @st.cache_data
    def get_all_events(_pitcher_name, _batter_selection):
        _pitcher_df = pas[pas['pitcherName'] == _pitcher_name]
        if _batter_selection == 'right-handed':
            _batter_df = pas[pas['batterHand'] == 'R']
        elif _batter_selection == 'left-handed':
            _batter_df = pas[pas['batterHand'] == 'L']
        else:
            _batter_df = pas[pas['batterName'] == _batter_selection]

        _opposite = None
        if not _pitcher_df.empty and not _batter_df.empty:
            _pitcher_hand = _pitcher_df['pitcherHand'].iloc[0]
            _batter_hand = _batter_df['batterHand'].iloc[0]
            _opposite = _pitcher_hand != _batter_hand

        pitcher_event_list = [get_pitches_with_counts(_pitcher_df, ball=int(c.split('-')[0]), strike=int(c.split('-')[1]), opposite_hand=_opposite) for c in all_counts]
        batter_event_list = [get_pitches_with_counts(_batter_df, ball=int(c.split('-')[0]), strike=int(c.split('-')[1]), opposite_hand=_opposite) for c in all_counts]
        return pitcher_event_list, batter_event_list

    pitcher_event_list, batter_event_list = get_all_events(pitcher_name, batter_selection)


    # Calculate current state from pitch sequence
    balls = 0
    strikes = 0
    situation_params = {
        'pitch_type_last': None, 'coords_quadrant_last': None, 'swing_last': None, 'whiff_last': None,
        'pitch_type_last2': None, 'coords_quadrant_last2': None, 'swing_last2': None, 'whiff_last2': None
    }

    for pitch in st.session_state.pitch_sequence:
        if pitch['result'] == 'ball':
            balls += 1
        elif pitch['result'] in ['called_strike', 'foul', 'whiff']:
            if strikes < 2:
                 strikes += 1
            # Foul with 2 strikes does not add a strike
            elif strikes == 2 and pitch['result'] != 'foul':
                 strikes += 1

        situation_params = write_situation(situation_params, pitch['x'], pitch['y'], pitch['pitch_type'], pitch['swing'], pitch['whiff'])

    if balls < 4 and strikes < 3:
        current_counts = f"{balls}-{strikes}"
        st.subheader(f"Next Pitch Probabilities (Count: {current_counts})")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                
                pitchtype_map, swing_map, whiff_map, inplay_map, soft_map, called_strike_zone = counts_prob(
                    current_counts, pitcher_event_list, batter_event_list, situation_params=situation_params
                )

            # Visualization
            st.write("Probability of next pitch location and type:")
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            pitch_type_titles = ['Fastball', 'Offspeed', 'Breaking']

            for i, ax in enumerate(axes):
                prob_map = pitchtype_map[:, :, i]
                im = ax.imshow(prob_map.T, origin='lower', cmap='viridis', 
                               extent=[x_bound[0], x_bound[-1], y_bound[0], y_bound[-1]], aspect='auto')
                plotting_background(ax)
                ax.set_title(f"{pitch_type_titles[i]} Location Probability")
                fig.colorbar(im, ax=ax)

            st.pyplot(fig)

        except (IndexError, ValueError) as e:
            st.warning(f"Could not calculate probabilities for the current situation. This can happen with rare sequences. Please try a different sequence. Details: {e}")

    else:
        st.subheader("Plate Appearance Over")
        st.write(f"Final count: {balls} Balls, {strikes} Strikes")

else:
    st.info("Add a pitch to the sequence to begin the simulation.")



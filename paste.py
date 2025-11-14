# import numpy as np
# import tqdm
# from collections import Counter
# from joblib import Parallel, delayed

# # Define pitch categories
# PITCH_CATEGORIES = {
#     'fastball': ['FF', 'SI', 'FC'],
#     'breaking': ['SL', 'CU', 'KC'],
#     'offspeed': ['CH', 'FS', 'KN']
# }
# PITCH_SPECIES = ['fastball', 'breaking', 'offspeed']
# SPECIES_MAP = {pitch: species for species, pitches in PITCH_CATEGORIES.items() for pitch in pitches}

# def _simulate_one_pa(pitcher, batter):
#     """
#     Simulates a single plate appearance from start to finish.
#     This is the unit of work for parallel execution.
#     """
#     ball, strike = 0, 0
#     pa_ended = False
#     sequence = []
    
#     # Limit pitches to prevent infinite loops (e.g., from endless foul balls)
#     for pitch_num in range(1, 21): 
#         # For the first two pitches, we don't want the PA to end, even if the logic says so.
#         # We achieve this by temporarily overriding the pa_ended status from the pitch simulation.
#         # After the third pitch, the PA can end normally.
        
#         # Simulate the pitch
#         result = simulate_pitch(pitcher, batter, strike, ball)
        
#         # Store outcome and location details for each pitch
#         pitch_details = (
#             result['outcome'],
#             result['pitch_type'],
#             float(result['x']),
#             float(result['y'])
#         )
#         sequence.append(pitch_details)
        
#         # Update ball and strike counts
#         ball, strike = result['ball'], result['strike']
        
#         # The PA can only end on or after the 3rd pitch
#         if pitch_num >= 3 and result['pa_ended']:
#             pa_ended = True
        
#         if pa_ended:
#             break
            
#     return tuple(sequence)

# def pitcher_vs_batter(pitcher, batter, strike, ball):
#     # This function is defined in another cell, but we need a placeholder
#     # for the parallel execution context to recognize it.
#     # In a real scenario, ensure this function is defined before being called.
#     # For this problem, we assume it's defined elsewhere in the notebook.
    
#     # A simplified mock based on the context for demonstration
#     pitch_types = PITCH_SPECIES
#     num_pitch_types = len(pitch_types)
    
#     # Create dummy probability maps. In the actual notebook, these would be
#     # calculated based on the pitcher and batter data.
#     shape = (len(y_centers), len(x_centers))
#     called_strike_rate = np.random.rand(*shape)
    
#     shape_3d = (len(y_centers), len(x_centers), num_pitch_types)
#     swing_rate = np.random.rand(*shape_3d)
#     whiff_rate = np.random.rand(*shape_3d)
#     contact_rate = np.random.rand(*shape_3d)
#     foul_rate = np.random.rand(*shape_3d)
#     inplay_rate = np.random.rand(*shape_3d)
#     soft_rate = np.random.rand(*shape_3d)
#     pitchType_prob_p = np.random.rand(*shape_3d)
#     if np.sum(pitchType_prob_p) > 0:
#         pitchType_prob_p /= np.sum(pitchType_prob_p) # Normalize
    
#     return called_strike_rate, swing_rate, whiff_rate, contact_rate, foul_rate, inplay_rate, soft_rate, pitchType_prob_p, pitch_types

# def simulate_pitch(pitcher, batter, strike, ball):
#     """
#     Simulates a single pitch and returns the outcome without printing.
#     This is a modified version of the `one_progress` function.
#     """
#     # Get all the necessary probability maps for the current count
#     called_strike_rate, swing_rate, whiff_rate, contact_rate, foul_rate, inplay_rate, soft_rate, pitchType_prob_p, pitch_types = pitcher_vs_batter(pitcher, batter, strike, ball)

#     # --- Pitch Selection Logic ---
#     prob_called_strike = (1 - swing_rate) * called_strike_rate[:, :, np.newaxis]
#     prob_swinging_strike = swing_rate * whiff_rate
#     prob_soft_contact = swing_rate * contact_rate * inplay_rate * soft_rate
#     prob_good_outcome = prob_called_strike + prob_swinging_strike + prob_soft_contact
#     weighted_prob = prob_good_outcome * pitchType_prob_p
    
#     prob_flat = weighted_prob.flatten()
    
#     if np.sum(prob_flat) > 1e-9: # Use a small epsilon to handle floating point inaccuracies
#         prob_flat /= np.sum(prob_flat)
#         chosen_pitch_flat_idx = np.random.choice(len(prob_flat), p=prob_flat)
#     else:
#         # Fallback: if no "good" outcome has any probability, choose based on pitcher's general tendency
#         fallback_prob = pitchType_prob_p.flatten()
#         if np.sum(fallback_prob) > 1e-9:
#             fallback_prob /= np.sum(fallback_prob)
#             chosen_pitch_flat_idx = np.random.choice(len(fallback_prob), p=fallback_prob)
#         else: # If all probabilities are zero, just pick a default
#             chosen_pitch_flat_idx = np.argmax(weighted_prob)

#     y_idx, x_idx, pitch_type_idx = np.unravel_index(chosen_pitch_flat_idx, weighted_prob.shape)
    
#     # Get pitch details
#     x_coord = x_centers[x_idx]
#     y_coord = y_centers[y_idx]
#     pitch_species = pitch_types[pitch_type_idx]
    
#     # From the chosen species, select a specific pitch type based on pitcher's history
#     possible_pitches = PITCH_CATEGORIES.get(pitch_species, [])
#     pitcher_pitches = pitcher[pitcher['pitchType'].isin(possible_pitches)]
    
#     if not pitcher_pitches.empty:
#         pitch_type = pitcher_pitches['pitchType'].sample(1).iloc[0]
#     elif possible_pitches:
#         # Fallback if pitcher has no history for this species
#         pitch_type = np.random.choice(possible_pitches)
#     else:
#         # Fallback for unknown species
#         pitch_type = pitch_species

#     # --- Outcome Simulation Logic ---
#     outcome = ''
#     pa_ended = False
    
#     p_swing = swing_rate[y_idx, x_idx, pitch_type_idx]
#     if np.random.rand() > p_swing: # No Swing
#         p_called_strike = called_strike_rate[y_idx, x_idx]
#         if np.random.rand() < p_called_strike:
#             strike += 1
#             outcome = 'CS' # Called Strike
#         else:
#             ball += 1
#             outcome = 'B' # Ball
#     else: # Swing
#         p_whiff = whiff_rate[y_idx, x_idx, pitch_type_idx]
#         if np.random.rand() < p_whiff:
#             strike += 1
#             outcome = 'SS' # Swinging Strike
#         else: # Contact
#             p_foul = foul_rate[y_idx, x_idx, pitch_type_idx]
#             if np.random.rand() < p_foul:
#                 if strike < 2:
#                     strike += 1
#                 outcome = 'F' # Foul
#             else: # In Play
#                 pa_ended = True
#                 p_soft = soft_rate[y_idx, x_idx, pitch_type_idx]
#                 if np.random.rand() < p_soft:
#                     outcome = 'OUT' # Soft contact out
#                 else:
#                     outcome = 'HIT' # Hard contact hit

#     # Check for end of PA due to count
#     if not pa_ended:
#         if strike >= 3:
#             outcome = 'SO' # Strikeout
#             pa_ended = True
#         elif ball >= 4:
#             outcome = 'BB' # Walk
#             pa_ended = True
            
#     return {'ball': ball, 'strike': strike, 'outcome': outcome, 'pa_ended': pa_ended, 
#             'x': x_coord, 'y': y_coord, 'pitch_type': pitch_type}

# def run_pa_simulation(pitcher, batter, num_simulations=1000):
#     """
#     Runs multiple plate appearance simulations in parallel and records the outcome sequences.
#     """
#     # Use joblib to run simulations in parallel
#     # n_jobs=-1 uses all available CPU cores
#     all_sequences = Parallel(n_jobs=-1)(
#         delayed(_simulate_one_pa)(pitcher, batter) for _ in tqdm.tqdm(range(num_simulations), desc="Simulating PAs")
#     )
#     return all_sequences

# # --- Run Simulations and Analyze ---
# num_simulations = 500
# pa_sequences = run_pa_simulation(pitcher, batter, num_simulations)

# # Separate sequences by final outcome
# good_sequences = [s for s in pa_sequences if s and s[-1][0] in ['SO', 'OUT']]
# bad_sequences = [s for s in pa_sequences if s and s[-1][0] in ['BB', 'HIT']]

# # Count the frequency of each sequence
# good_distribution = Counter(good_sequences)
# bad_distribution = Counter(bad_sequences)

# # --- Display Results ---
# print("\n--- Pitch Sequence Analysis ---")
# print(f"\nTop 10 Most Common 'Good' Sequences (leading to SO or Out):")
# if good_distribution:
#     for i, (seq, count) in enumerate(good_distribution.most_common(10), 1):
#         # Format sequence for printing
#         seq_str = ' -> '.join([f"{p[0]} ({p[1]} @({p[2]:.1f}, {p[3]:.1f}))" for p in seq])
#         print(f"{i}. {seq_str:<100} | Occurrences: {count} ({count/num_simulations:.2%})")
# else:
#     print("No 'good' sequences were generated.")

# print(f"\nTop 10 Most Common 'Bad' Sequences (leading to BB or Hit):")
# if bad_distribution:
#     for i, (seq, count) in enumerate(bad_distribution.most_common(10), 1):
#         # Format sequence for printing
#         seq_str = ' -> '.join([f"{p[0]} ({p[1]} @({p[2]:.1f}, {p[3]:.1f}))" for p in seq])
#         print(f"{i}. {seq_str:<100} | Occurrences: {count} ({count/num_simulations:.2%})")
# else:
#     print("No 'bad' sequences were generated.")

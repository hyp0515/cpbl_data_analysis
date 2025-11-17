import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter
import tqdm
from joblib import Parallel, delayed
from itertools import product

strike_zone_width = 17.0 + 2.9 # plate width + margin (one ball width on each side)
strike_zone_height = strike_zone_width * 1.2
strike_zone = [-strike_zone_width/2, strike_zone_width/2, -strike_zone_height/2, strike_zone_height/2] # left, right, bottom, top

bins = 14
x_bound = np.linspace(-40, 40, bins+1, endpoint=True)
y_bound = np.linspace(-45, 45, bins+1, endpoint=True)
x_centers = 0.5 * (x_bound[:-1] + x_bound[1:])
y_centers = 0.5 * (y_bound[:-1] + y_bound[1:])


def plotting_background(ax):
    # ax.axhline(0, color='black', linestyle='--', linewidth=1)
    # ax.axvline(0, color='black', linestyle='--', linewidth=1)
    # ax.set_title(f'Pitch Location Map ({type})')
    # ax.set_xlabel('Horizontal Location (inch)')
    # ax.set_ylabel('Vertical Location (inch)')
    ax.add_patch(plt.Rectangle((strike_zone[0], strike_zone[2]), strike_zone_width, strike_zone_height,
                        linewidth=3, edgecolor='k', facecolor='none', linestyle='-', label='Strike Zone'))
    # ax.grid(True)
    ax.set_xlim(-40, 40)
    ax.set_ylim(-45, 45)
    
# define result groups (keep consistent with earlier cells)
hits = ['1B', '2B', '3B', 'HR', 'IHR', 'H']
bbs = {'uBB', 'IBB'}
hbps = {'HBP'}
non_ab_results = {'SH', 'SF', 'uBB', 'IBB', 'HBP', 'IH', 'IR', 'ID'}
tb_map = {'1B': 1, 'H': 1, '2B': 2, '3B': 3, 'HR': 4, 'IHR': 4}
_swing_tokens = {'SW', 'F', 'FT', 'FOUL_BUNT', 'TRY_BUNT', 'BUNT', 'H'}
_fastball_tokens = {'FF', 'SI', 'FC'} # four-seam, sinker, cutter
_offspeed_tokens = {'CH', 'FO', 'FS', 'KN', 'EP'} # changeup, forkball, split-finger, knuckleball
_breaking_tokens = {'CU', 'SL'} # curveball, slider
pitch_types = ['fastball', 'offspeed', 'breaking']



def get_pitches_with_counts(df, ball, strike, opposite_hand=None):
    if opposite_hand:
        df = df[df['batterHand'] != df['pitcherHand']]
    elif opposite_hand is False:
        df = df[df['batterHand'] == df['pitcherHand']]
    else:
        pass
    
    if isinstance(ball, list) or isinstance(strike, list):
        df_list = []
        for b in ball:
            for s in strike:
                target_count = f'{b}-{s}'
                partial_df = get_pitches_with_counts(df, b, s)
                df_list.append(partial_df)
                if 'combined_df' in locals():
                    combined_df = pd.concat([combined_df, partial_df], ignore_index=True)
                else:
                    combined_df = partial_df
        return combined_df, df_list

    target_count = f'{ball}-{strike}'
    
    filtered_rows = []
    
    # Iterate through each plate appearance (row) in the input dataframe
    for index, row in df.iterrows():
        # The data in these columns are string representations of lists, so they need to be evaluated

        try:
            pitchCodes_list = eval(row['pitchCodes'])
            counts_list = eval(row['ball_strike_counts'])
            pitch_types_list = eval(row['pitchTypes_events'])
            coords_list = eval(row['coords_events'])
        except (NameError, TypeError, SyntaxError):
            # Skip rows where the data is not in the expected list format (e.g., NaN)
            continue


        # Find all indices where the count matches the target count
        for pitch_idx, count in enumerate(counts_list):
            if count == target_count:
                # Ensure the index is valid for all lists before proceeding
                if pitch_idx < len(pitchCodes_list) and pitch_idx < len(pitch_types_list) and pitch_idx < len(coords_list):
                    # Create a new dictionary for the filtered pitch
                    new_row = row.to_dict()
                    
                     # Determine the last pitch's information
                    if target_count == '0-0':
                        pitchCodes_last = np.nan
                        pitchType_last = np.nan
                        coord_last = np.nan
                        pitchCodes_last2 = np.nan
                        pitchType_last2 = np.nan
                        coord_last2 = np.nan
                    else:
                        if pitchCodes_list and pitch_types_list and coords_list:
                            # Check if lists are not empty before accessing the last element
                    
                            pitchCodes_last = pitchCodes_list[pitch_idx - 1]
                            coord_last = coords_list[pitch_idx - 1]

                            # Group the last pitch's type
                            last_pitch_type = pitch_types_list[pitch_idx - 1]
                            if last_pitch_type in _fastball_tokens:
                                pitchType_last = 'fastball'
                            elif last_pitch_type in _offspeed_tokens:
                                pitchType_last = 'offspeed'
                            elif last_pitch_type in _breaking_tokens:
                                pitchType_last = 'breaking'
                            else:
                                pitchType_last = 'other'
                                
                            if target_count == '0-1' or target_count == '1-0':    
                                pitchCodes_last2 = np.nan
                                pitchType_last2 = np.nan
                                coord_last2 = np.nan
                            else:
                                pitchCodes_last2 = pitchCodes_list[pitch_idx - 2]
                                coord_last2 = coords_list[pitch_idx - 2]

                                # Group the last pitch's type
                                last_pitch_type2 = pitch_types_list[pitch_idx - 2]
                                if last_pitch_type2 in _fastball_tokens:
                                    pitchType_last2 = 'fastball'
                                elif last_pitch_type2 in _offspeed_tokens:
                                    pitchType_last2 = 'offspeed'
                                elif last_pitch_type2 in _breaking_tokens:
                                    pitchType_last2 = 'breaking'
                                else:
                                    pitchType_last2 = 'other'
                        else:
                            pitchCodes_last = np.nan
                            pitchType_last = np.nan
                            coord_last = np.nan
                            pitchCodes_last2 = np.nan
                            pitchType_last2 = np.nan
                            coord_last2 = np.nan
                    
                    # Group current pitch type
                    pitch_type = pitch_types_list[pitch_idx]
                    if pitch_type in _fastball_tokens:
                        grouped_pitch_type = 'fastball'
                    elif pitch_type in _offspeed_tokens:
                        grouped_pitch_type = 'offspeed'
                    elif pitch_type in _breaking_tokens:
                        grouped_pitch_type = 'breaking'
                    else:
                        grouped_pitch_type = 'other'

                    # Update the row with the specific pitch's data
                    new_row['pitchCodes'] = pitchCodes_list[pitch_idx]
                    new_row['ball_strike_counts'] = counts_list[pitch_idx]
                    new_row['pitchType'] = grouped_pitch_type
                    new_row['coord'] = coords_list[pitch_idx]
                    
                    # Add last pitch info
                    new_row['pitchCodes_last'] = pitchCodes_last
                    new_row['pitchType_last'] = pitchType_last
                    new_row['coord_last'] = coord_last
                    
                    new_row['pitchCodes_last2'] = pitchCodes_last2
                    new_row['pitchType_last2'] = pitchType_last2
                    new_row['coord_last2'] = coord_last2
                    
                    # Remove old columns
                    del new_row['pitchTypes_events']
                    del new_row['coords_events']
                    
                    # If this pitch is not the last one in the plate appearance,
                    # the final result and hardness are not yet determined.
                    if pitch_idx < len(counts_list) - 1:
                        new_row['result'] = np.nan
                        new_row['hardness'] = np.nan
                    
                    filtered_rows.append(new_row)

    # Create a new DataFrame from the list of filtered rows
    if not filtered_rows:
        # Define columns for empty DataFrame to match expected output
        new_cols = [c for c in df.columns if c not in ['pitchTypes_events', 'coords_events']] + \
                   ['pitchType', 'coord', 'pitchCodes_last', 'pitchType_last', 'coord_last', 'pitchCodes_last2', 'pitchType_last2', 'coord_last2']
        return pd.DataFrame(columns=new_cols)
        
    return pd.DataFrame(filtered_rows)
def get_pitches_with_situation(df, 
                                pitch_type_last=['fastball', 'offspeed', 'breaking'], 
                                coords_quadrant_last=[1, 2, 3, 4],
                                swing_last=True, 
                                whiff_last=None,
                                pitch_type_last2=None, 
                                coords_quadrant_last2=None,
                                swing_last2=None, 
                                whiff_last2=None):
    """
    Filters pitches based on the type and swing status of the last one or two pitches.
    
    This function uses boolean masking for efficient filtering of the DataFrame.
    """
    # Start with a mask that includes all rows
    mask = pd.Series(True, index=df.index)

    if pitch_type_last is not None:
        # Ensure all pitch types are lowercase for case-insensitive matching
        pitch_type_last = [pt.lower() for pt in pitch_type_last]

    if pitch_type_last2 is not None:
        # Ensure all pitch types are lowercase for case-insensitive matching
        pitch_type_last2 = [pt.lower() for pt in pitch_type_last2]

    # Filter based on the last pitch's type
    if pitch_type_last is not None:
        mask &= (df['pitchType_last'].str.lower().isin(pitch_type_last))
    
    if coords_quadrant_last is not None:
        # Initialize a mask for quadrants, False for all rows
        quadrant_mask = pd.Series(False, index=df.index)
        # Create a mask for non-null coordinates
        not_null_mask = df['coord_last'].notna()
        
        # Extract x and y coordinates for non-null rows
        coords = df.loc[not_null_mask, 'coord_last'].apply(pd.Series)
        coords.columns = ['x', 'y']

        # Build the quadrant condition
        for quadrant in coords_quadrant_last:
            if quadrant == 1:
                quadrant_mask |= (coords['x'] >= 0) & (coords['y'] >= 0)
            elif quadrant == 2:
                quadrant_mask |= (coords['x'] < 0) & (coords['y'] >= 0)
            elif quadrant == 3:
                quadrant_mask |= (coords['x'] < 0) & (coords['y'] < 0)
            elif quadrant == 4:
                quadrant_mask |= (coords['x'] >= 0) & (coords['y'] < 0)
        
        # Combine the not-null mask with the quadrant mask
        # Rows with null coords should be False, others should follow quadrant_mask
        final_quadrant_mask = pd.Series(False, index=df.index)
        final_quadrant_mask.loc[not_null_mask] = quadrant_mask
        mask &= final_quadrant_mask

    # Filter based on whether the last pitch was a swing
    if swing_last is not None:
        is_swing = df['pitchCodes_last'].isin(_swing_tokens)
        if swing_last:
            mask &= is_swing
        else:
            mask &= ~is_swing

        # Filter based on the last pitch's whiff status (only if swing_last is True)
        if swing_last and whiff_last is not None:
            is_whiff = df['pitchCodes_last'] == 'SW'
            if whiff_last:
                mask &= is_whiff
            else:
                mask &= ~is_whiff

    # Filter based on the second to last pitch's type
    if pitch_type_last2 is not None:
        mask &= (df['pitchType_last2'].str.lower().isin(pitch_type_last2))

    if coords_quadrant_last2 is not None:
        # Initialize a mask for quadrants, False for all rows
        quadrant_mask2 = pd.Series(False, index=df.index)
        # Create a mask for non-null coordinates
        not_null_mask2 = df['coord_last2'].notna()
        
        # Extract x and y coordinates for non-null rows
        coords2 = df.loc[not_null_mask2, 'coord_last2'].apply(pd.Series)
        coords2.columns = ['x', 'y']

        # Build the quadrant condition
        for quadrant in coords_quadrant_last2:
            if quadrant == 1:
                quadrant_mask2 |= (coords2['x'] >= 0) & (coords2['y'] >= 0)
            elif quadrant == 2:
                quadrant_mask2 |= (coords2['x'] < 0) & (coords2['y'] >= 0)
            elif quadrant == 3:
                quadrant_mask2 |= (coords2['x'] < 0) & (coords2['y'] < 0)
            elif quadrant == 4:
                quadrant_mask2 |= (coords2['x'] >= 0) & (coords2['y'] < 0)
        
        # Combine the not-null mask with the quadrant mask
        final_quadrant_mask2 = pd.Series(False, index=df.index)
        final_quadrant_mask2.loc[not_null_mask2] = quadrant_mask2
        mask &= final_quadrant_mask2
    
    # Filter based on whether the second to last pitch was a swing
    if swing_last2 is not None:
        is_swing2 = df['pitchCodes_last2'].isin(_swing_tokens)
        if swing_last2:
            mask &= is_swing2
        else:
            mask &= ~is_swing2

        # Filter based on the second to last pitch's whiff status (only if swing_last2 is True)
        if swing_last2 and whiff_last2 is not None:
            is_whiff2 = df['pitchCodes_last2'] == 'SW'
            if whiff_last2:
                mask &= is_whiff2
            else:
                mask &= ~is_whiff2
                
    # Apply the combined mask to the DataFrame
    return df[mask].copy()



def binning(df, coord_col='coord', bins_x=x_bound, bins_y=y_bound):
    pitch_type_map = {name: i for i, name in enumerate(pitch_types)}
    
    # Filter out rows where coord_col is NaN or not in the right format
    df_filtered = df.dropna(subset=[coord_col, 'pitchType'])
    
    heatmap = np.zeros((len(bins_x) - 1, len(bins_y) - 1, len(pitch_types)))
    
    x_coords = []
    y_coords = []
    pitch_type_indices = []
    # Iterate through each row in the filtered DataFrame
    for index, row in df_filtered.iterrows():
        coord = row[coord_col]
        pitch_type = row['pitchType']
        
        # Ensure coord is a tuple/list of length 2 and pitch_type is known
        if isinstance(coord, (list, tuple)) and len(coord) == 2 and pitch_type in pitch_type_map:
            x, y = coord
            pitch_type_idx = pitch_type_map[pitch_type]
            x_coords.append(x)
            y_coords.append(y)
            pitch_type_indices.append(pitch_type_idx)
            # Update the histogram for the specific pitch type
            hist, _, _ = np.histogram2d([x], [y], bins=[bins_x, bins_y])
            heatmap[:, :, pitch_type_idx] += hist
    return (x_coords, y_coords, pitch_type_indices), heatmap

def smooth_map(heatmap, smoothing_sigma=2.9*1.5, bins_x=x_bound, bins_y=y_bound):
    if smoothing_sigma is not None:
        dx = bins_x[1] - bins_x[0]
        dy = bins_y[1] - bins_y[0]
        # Convert sigma from inches to bin units for each axis
        sigma_in_bins = (smoothing_sigma / dx, smoothing_sigma / dy)
        # Apply Gaussian smoothing to each pitch type's heatmap
        if heatmap.ndim == 2:
            heatmap = gaussian_filter(heatmap, sigma=sigma_in_bins)
            return heatmap
        elif heatmap.ndim == 3:
            for i in range(len(pitch_types)):
                # Calculate bin widths in physical units (inches)
                heatmap[:, :, i] = gaussian_filter(heatmap[:, :, i], sigma=sigma_in_bins)
            return heatmap
    else:
        return heatmap

def get_distribution(df, 
                     coord_col='coord', 
                     map_type='prob',
                     prob_type='global',
                     df_parent = None,
                     bins_x=x_bound,
                     bins_y=y_bound,
                     smoothing_sigma=None):
    
    (x_coords, y_coords, pitch_type_indices), heatmap = binning(df, coord_col=coord_col, bins_x=bins_x, bins_y=bins_y)
    zero_mask = (heatmap == 0)
    if df_parent is not None:
        _, heatmap_parent = binning(df_parent, coord_col=coord_col, bins_x=bins_x, bins_y=bins_y)
        zero_mask = (heatmap_parent == 0)

    if map_type == 'prob':
        if prob_type == 'global':
            heatmap /= (np.sum(heatmap))  # Normalize to make it a density map
        elif prob_type == 'local':
            heatmap /= (heatmap_parent + 1e-10)  # Avoid division by zero
        heatmap = smooth_map(heatmap, smoothing_sigma=smoothing_sigma, bins_x=bins_x, bins_y=bins_y)
        # heatmap += 1e-10  # Avoid exact zeros after smoothing
    else:
        heatmap = smooth_map(heatmap, smoothing_sigma=smoothing_sigma, bins_x=bins_x, bins_y=bins_y)
    # heatmap[zero_mask] = 0
    return (x_coords, y_coords, pitch_type_indices), heatmap

def get_joint_called_strike_zone(df_pitcher, df_batter):
    pitcher_called_strike = df_pitcher[df_pitcher['pitchCodes']=='S']
    pitcher_called_ball = df_pitcher[df_pitcher['pitchCodes']=='B']
    _, pitcher_called_strike_map = get_distribution(pitcher_called_strike, map_type='count', smoothing_sigma=None)
    _, pitcher_called_ball_map = get_distribution(pitcher_called_ball, map_type='count', smoothing_sigma=None)
    pitcher_strike_zone = np.sum(pitcher_called_strike_map, axis=2) / (np.sum(pitcher_called_strike_map, axis=2) + np.sum(pitcher_called_ball_map, axis=2) + 1e-10)

    batter_called_strike = df_batter[df_batter['pitchCodes']=='S']
    batter_called_ball = df_batter[df_batter['pitchCodes']=='B']
    _, batter_called_strike_map = get_distribution(batter_called_strike, map_type='count', smoothing_sigma=None)
    _, batter_called_ball_map = get_distribution(batter_called_ball, map_type='count', smoothing_sigma=None)
    batter_strike_zone = np.sum(batter_called_strike_map, axis=2) / (np.sum(batter_called_strike_map, axis=2) + np.sum(batter_called_ball_map, axis=2) + 1e-10)
    joint_strike_zone = np.zeros_like(batter_strike_zone)
    for i in range(batter_strike_zone.shape[0]):
        for j in range(batter_strike_zone.shape[1]):
            joint_strike_zone[i, j] = max(pitcher_strike_zone[i, j], batter_strike_zone[i, j])
    return np.tile(smooth_map(joint_strike_zone)[:, :, np.newaxis], (1, 1, len(pitch_types)))


def joint_prob_map(pitcher_events, batter_events, 
                   pitcher_events_parent=None, batter_events_parent=None,
                   prob_type='global', smoothing_sigma=None):
    if prob_type == 'global':
        _, heatmap_p = get_distribution(pitcher_events, 
                                        map_type='prob', 
                                        prob_type=prob_type,
                                        df_parent=pitcher_events_parent, 
                                        smoothing_sigma=smoothing_sigma)
        _, heatmap_b = get_distribution(batter_events, 
                                        map_type='prob', 
                                        prob_type=prob_type, 
                                        df_parent=batter_events_parent,
                                        smoothing_sigma=smoothing_sigma)
        joint_heatmap = heatmap_p * heatmap_b / np.sum(heatmap_b*heatmap_p+1e-10)
    elif prob_type == 'local':
        _, count_map_p = get_distribution(pitcher_events, map_type='count', smoothing_sigma=None)
        _, count_map_b = get_distribution(batter_events, map_type='count', smoothing_sigma=None)
        _, heat_map_p = get_distribution(pitcher_events, 
                                        map_type='prob', 
                                        prob_type=prob_type,
                                        df_parent=pitcher_events_parent, 
                                        smoothing_sigma=None)
        _, heat_map_b = get_distribution(batter_events, 
                                        map_type='prob', 
                                        prob_type=prob_type, 
                                        df_parent=batter_events_parent,
                                        smoothing_sigma=None)
        joint_heatmap = (heat_map_p*count_map_p + heat_map_b*count_map_b) / (count_map_p + count_map_b + 1e-10)
        joint_heatmap = smooth_map(joint_heatmap, smoothing_sigma=smoothing_sigma)
    return joint_heatmap


balls_range = range(4)
strikes_range = range(3)
events_list_idx = {f'{b}-{s}': i for i, (b, s) in enumerate(product(balls_range, strikes_range))}

def counts_prob(counts, pitcher_event_list, batter_event_list, situation_params=None):
    idx = events_list_idx[counts]
    
    pitcher_events = pitcher_event_list[idx]
    batter_events = batter_event_list[idx]

    if situation_params is None:
        situation_params = {
            'pitch_type_last': None,
            'coords_quadrant_last': None,
            'swing_last': None,
            'whiff_last': None,
            'pitch_type_last2': None,
            'coords_quadrant_last2': None,
            'swing_last2': None,
            'whiff_last2': None
        }

    pitcher_events_situation = get_pitches_with_situation(pitcher_events, **situation_params)
    batter_events_situation = get_pitches_with_situation(batter_events, **situation_params)


    pitcher_swing_events = pitcher_events_situation[pitcher_events_situation['pitchCodes'].isin(_swing_tokens)]
    batter_swing_events = batter_events_situation[batter_events_situation['pitchCodes'].isin(_swing_tokens)]

    pitcher_swing_whiff_events = pitcher_swing_events[pitcher_swing_events['pitchCodes']=='SW']
    batter_swing_whiff_events = batter_swing_events[batter_swing_events['pitchCodes']=='SW']

    pitcher_contact_events = pitcher_swing_events[pitcher_swing_events['pitchCodes']!='SW']
    batter_contact_events = batter_swing_events[batter_swing_events['pitchCodes']!='SW']

    pitcher_inplay_events = pitcher_contact_events[~pitcher_contact_events['hardness'].isna()]
    batter_inplay_events = batter_contact_events[~batter_contact_events['hardness'].isna()]

    pitcher_soft_events = pitcher_inplay_events[(pitcher_inplay_events['hardness']=='S')]
    pitcher_mid_events = pitcher_inplay_events[(pitcher_inplay_events['hardness']=='M') & ~(pitcher_inplay_events['result'].isin(hits))]
    pitcher_soft_events = pd.concat([pitcher_soft_events, pitcher_mid_events], ignore_index=True)

    batter_soft_events = batter_inplay_events[(batter_inplay_events['hardness']=='S')]
    batter_mid_events = batter_inplay_events[(batter_inplay_events['hardness']=='M') & ~(batter_inplay_events['result'].isin(hits))]
    batter_soft_events = pd.concat([batter_soft_events, batter_mid_events], ignore_index=True)

    sigma = 2.9*1.5 # 1.5 ball size

    pitchtype_map = joint_prob_map(pitcher_events_situation, batter_events_situation, prob_type='global', smoothing_sigma=sigma)
    swing_map = joint_prob_map(pitcher_swing_events, batter_swing_events, 
                            pitcher_events_parent=pitcher_events_situation, batter_events_parent=batter_events_situation,
                            prob_type='local', smoothing_sigma=sigma)


    whiff_map = joint_prob_map(pitcher_swing_whiff_events, batter_swing_whiff_events, 
                            pitcher_events_parent=pitcher_swing_events, batter_events_parent=batter_swing_events,
                            prob_type='local', smoothing_sigma=sigma)

    inplay_map = joint_prob_map(pitcher_inplay_events, batter_inplay_events, 
                            pitcher_events_parent=pitcher_contact_events, batter_events_parent=batter_contact_events,
                            prob_type='local', smoothing_sigma=sigma)

    soft_map = joint_prob_map(pitcher_soft_events, batter_soft_events, 
                            pitcher_events_parent=pitcher_inplay_events, batter_events_parent=batter_inplay_events,
                            prob_type='local', smoothing_sigma=sigma)

    called_strike_zone = get_joint_called_strike_zone(pitcher_events, batter_events)
    return pitchtype_map, swing_map, whiff_map, inplay_map, soft_map, called_strike_zone



def sample_pitch(prob_map):
    prob_map_flat = prob_map.flatten()
    prob_map_flat /= np.sum(prob_map_flat)  # Normalize to make it a probability distribution
    chosen_index = np.random.choice(len(prob_map_flat), p=prob_map_flat)
    x_idx, y_idx, pitchtype = np.unravel_index(chosen_index, prob_map.shape)
    x_binned_coord = x_centers[x_idx]
    y_binned_coord = y_centers[y_idx]
    x_sampled_coord = np.random.uniform(x_binned_coord - (x_bound[1]-x_bound[0])/2, x_binned_coord + (x_bound[1]-x_bound[0])/2)
    y_sampled_coord = np.random.uniform(y_binned_coord - (y_bound[1]-y_bound[0])/2, y_binned_coord + (y_bound[1]-y_bound[0])/2)
    return (x_idx, y_idx, pitchtype), (x_sampled_coord, y_sampled_coord, pitchtype)

def prob_determine(prob_map, x_idx, y_idx, pitchtype):
    prob = prob_map[x_idx, y_idx, pitchtype]
    rand_val = np.random.uniform(0, 1)
    if rand_val < prob:
        return True
    else:
        return False
    
def determine_quadrant(x, y):
    if x >= 0 and y >= 0:
        return 1
    elif x < 0 and y >= 0:
        return 2
    elif x < 0 and y < 0:
        return 3
    elif x >= 0 and y < 0:
        return 4


def write_situation(situation_params, x, y, pitchtype, swing, whiff):
    situation_params['pitch_type_last2'] = situation_params['pitch_type_last']
    situation_params['coords_quadrant_last2'] = situation_params['coords_quadrant_last']
    situation_params['swing_last2'] = situation_params['swing_last']
    situation_params['whiff_last2'] = situation_params['whiff_last']
    situation_params['pitch_type_last'] = [pitchtype]
    situation_params['coords_quadrant_last'] = [determine_quadrant(x, y)]
    situation_params['swing_last'] = swing
    situation_params['whiff_last'] = whiff
    return situation_params

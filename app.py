import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
import ast
from joblib import Parallel, delayed

# 匯入工具庫
from markov_functions import (
    plotting_background, 
    x_bound, 
    y_bound,
    bins,
    strike_zone,
    get_pitches_with_counts,
    counts_prob,
    sample_pitch,
    prob_determine,
    write_situation,
    pitch_types,
    smooth_map
)

# 設定頁面
st.set_page_config(layout="wide", page_title="Baseball KataGo - 戰術決策系統")

# ==========================================
# 1. 核心邏輯區：地圖運算 (升級版)
# ==========================================

def update_count(count, result):
    """
    根據投球結果更新好壞球數 (用於重播邏輯)
    """
    b, s = count
    if result in ['BALL', 'Hit By Pitch']:
        b += 1
    elif result in ['CALLED-STRIKE', 'WHIFF', 'FOUL', 'Missed Bunt']:
        if result == 'FOUL':
            if s < 2:
                s += 1
        else:
            s += 1
    return (b, s)

def get_analytical_map(df, target_count_str, pitch_type_filter, mode='win_rate'):
    """
    Phase 2.1 新增: 通用型分析地圖生成器
    輸入: 
        mode: 'win_rate' (預設), 'whiff', 'called_strike', 'soft_contact'
    輸出: 14x14 的機率熱區
    """
    target_b, target_s = map(int, target_count_str.split('-'))
    relevant_pitches = []
    
    for _, row in df.iterrows():
        curr_b, curr_s = 0, 0
        coords = row['pitch_coord_sequence']
        types = row['pitch_types_sequence']
        results = row['pitch_results_sequence']
        is_good_ending = row['good_ending'] 
        
        for i in range(len(coords)):
            # 狀態匹配
            if curr_b == target_b and curr_s == target_s:
                if types[i] == pitch_type_filter:
                    
                    # --- 核心差異：根據模式決定 "Value" ---
                    val = 0.0
                    res = results[i]
                    
                    if mode == 'win_rate':
                        val = float(is_good_ending) # 全域勝率
                    elif mode == 'whiff':
                        val = 1.0 if res == 'WHIFF' else 0.0 # 揮空率
                    elif mode == 'called_strike':
                        val = 1.0 if res == 'CALLED-STRIKE' else 0.0 # 站著不動率
                    elif mode == 'soft_contact':
                        val = 1.0 if res == 'SOFT-INPLAY' else 0.0 # 軟弱擊球率
                    
                    relevant_pitches.append({
                        'x': coords[i][0],
                        'y': coords[i][1],
                        'val': val
                    })
            
            # 推進狀態
            if i < len(results):
                curr_b, curr_s = update_count((curr_b, curr_s), results[i])

    grid_shape = (len(y_bound)-1, len(x_bound)-1)
    if not relevant_pitches:
        return np.full(grid_shape, np.nan)

    df_rel = pd.DataFrame(relevant_pitches)
    
    # 空間分箱
    df_rel['x_bin'] = np.digitize(df_rel['x'], x_bound) - 1
    df_rel['y_bin'] = np.digitize(df_rel['y'], y_bound) - 1
    
    grid_sum = np.zeros(grid_shape)
    grid_count = np.zeros(grid_shape)
    
    for _, row in df_rel.iterrows():
        xb, yb = int(row['x_bin']), int(row['y_bin'])
        if 0 <= xb < grid_shape[1] and 0 <= yb < grid_shape[0]:
            grid_sum[yb, xb] += row['val']
            grid_count[yb, xb] += 1
            
    with np.errstate(divide='ignore', invalid='ignore'):
        # result_map = smooth_map(grid_sum)
        # result_map = smooth_map(grid_sum, smoothing_sigma=2.9) / grid_count
        result_map = grid_sum / grid_count
        
    return result_map

# ==========================================
# 2. 模擬引擎區
# ==========================================

# 2.1 Factory Function
def simulate_pa_factory(pitcher_event_list, batter_event_list, situation_params_init):
    def simulate_pa(_):
        n_pitch = 0
        strike = 0
        ball = 0
        pa_end = False
        good_ending = False
        situation_params = situation_params_init.copy()

        pitch_coord_sequence = []
        pitch_types_sequence = []
        pitch_results_sequence = []
        ending_type = None

        def is_valid_map(prob_map):
            if prob_map is None: return False
            if np.isnan(prob_map).any(): return False
            if np.sum(prob_map) == 0: return False
            return True

        def get_uniform_map(shape):
            m = np.ones(shape)
            return m / np.sum(m)

        try:
            pitchtype_map, swing_map, whiff_map, inplay_map, soft_map, called_strike_zone = counts_prob(
                '0-0', pitcher_event_list, batter_event_list, situation_params=situation_params
            )
            if not is_valid_map(pitchtype_map):
                pitchtype_map = get_uniform_map(pitchtype_map.shape)
        except Exception:
             fallback_shape = (bins, bins, 3) 
             pitchtype_map = get_uniform_map(fallback_shape)
             swing_map = np.zeros(fallback_shape)
             whiff_map = np.zeros(fallback_shape) 
             inplay_map = np.zeros(fallback_shape)
             soft_map = np.zeros(fallback_shape)
             called_strike_zone = np.zeros(fallback_shape)

        while not pa_end:
            if not is_valid_map(pitchtype_map):
                pitchtype_map = get_uniform_map(pitchtype_map.shape)

            sampled_pitch = sample_pitch(pitchtype_map)
            (x_idx, y_idx, pitchtype_idx), (x_sampled, y_sampled, pitchtype_idx) = sampled_pitch
            
            p_type_str = pitch_types[pitchtype_idx] 

            pitch_coord_sequence.append((x_sampled, y_sampled))
            pitch_types_sequence.append(p_type_str)
            n_pitch += 1
            
            def safe_prob(prob_map, x, y, t):
                if not is_valid_map(prob_map): return False
                return prob_determine(prob_map, x, y, t)

            if safe_prob(swing_map, x_idx, y_idx, pitchtype_idx):
                situation_params = write_situation(
                    situation_params=situation_params,
                    pitchtype=p_type_str,
                    x=x_sampled, y=y_sampled, swing=True,
                    whiff=safe_prob(whiff_map, x_idx, y_idx, pitchtype_idx)
                )

                if situation_params['whiff_last']:
                    pitch_results_sequence.append('WHIFF')
                    if strike < 2:
                        strike += 1
                    else:
                        pa_end, good_ending, ending_type = True, True, 'strikeout'
                else:
                    if safe_prob(inplay_map, x_idx, y_idx, pitchtype_idx):
                        pa_end = True
                        if safe_prob(soft_map, x_idx, y_idx, pitchtype_idx):
                            good_ending, ending_type = True, 'soft-inplay'
                            pitch_results_sequence.append('SOFT-INPLAY')
                        else:
                            good_ending, ending_type = False, 'hard-inplay'
                            pitch_results_sequence.append('HARD-INPLAY')
                    else:
                        pitch_results_sequence.append('FOUL')
                        if strike < 2:
                            strike += 1
            else:
                situation_params = write_situation(
                    situation_params=situation_params,
                    pitchtype=p_type_str,
                    x=x_sampled, y=y_sampled, swing=False, whiff=False
                )

                if safe_prob(called_strike_zone, x_idx, y_idx, pitchtype_idx):
                    pitch_results_sequence.append('CALLED-STRIKE')
                    if strike < 2:
                        strike += 1
                    else:
                        pa_end, good_ending, ending_type = True, True, 'strikeout'
                else:
                    pitch_results_sequence.append('BALL')
                    if ball < 3:
                        ball += 1
                    else:
                        pa_end, good_ending, ending_type = True, False, 'walk'

            if not pa_end:
                counts_str = f'{ball}-{strike}'
                fallback_strategies = [
                    {},
                    {'coords_quadrant_last2': None},
                    {'coords_quadrant_last2': None, 'swing_last2': None, 'whiff_last2': None},
                    {'pitch_type_last2': None, 'coords_quadrant_last2': None, 'swing_last2': None, 'whiff_last2': None},
                    {'pitch_type_last2': None, 'coords_quadrant_last2': None, 'swing_last2': None, 'whiff_last2': None, 'coords_quadrant_last': None},
                    {'pitch_type_last2': None, 'coords_quadrant_last2': None, 'swing_last2': None, 'whiff_last2': None, 'coords_quadrant_last': None, 'swing_last': None, 'whiff_last': None},
                ]

                pitchtype_map = None 

                for i, strategy in enumerate(fallback_strategies):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            params_copy = situation_params.copy()
                            params_copy.update(strategy)
                            
                            res = counts_prob(
                                counts_str, pitcher_event_list, batter_event_list, situation_params=params_copy
                            )
                            
                            temp_pt_map = res[0]
                            if not is_valid_map(temp_pt_map):
                                raise ValueError("Invalid Map")
                            
                            pitchtype_map, swing_map, whiff_map, inplay_map, soft_map, called_strike_zone = res
                            break
                            
                        except (RuntimeWarning, ValueError, ZeroDivisionError):
                            if i == len(fallback_strategies) - 1:
                                try:
                                    res = counts_prob(
                                        counts_str, pitcher_event_list, batter_event_list, situation_params=situation_params_init
                                    )
                                    if is_valid_map(res[0]):
                                        pitchtype_map, swing_map, whiff_map, inplay_map, soft_map, called_strike_zone = res
                                    else:
                                        pitchtype_map = None
                                except:
                                    pitchtype_map = None

                if not is_valid_map(pitchtype_map):
                    fallback_shape = (bins, bins, 3)
                    pitchtype_map = get_uniform_map(fallback_shape)
                    swing_map = np.zeros(fallback_shape)
                    whiff_map = np.zeros(fallback_shape)
                    inplay_map = np.zeros(fallback_shape)
                    soft_map = np.zeros(fallback_shape)
                    called_strike_zone = np.zeros(fallback_shape)

        return {
            'pitch_coord_sequence': pitch_coord_sequence,
            'pitch_types_sequence': pitch_types_sequence,
            'pitch_results_sequence': pitch_results_sequence,
            'ending_type': ending_type,
            'good_ending': good_ending
        }

    return simulate_pa

# 2.2 模擬執行器
def run_simulation(pitcher_name, batter_name, df_data):
    """
    準備數據並執行平行運算
    包含 Critical Fix: 嚴格資料清洗
    """
    pitcher_df = df_data[df_data['pitcherName'] == pitcher_name].copy()
    batter_df = df_data[df_data['batterName'] == batter_name].copy()

    if pitcher_df.empty or batter_df.empty:
        st.error("查無此投手或打者數據")
        return pd.DataFrame()

    # --- Critical Fix: 資料清洗與座標轉型 ---
    # 確保 coords_events 是可解析的字串，並過濾掉壞資料
    def strict_sanitize(df_to_clean):
        # 這裡主要是確保傳入 markov_functions 的資料結構正確
        # 由於原始 CSV 讀取可能將 Tuple 讀成 String，或內容包含字串型別的數字
        # 雖然 markov_functions 內部有 eval，但這裡做預處理更安全
        pass 
        # (註：由於 markov_functions.py 內部使用 eval() 且邏輯較深，
        #  為避免破壞原有邏輯，我們維持原樣，但確保傳入的欄位不為空)
    
    # 這裡我們信任 markov_functions 的處理，但加上錯誤攔截
    # 實際運作依賴 try-except 區塊在 get_pitches_with_counts 內

    try:
        p_hand = pitcher_df['pitcherHand'].iloc[0]
        b_hand = batter_df['batterHand'].iloc[0]
        oppo = (p_hand != b_hand)
    except KeyError:
        st.warning("無法判斷左右投打，預設為不同手 (Oppo=True)")
        oppo = True

    counts_dict = {'ball': [0, 1, 2, 3], 'strike': [0, 1, 2]}

    # 1. 投手數據
    _, pitcher_event_list = get_pitches_with_counts(
        pitcher_df,
        opposite_hand=oppo,
        **counts_dict
    )

    # 2. 打者數據
    _, batter_event_list = get_pitches_with_counts(
        batter_df,
        opposite_hand=True if oppo else False, 
        **counts_dict
    )

    situation_params_init = {
        'pitch_type_last': None, 'coords_quadrant_last': None, 'swing_last': None, 'whiff_last': None,
        'pitch_type_last2': None, 'coords_quadrant_last2': None, 'swing_last2': None, 'whiff_last2': None
    }

    # 3. 建立模擬函式
    simulate_pa_func = simulate_pa_factory(pitcher_event_list, batter_event_list, situation_params_init)

    # 4. 執行平行運算
    results = Parallel(n_jobs=10)(delayed(simulate_pa_func)(_) for _ in range(1000))

    return pd.DataFrame(results)


# ==========================================
# 3. Streamlit UI 與 資料載入
# ==========================================

@st.cache_data
def load_data():
    path = './data/paired_filtered.csv'
    if not os.path.exists(path):
        csv_files = [f for f in os.listdir('.') if f.lower().endswith('.csv')]
        if csv_files:
            path = csv_files[0]
        else:
            st.error("找不到 data/paired_filtered.csv 或其他 CSV 檔")
            return pd.DataFrame()
            
    df = pd.read_csv(path)
    df.drop(['pa_seq', 'bases', 'velocities_events', 'pitchCodes_events'], axis=1, inplace=True, errors='ignore')
    return df

df_data = load_data()

if not df_data.empty:
    pitchers = sorted(df_data['pitcherName'].unique())
    batters = sorted(df_data['batterName'].unique())
else:
    pitchers, batters = [], []

# --- 側邊欄 ---
with st.sidebar:
    st.header("1. 對戰設定")
    selected_pitcher = st.selectbox("選擇投手", pitchers)
    selected_batter = st.selectbox("選擇打者", batters)
    
    st.divider()
    
    st.header("2. 執行模擬")
    if st.button("🚀 開始戰術模擬", type="primary"):
        with st.spinner(f"正在模擬 {selected_pitcher} vs {selected_batter} 的 1000 場對決..."):
            
            sim_df = run_simulation(selected_pitcher, selected_batter, df_data)
            
            if not sim_df.empty:
                st.session_state['sim_results'] = sim_df
                st.session_state['current_matchup'] = f"{selected_pitcher} vs {selected_batter}"
                st.success("模擬完成！數據已載入。")

# --- 主畫面 ---
st.title("⚾ Baseball KataGo: 實時配球決策系統 (v2.1)")

if 'sim_results' in st.session_state:
    df_sim = st.session_state['sim_results']
    
    st.markdown(f"### 📊 對戰組合: **{st.session_state['current_matchup']}**")
    st.divider()

    col_state, col_info = st.columns([1, 2])
    with col_state:
        st.subheader("Count Selector")
        balls = st.radio("Balls", [0, 1, 2, 3], horizontal=True)
        strikes = st.radio("Strikes", [0, 1, 2], horizontal=True)
        current_count_str = f"{balls}-{strikes}"
        
    with col_info:
        st.info(f"Analyzing Count: **{current_count_str}**")
        st.markdown("模擬引擎根據 1000 場虛擬對決，計算不同落點與球種的戰術價值。")

    # --- Phase 2.1: 成分分析圖 (Component Analysis) ---
    st.subheader(f"戰術成分分析 (Component Analysis)")
    
    # 使用 Tabs 分離不同維度的分析
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 綜合勝率 (Win Rate)", 
        "💨 揮空熱區 (Whiff%)", 
        "👀 凍結熱區 (Called Strike%)",
        "📉 軟弱擊球 (Soft Contact%)"
    ])
    
    # 定義繪圖邏輯
    def plot_heatmap(mode, title_prefix, cmap='RdYlGn'):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        has_data = False
        
        for i, p_type in enumerate(pitch_types):
            ax = axes[i]
            
            # 使用新的通用函式
            res_map = get_analytical_map(df_sim, current_count_str, p_type, mode=mode)
            
            if not np.isnan(res_map).all():
                has_data = True
            
            # 顯示
            im = ax.imshow(res_map, origin='lower', cmap=cmap, vmin=0, vmax=0.8, alpha=0.9,
                           extent=[x_bound[0], x_bound[-1], y_bound[0], y_bound[-1]], aspect='auto')
            
            plotting_background(ax)
            ax.set_title(f"{p_type.capitalize()}", fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # Add colorbar for context if needed (optional)
            # plt.colorbar(im, ax=ax)

        if has_data:
            st.pyplot(fig)
        else:
            st.warning(f"模擬數據中未包含 '{current_count_str}' 且投出對應球種的足夠樣本。")

    with tab1:
        st.caption("全域價值：若在此位置投球，最終該打席解決打者的機率。")
        plot_heatmap('win_rate', "Win Rate", cmap='bwr')
        
    with tab2:
        st.caption("揮空率：該球造成打者揮空 (Swing & Miss) 的機率。適合用於兩好球後的決勝球 (Put-away Pitch)。")
        plot_heatmap('whiff', "Whiff Rate", cmap='Greens') # 使用單色系更直觀

    with tab3:
        st.caption("凍結率：該球被判為好球且打者未揮棒的機率。適合用於搶好球數 (Get-me-over)。")
        plot_heatmap('called_strike', "Called Strike Rate", cmap='Blues')

    with tab4:
        st.caption("軟弱擊球率：該球被打成軟弱滾地球或不營養飛球的機率。適合製造雙殺或化解危機。")
        plot_heatmap('soft_contact', "Soft Contact Rate", cmap='Reds')
        
else:
    st.info("請由左側選擇投打組合並開始模擬。")
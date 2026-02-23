import pandas as pd
import os
import networkx as nx
import numpy as np
import sys
from collections import Counter

sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")

from scripts.pretrain.transitions.compute_transitions_utils import (
    compute_transitions_and_persistence, 
)

from utils.plotting.class_colors import CLOUD_CLASS_INFO

RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
OUT_DIR = f"{BASE_DIR}/pathway_analysis"
os.makedirs(OUT_DIR, exist_ok=True)
SELECTED_CLASSES = [1,2,4]

path = f"{OUT_DIR}/df_for_transition_matrix_no_dom.csv"
df = pd.read_csv(path, low_memory=False)
print(f"Loaded dataframe with {len(df)} rows and {df['storm_id'].nunique()} unique storms")

# ============================================================================
# FILTER: Remove trajectories that don't contain any interesting class
# ============================================================================

def filter_trajectories_with_interesting_classes(df, interesting_classes):
    """
    Remove entire trajectories (storms) that don't contain at least one occurrence 
    of any of the interesting classes.
    """
    # Find storms that have at least one interesting class
    interesting_mask = df['label'].isin(interesting_classes)
    storms_with_interesting = df[interesting_mask]['storm_id'].unique()
    
    # Filter to keep only these storms
    df_filtered = df[df['storm_id'].isin(storms_with_interesting)].copy()
    
    return df_filtered

initial_storms = df['storm_id'].nunique()
df = filter_trajectories_with_interesting_classes(df, SELECTED_CLASSES)
final_storms = df['storm_id'].nunique()
final_rows = len(df)

print(f"Filtered trajectories: {initial_storms} → {final_storms} storms ({initial_storms - final_storms} removed)")
print(f"Remaining rows: {final_rows}")
print()

cloud_items_ordered = sorted(
    CLOUD_CLASS_INFO.items(),
    key=lambda x: x[1]["order"]
)

labels_ordered = [lbl for lbl, _ in cloud_items_ordered]
short_labels = [info["short"] for _, info in cloud_items_ordered]
colors_ordered = [info["color"] for _, info in cloud_items_ordered]
print(colors_ordered)

selected_labels_ordered = [lbl for lbl in labels_ordered if lbl in SELECTED_CLASSES]
selected_short_labels = [short_labels[i] for i, lbl in enumerate(labels_ordered) if lbl in SELECTED_CLASSES]
selected_colors_ordered = [colors_ordered[i] for i, lbl in enumerate(labels_ordered) if lbl in SELECTED_CLASSES]
print(f"Selected labels: {selected_labels_ordered}")
print(f"Selected short labels: {selected_short_labels}")
print(f"Selected colors: {selected_colors_ordered}")

# Compute initial and final states for each trajectory
initial_state = []
final_state = []
for storm_id, g in df.groupby("storm_id"):
    g = g.sort_values("datetime")
    first_row = g.iloc[0]
    end_row = g.iloc[-1]
    base_label_init = first_row["label"]
    base_label_end = end_row["label"]
    
    # Only keep states from selected classes
    if base_label_init not in SELECTED_CLASSES:
        continue
    if base_label_end not in SELECTED_CLASSES:
        continue
    
    # No dominance split - use label directly
    state_init = str(base_label_init)
    state_end = str(base_label_end)
    
    initial_state.append({
        "storm_id": storm_id,
        "initial_state": state_init
    })
    final_state.append({
        "storm_id": storm_id,
        "final_state": state_end
    })

initial_state_counts = Counter([s["initial_state"] for s in initial_state])
final_state_counts = Counter([s["final_state"] for s in final_state])
print("Initial state counts:") #sort them by count descending
for state, count in sorted(initial_state_counts.items(), key=lambda x: -x[1]):
    print(f"{state}: {count}")
print("Final state counts:")
for state, count in sorted(final_state_counts.items(), key=lambda x: -x[1]):
    print(f"{state}: {count}")



#compute transitions and persistence without dominance split
res = compute_transitions_and_persistence(
    df,
    label_col="label",
    time_col="datetime",
    lat_col="lat",
    lon_col="lon",
    labels=selected_labels_ordered,
)

labels = [str(l) for l in res["labels"]]  # Convert to strings: ['1', '2', '4']
P = res["transition_matrix"]  # shape (n, n)

print("Transition Matrix P:")
print(P)
print("Labels:", labels)


# ============================================================================
# GENERATE ALL POSSIBLE PATHWAYS SYSTEMATICALLY
# ============================================================================

def generate_all_pathways(labels):
    """
    Generate all mathematically possible pathways:
    1. Persistence: [X] for each class (same state throughout)
    2. Binary: [X, Y] for all X != Y (two different states)
    3. Trinary: [X, Y, Z] where Y differs from both X and Z (middle state different)
    """
    pathways = []
    
    # 1. PERSISTENCE PATHWAYS (single state)
    for label in labels:
        pathways.append(([label], "persistence"))
    
    # 2. BINARY PATHWAYS (two states, different)
    for i, start in enumerate(labels):
        for j, end in enumerate(labels):
            if start != end:
                pathways.append(([start, end], "binary"))
    
    # 3. TRINARY PATHWAYS (three states, middle differs from both start and end)
    for i, start in enumerate(labels):
        for j, middle in enumerate(labels):
            if middle == start:
                continue  # Middle must differ from start
            for k, end in enumerate(labels):
                if middle == end:
                    continue  # Middle must differ from end
                pathways.append(([start, middle, end], "trinary"))
    
    return pathways

all_pathways = generate_all_pathways(labels)

print("\n" + "="*100)
print(f"GENERATED {len(all_pathways)} POSSIBLE PATHWAYS")
print("="*100)
print(f"  Persistence: {sum(1 for _, t in all_pathways if t == 'persistence')} pathways")
print(f"  Binary:      {sum(1 for _, t in all_pathways if t == 'binary')} pathways")
print(f"  Trinary:     {sum(1 for _, t in all_pathways if t == 'trinary')} pathways")
print("="*100 + "\n")



# ============================================================================
# COMPUTE PATHWAY PROBABILITIES FROM TRANSITION MATRIX
# ============================================================================

def compute_pathway_probability(path, P, labels):
    """
    Compute pathway probability from transition matrix.
    For persistence pathways, return None (computed separately from data).
    """
    if len(path) == 1:
        # Persistence - will be computed from actual trajectories
        return None
    
    prob = 1.0
    for i in range(len(path) - 1):
        state_from = path[i]
        state_to = path[i + 1]
        
        idx_from = labels.index(state_from)
        idx_to = labels.index(state_to)
        
        transition_prob = P[idx_from, idx_to]
        
        if not np.isfinite(transition_prob) or transition_prob == 0:
            return 0.0
        
        prob *= transition_prob
    
    return prob

# Compute probabilities for all pathways
pathway_data = []
for path, pathway_type in all_pathways:
    prob = compute_pathway_probability(path, P, labels)
    pathway_data.append({
        'path': path,
        'type': pathway_type,
        'probability': prob
    })

print("Pathway probabilities computed.\n")

# ============================================================================
# CREATE STATE COLUMN IN DATAFRAME
# ============================================================================

def effective_state(row):
    """Get state based on label only (no dominance)"""
    return str(row["label"])

df["state"] = df.apply(effective_state, axis=1)
#print all states in df with their counts
state_counts = Counter(df["state"])
print("State counts in dataframe:")
for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
    print(f"{state}: {count}")

# ============================================================================
# TRAJECTORY MATCHING FUNCTIONS
# ============================================================================

#funciton to removes states that arenot interesting

def remove_non_interesting_states(states, interesting_states):
    """Remove states that are not in the interesting states set."""
    return [s for s in states if s in interesting_states]


def trajectory_matches_persistence(states, target_state):
    """
    Check if trajectory contains only one state from the interesting classes.
    - Must have target_state
    - Cannot have other interesting states (other than target_state)
    - Can have non-interesting states
    """
    states = remove_non_interesting_states(states, set(labels))  # Keep only interesting states in the trajectory
    interesting_states = set(labels)  # The selected classes as strings
    
    # Get all interesting states present in trajectory
    interesting_states_in_traj = set(s for s in states if s in interesting_states)
    
    # Should only have the target_state from interesting classes
    return interesting_states_in_traj == {target_state}

def trajectory_matches_binary(states, path):
    """
    Match binary pathway [A, B]:
    - Trajectory starts with A (from interesting states)
    - Trajectory ends with B (from interesting states)
    - Only A and B from interesting states are present (can have non-interesting states)
    - At least one transition A->B occurs
    """
    states = remove_non_interesting_states(states, set(labels))  # Keep only interesting states in the trajectory
    if len(path) != 2:
        return False
    
    if not states or len(states) < 2:
        return False
    
    interesting_states = set(labels)  # The selected classes as strings
    
    # Filter to only interesting states in the trajectory
    interesting_states_in_traj = [s for s in states if s in interesting_states]
    
    if not interesting_states_in_traj:
        return False
    
    # Must start with path[0] and end with path[1] (considering only interesting states)
    if interesting_states_in_traj[0] != path[0] or interesting_states_in_traj[-1] != path[1]:
        return False
    
    # Only path states allowed from interesting states
    if any(s not in path for s in interesting_states_in_traj):
        return False
    
    # Must have at least one transition from path[0] to path[1] (considering all states)
    has_transition = any(states[i] == path[0] and states[i+1] == path[1] 
                        for i in range(len(states)-1))
    
    return has_transition

def trajectory_matches_trinary(states, path):
    """
    Match trinary pathway [A, B, C]:
    - Trajectory starts with A (from interesting states)
    - Trajectory ends with C (from interesting states)
    - Only A, B, C from interesting states are present (can have non-interesting states)
    - All three states appear in order (A before B before C in sequence)
    - B must differ from both A and C
    """
    states = remove_non_interesting_states(states, set(labels))  # Keep only interesting states in the trajectory
    if len(path) != 3:
        return False
    
    if not states or len(states) < 3:
        return False
    
    interesting_states = set(labels)  # The selected classes as strings
    
    # Filter to only interesting states in the trajectory
    interesting_states_in_traj = [s for s in states if s in interesting_states]
    
    if not interesting_states_in_traj or len(interesting_states_in_traj) < 3:
        return False
    
    # Must start with path[0] and end with path[2] (considering only interesting states)
    if interesting_states_in_traj[0] != path[0] or interesting_states_in_traj[-1] != path[2]:
        return False
    
    # Only path states allowed from interesting states
    if any(s not in path for s in interesting_states_in_traj):
        return False
    
    # Find if states appear in order: A, then B, then C (in the full trajectory)
    idx_a = -1
    idx_b = -1
    idx_c = -1
    
    for i, s in enumerate(states):
        if s == path[0] and idx_b == -1:  # Still in A phase (before B appears)
            idx_a = i
        elif s == path[1]:  # In B phase
            if idx_a >= 0:  # A has appeared
                idx_b = i
        elif s == path[2]:  # In C phase
            if idx_b >= 0:  # B has appeared
                idx_c = i
                break
    
    # All three states must appear in sequence
    return idx_a >= 0 and idx_b > idx_a and idx_c > idx_b

# ============================================================================
# COUNT STORMS MATCHING EACH PATHWAY
# ============================================================================

print("Counting storms for each pathway...")

for pathway_info in pathway_data:
    path = pathway_info['path']
    pathway_type = pathway_info['type']
    matched_ids = []
    
    for storm_id, g in df.groupby("storm_id"):
        states = g.sort_values("datetime")["state"].tolist()
        
        if pathway_type == "persistence":
            if trajectory_matches_persistence(states, path[0]):
                matched_ids.append(storm_id)
        elif pathway_type == "binary":
            if trajectory_matches_binary(states, path):
                matched_ids.append(storm_id)
        elif pathway_type == "trinary":
            if trajectory_matches_trinary(states, path):
                matched_ids.append(storm_id)
    
    pathway_info['storm_ids'] = matched_ids
    pathway_info['storm_count'] = len(matched_ids)

# ============================================================================
# SORT PATHWAYS BY PROBABILITY (DESCENDING)
# ============================================================================

# Sort by probability (None values at the end)
pathway_data_sorted = sorted(
    pathway_data,
    key=lambda x: (x['probability'] is None, -(x['probability'] if x['probability'] is not None else 0))
)

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "="*110)
print("ALL PATHWAYS WITH PROBABILITIES AND STORM COUNTS")
print("="*110)
print(f"{'Rank':<5} | {'Pathway':<45} | {'Type':<12} | {'Prob':>10} | {'Storms':>6}")
print("-"*110)

rank = 1
for pathway_info in pathway_data_sorted:
    path = pathway_info['path']
    pathway_type = pathway_info['type']
    prob = pathway_info['probability']
    storm_count = pathway_info['storm_count']
    
    # Skip pathways with 0 storms
    if storm_count == 0:
        continue
    
    # Format pathway string with class names
    pathway_str = " -> ".join([
        selected_short_labels[selected_labels_ordered.index(int(s))] 
        for s in path
    ])
    
    # Format probability
    if prob is None:
        prob_str = "N/A"
    elif prob == 0.0:
        prob_str = "0.0000"
    else:
        prob_str = f"{prob:.4f}"
    
    print(f"{rank:<5d} | {pathway_str:<45} | {pathway_type:<12} | {prob_str:>10} | {storm_count:6d}")
    rank += 1

print("-"*110)

# Summary statistics
total_matched_storms = sum(p['storm_count'] for p in pathway_data)
total_storms = df["storm_id"].nunique()
print(f"\nTotal storm-pathway assignments: {total_matched_storms}")
print(f"Total unique storms in dataset: {total_storms}")
print(f"Note: Storms can match multiple pathways if compatible\n")

# Breakdown by type
print("Breakdown by pathway type:")
for ptype in ["persistence", "binary", "trinary"]:
    pathways_of_type = [p for p in pathway_data_sorted if p['type'] == ptype and p['storm_count'] > 0]
    total_storms_type = sum(p['storm_count'] for p in pathways_of_type)
    print(f"  {ptype.capitalize():<12}: {len(pathways_of_type):2d} pathways with storms, {total_storms_type:5d} total matches")

print("="*110 + "\n")

# ============================================================================
# CHECK STORMS WITHOUT PATHWAY ASSIGNMENT
# ============================================================================

# Collect all storms that matched at least one pathway
storms_with_pathways = set()
for pathway_info in pathway_data:
    if pathway_info['storm_count'] > 0:
        storms_with_pathways.update(pathway_info['storm_ids'])

# Get all unique storms in dataset
all_storms = set(df["storm_id"].unique())

# Find storms without any pathway
storms_without_pathway = all_storms - storms_with_pathways

print("="*110)
print(f"STORMS WITHOUT PATHWAY ASSIGNMENT: {len(storms_without_pathway)} out of {len(all_storms)} total storms")
print("="*110)

if storms_without_pathway:
    print(f"\nShowing state sequences for {len(storms_without_pathway)} unassigned storms:\n")
    print(f"{'Storm ID':<50} | {'State Sequence':<55}")
    print("-"*110)
    
    for storm_id in sorted(storms_without_pathway):
        g = df[df["storm_id"] == storm_id].sort_values("datetime")
        states = g["state"].tolist()
        
        state_sequence = " -> ".join(states)
        
        # Truncate if too long
        if len(state_sequence) > 53:
            state_sequence = state_sequence[:50] + "..."
        
        #print(f"{storm_id:<10} | {state_sequence:<100}")
        #print(f"{storm_id:<10} | {len(set(states))} unique states")
        #print all states reovng states consecutively repeated
        states_no_consec = [s for i, s in enumerate(states) if i == 0 or s != states[i-1]]
        print(f"{storm_id:<10} | {' -> '.join(states_no_consec)}")
        #print the hail and rain intesity (max) ofthis storm
        print(f"  Hail intensity: {g['max_hail_intensity'].max():.2f}, Precip intensity: {g['precipitation99'].max():.2f}")
    
    print("-"*110)
    
else:
    print("\nAll storms have been assigned to at least one pathway!")

print("="*110 + "\n")



# ============================================================================
# CREATE MERGED DATAFRAME FOR SAVING
# ============================================================================

# Collect all storms that matched at least one pathway
storms_with_pathways = set()
for pathway_info in pathway_data:
    if pathway_info['storm_count'] > 0:
        storms_with_pathways.update(pathway_info['storm_ids'])

#print yhr storm with the max hail intensity among the storms with pathway assignment
if storms_with_pathways:
    max_hail_storm = None
    max_hail_intensity = -1
    for storm_id in storms_with_pathways:
        g = df[df["storm_id"] == storm_id]
        max_hail_intensity_storm = g['max_hail_intensity'].max()
        if max_hail_intensity_storm > max_hail_intensity:
            max_hail_intensity = max_hail_intensity_storm
            max_hail_storm = storm_id
    print(f"Storm with highest hail intensity: {max_hail_storm} with intensity {max_hail_intensity:.2f}")

#print the storm with the max precip intensity among the storms with pathway assignment
if storms_with_pathways:
    max_precip_storm = None
    max_precip_intensity = -1
    for storm_id in storms_with_pathways:
        g = df[df["storm_id"] == storm_id]
        max_precip_intensity_storm = g['precipitation99'].max()
        if max_precip_intensity_storm > max_precip_intensity:
            max_precip_intensity = max_precip_intensity_storm
            max_precip_storm = storm_id
    print(f"Storm with highest precip intensity: {max_precip_storm} with intensity {max_precip_intensity:.2f}")



# Create dataframe subsets for each pathway
df_pathways_list = []

for i, pathway_info in enumerate(pathway_data_sorted, 1):
    if pathway_info['storm_count'] == 0:
        continue
    
    path = pathway_info['path']
    storm_ids = pathway_info['storm_ids']
    prob = pathway_info['probability'] if pathway_info['probability'] is not None else np.nan
    pathway_type = pathway_info['type']
    
    # Create pathway string
    pathway_str = " -> ".join(path)
    
    # Subset dataframe for these storms
    df_subset = df[df["storm_id"].isin(storm_ids)].copy()
    
    # Add pathway information
    df_subset['pathway_id'] = i
    df_subset['pathway'] = pathway_str
    df_subset['pathway_prob'] = prob
    df_subset['pathway_type'] = pathway_type
    
    df_pathways_list.append(df_subset)

# Merge all pathway dataframes
if df_pathways_list:
    df_pathways_merged = pd.concat(df_pathways_list, ignore_index=True)
    
    # Filter to keep ONLY storms with pathway assignments
    df_pathways_merged = df_pathways_merged[df_pathways_merged['storm_id'].isin(storms_with_pathways)].copy()
    
    print(f"Created merged pathways dataframe with {len(df_pathways_merged)} rows")
    print(f"Total unique storms with pathways: {df_pathways_merged['storm_id'].nunique()}")
    print(f"Columns: {df_pathways_merged.columns.tolist()}")
    
    # Save to CSV
    output_file = os.path.join(OUT_DIR, "df_pathways_merged_no_dominance.csv")
    df_pathways_merged.to_csv(output_file, index=False)
    print(f"\nSaved merged pathways dataframe to: {output_file}")
    print(f"Only storms with pathway assignments were saved ({len(storms_with_pathways)} storms)")
else:
    print("No pathways with matching storms to save.")

print("\n" + "="*110)
print("PATHWAY ANALYSIS COMPLETE")
print("="*110)


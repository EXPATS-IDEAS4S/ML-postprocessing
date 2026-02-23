import pandas as pd
import os
import networkx as nx
import numpy as np
import sys

import matplotlib as mpl

mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["text.usetex"] = False


sys.path.append("/home/Daniele/codes/VISSL_postprocessing/")

from scripts.pretrain.transitions.compute_transitions_utils import (
    compute_transitions_and_persistence_weighted, 
)

from utils.plotting.class_colors import CLOUD_CLASS_INFO

RUN_NAME = "dcv2_resnet_k7_ir108_100x100_2013-2017-2021-2025_2xrandomcrops_1xtimestamp_cma_nc"
BASE_DIR = f"/data1/fig/{RUN_NAME}/epoch_800/test_traj"
OUT_DIR = f"{BASE_DIR}/pathway_analysis"
os.makedirs(OUT_DIR, exist_ok=True)
SELECTED_CLASSES = [1,2,4]

path = f"{OUT_DIR}/df_for_transition_matrix.csv"
df = pd.read_csv(path, low_memory=False)
print(df)

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


#for each trajectory (storm_id) checkthe initial/final crop label and dominance wperc
#loop over storm_id
initial_state = []
final_state = []
for storm_id, g in df.groupby("storm_id"):
    g = g.sort_values("datetime")
    first_row = g.iloc[0]
    end_row = g.iloc[-1]
    base_label_init = first_row["label"]
    base_label_end = end_row["label"]
    wcol_init = f"wperc_label_{base_label_init}"
    wcol_end = f"wperc_label_{base_label_end}"
    if pd.isna(first_row[wcol_init]):
        level = None
    else:
        level = "high" if first_row[wcol_init] >= 75 else "low"
    if pd.isna(end_row[wcol_end]):
        level_end = None
    else:
        level_end = "high" if end_row[wcol_end] >= 75 else "low"
    state_init = f"{base_label_init}_{level}"
    state_end = f"{base_label_end}_{level_end}"
    initial_state.append({
        "storm_id": storm_id,
        "initial_state": state_init
    })
    final_state.append({
        "storm_id": storm_id,
        "final_state": state_end
    })
#print(initial_state)
#make a stat of initial_state
from collections import Counter
initial_state_counts = Counter([s["initial_state"] for s in initial_state])
final_state_counts = Counter([s["final_state"] for s in final_state])
print("Initial state counts:") #sort them by count descending
for state, count in sorted(initial_state_counts.items(), key=lambda x: -x[1]):
    print(f"{state}: {count}")
print("Final state counts:")
for state, count in sorted(final_state_counts.items(), key=lambda x: -x[1]):
    print(f"{state}: {count}")


#compute transitions and persistence splitting the class based on dominance 75% threshold
res = compute_transitions_and_persistence_weighted(
    df,
    label_col="label",
    time_col="datetime",
    lat_col="lat",
    lon_col="lon",
    storm_col="storm_id",
    wperc_prefix="wperc_label_",
    labels=selected_labels_ordered,
    wperc_threshold=75,
    n_bootstrap=100,
    ci_level=95,
)

labels = res["labels"] #['1_low' '1_high' '2_low' '2_high' '4_low' '4_high']
P = res["transition_matrix"]  # shape (n, n)

print(P)
print("Labels:", labels)



def build_transition_graph(P, labels, min_prob=0.0):
    G = nx.DiGraph()
    for i, src in enumerate(labels):
        for j, dst in enumerate(labels):
            if np.isfinite(P[i, j]) and P[i, j] > min_prob:
                cost = -np.log(P[i, j])
                G.add_edge(src, dst, weight=cost, prob=P[i, j])
    return G

G = build_transition_graph(P, labels, min_prob=0.01)
#print(G)

def most_probable_path(G, start, end):
    path = nx.shortest_path(G, start, end, weight="weight")
    prob = np.prod([G[u][v]["prob"] for u, v in zip(path[:-1], path[1:])])
    return path, prob

def k_most_probable_paths(G, start, end, k=3):
    paths = []

    gen = nx.shortest_simple_paths(
        G,
        start,
        end,
        weight="weight"   # this is -log(prob)
    )

    for path in gen:
        prob = np.prod(
            [G[u][v]["prob"] for u, v in zip(path[:-1], path[1:])]
        )
        paths.append((path, prob))
        if len(paths) == k:
            break

    return paths


def most_probable_path_with_cycles(G, start, end, max_len=10):
    best_path = None
    best_logp = -np.inf

    def dfs(path, logp):
        nonlocal best_path, best_logp

        if len(path) > max_len:
            return

        current = path[-1]
        if current == end and len(path) >= 2:
            if logp > best_logp:
                best_logp = logp
                best_path = list(path)

        for _, nxt in G.out_edges(current):
            prob = G[current][nxt].get("prob", 0.0)
            if prob <= 0:
                continue
            dfs(path + [nxt], logp + np.log(prob))

    dfs([start], 0.0)

    if best_path is None:
        return None, None

    return best_path, float(np.exp(best_logp))


#select first 4 initial state and last 4 final states and make all combinations of them to find the most probable path between them
#selct them from initial_state_counts and final_state_counts
initial_states = [s for s, c in initial_state_counts.most_common(4)]
final_states = [s for s, c in final_state_counts.most_common(4)]
print("Initial states:", initial_states)
print("Final states:", final_states)

#make all combinations of initial and final states
from itertools import product
state_pairs = list(product(initial_states, final_states))


paths = []
probs = []
for start, end in state_pairs:
    try:
        paths_probs = k_most_probable_paths(G, start, end, k=5)
        for path, prob in paths_probs:
            if path is not None:
                paths.append(path)
                probs.append(prob)
                print(f"Most probable path from {start} to {end}: {' → '.join(path)} with probability {prob:.3e}")
            else:
                print(f"No path from {start} to {end}")
    except nx.NetworkXNoPath:
        print(f"No path from {start} to {end}")

#order the paths based on descending probability
ordered_paths = sorted(zip(paths, probs), key=lambda x: -x[1])
print("Ordered paths:")
for path, prob in ordered_paths:
    print(f"{' → '.join(path)}: {prob:.3e}")


def effective_state(row, wperc_threshold=75):
    base = row["label"]
    wcol = f"wperc_label_{base}"
    if pd.isna(row[wcol]):
        return None
    level = "high" if row[wcol] >= wperc_threshold else "low"
    return f"{base}_{level}"


df["state"] = df.apply(effective_state, axis=1)


def trajectory_matches_path_old(states, path):
    """Check if path is a subsequence of states"""
    it = iter(states)
    return all(p in it for p in path)

def trajectory_matches_path_strict_ordered(states, path):
    """
    Returns True if:
      - trajectory starts with path[0]
      - trajectory ends with path[-1]
      - all states in path appear at least once
      - they appear in the correct order
      - other states in between are allowed
    """

    if not states or not path:
        return False

    # Strict start and end condition
    if states[0] != path[0]:
        return False
    if states[-1] != path[-1]:
        return False

    path_idx = 0

    for s in states:
        if s == path[path_idx]:
            path_idx += 1
            if path_idx == len(path):
                return True

    return False


def trajectory_matches_path(states, path):
    return states == path


pathway_storms = []

for path, _ in ordered_paths: 
    matched_ids = []

    for storm_id, g in df.groupby("storm_id"):
        states = g.sort_values("datetime")["state"].tolist()
        if trajectory_matches_path_strict_ordered(states, path):
            matched_ids.append(storm_id)

    pathway_storms.append(matched_ids)

for i, ids in enumerate(pathway_storms, 1):
    print(f"Path {i}: {len(ids)} storms")


def subset_for_pathway(df, storm_ids):
    return df[df["storm_id"].isin(storm_ids)].copy()

df_pathways = [
    subset_for_pathway(df, ids)
    for ids in pathway_storms
]

#merge df_pathways to unique df
df_pathways_merged = pd.concat(
    [
        d.assign(
            pathway_id=i + 1,
            pathway=" -> ".join(path),
            pathway_prob=prob
        )
        for i, (d, (path, prob)) in enumerate(zip(df_pathways, ordered_paths))
    ],
    ignore_index=True
)
print(df_pathways_merged.columns.tolist())


#save df_pathways_merged to csv
output_file = os.path.join(OUT_DIR, "df_pathways_merged.csv")
df_pathways_merged.to_csv(output_file, index=False)
print(f"Saved merged pathways dataframe to {output_file}")

